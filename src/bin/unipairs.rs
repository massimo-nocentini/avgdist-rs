//! Estimation of the average distance between connected pairs of nodes, and of
//! a lower bound on the diameter, by sampling uniformly distributed connected
//! pairs.
//!
//! ```text
//! unipairs <basename> <num_threads> <epsilon> <exact>
//! ```
//!
//! * `basename` basename of a BvGraph (`.graph`, `.properties`, `.ef`);
//! * `num_threads` size of the Rayon thread pool;
//! * `epsilon` additive error driving the sample size,
//!   `|S| = ceil(log2(|V|) / (2 epsilon^2))`;
//! * `exact` `true` visits every node instead of sampling.
//!
//! Standard output carries **only** the final estimate,
//!
//! ```text
//! ((average distance 31.201229) (diameter 25456))
//! ```
//!
//! so that `unipairs ... > result.out` produces a file a Lisp reader can slurp.
//! Graph statistics, progress and timings go through the `log` crate — hence to
//! standard error — driven by a [`dsi_progress_logger`], exactly as the
//! `webgraph` binaries do. Progress is reported at most every [`LOG_INTERVAL`],
//! and only when a sample completes: the logger has no background ticker, so in
//! sampled mode a long run of rejected draws is silent. `RUST_LOG=debug` logs
//! every accepted sample and every rejection, which is what earlier versions
//! used to interleave with the result on standard output.

use anyhow::{Context, Result, ensure};
use dsi_progress_logger::prelude::*;
use rand::Rng;
use rand::distributions::Uniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::env;
use std::ffi::OsStr;
use std::io::{Write, stdout};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use sux::bits::BitVec;
use webgraph::prelude::*;

/// How often progress is reported.
const LOG_INTERVAL: Duration = Duration::from_secs(30);

/// How many draws a single sample may reject before the run is given up.
///
/// A sample needing more than this many visits means the connected pairs are so
/// rare that the run would not finish anyway; failing with a message beats
/// spinning silently, which is what the unbounded loop used to do.
const MAX_DRAWS_PER_SAMPLE: usize = 1 << 20;

/// How often rejections are logged, in draws.
const REJECTION_LOG_MASK: usize = (1 << 12) - 1;

/// Outcome of a single breadth-first visit.
struct Visit {
    /// Eccentricity of the source: the largest distance from it.
    eccentricity: usize,
    /// Sum of the distances from the source to the nodes it reaches.
    distance_sum: usize,
    /// Number of nodes reached, the source excluded.
    reached: usize,
    /// The nodes reached, the source included.
    seen: BitVec,
}

/// Visits `graph` breadth-first from `start`.
///
/// The `seen` bit of a node is set when the node is *enqueued*, so every node is
/// enqueued at most once and the level at which it is enqueued is its distance
/// from `start`. The source is marked as seen but is deliberately not counted,
/// so `reached` and `distance_sum` range over the reachable set minus `start`
/// itself, and `distance_sum / reached` is the mean distance from `start`.
fn bfs<G: RandomAccessGraph>(start: usize, graph: &G) -> Visit {
    let mut eccentricity = 0;
    let mut distance_sum = 0;
    let mut reached = 0;
    // `BitVec::new` goes through `vec![0usize; _]`, hence `alloc_zeroed`, hence
    // a fresh anonymous mapping the kernel zeroes page by page on first touch.
    // A visit reaching few nodes therefore pays almost nothing; one reaching the
    // giant component pays the equivalent of a full memset either way, so
    // hoisting the bit vector out would only help if we tracked the words to
    // clear.
    let mut seen = BitVec::new(graph.num_nodes());
    let mut frontier = vec![start];

    seen.set(start, true);

    let mut distance = 0;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        distance += 1;

        for node in frontier {
            for succ in graph.successors(node) {
                if !seen.get(succ) {
                    seen.set(succ, true);
                    eccentricity = distance;
                    reached += 1;
                    distance_sum += distance;
                    next.push(succ);
                }
            }
        }

        frontier = next;
    }

    Visit {
        eccentricity,
        distance_sum,
        reached,
        seen,
    }
}

/// Running totals over the samples.
///
/// The sums are widened to [`u128`] because in exact mode `distance_sum` is
/// added once per node: on a graph with 3·10⁸ nodes and diameter 2.5·10⁴ the
/// total overruns [`u64`], and this crate builds with overflow checks off.
#[derive(Clone, Copy, Default)]
struct Totals {
    /// Largest eccentricity seen, i.e. the lower bound on the diameter.
    diameter: usize,
    /// Sum, over the samples, of the distances from the source.
    distance_sum: u128,
    /// Sum, over the samples, of the number of nodes reached.
    reached: u128,
    /// Sum, over the samples, of the mean distance from the source.
    mean_sum: f64,
}

impl Totals {
    fn merge(self, other: Self) -> Self {
        Self {
            diameter: self.diameter.max(other.diameter),
            distance_sum: self.distance_sum + other.distance_sum,
            reached: self.reached + other.reached,
            mean_sum: self.mean_sum + other.mean_sum,
        }
    }
}

/// Draws the `index`-th sample.
///
/// In exact mode the source is `index` itself. Otherwise a pair `(v, w)` of
/// distinct nodes is drawn uniformly and `v` is accepted only if it reaches
/// `w`; a source is therefore accepted with probability `reached(v) / (|V| - 1)`,
/// which is exactly the weighting that makes the mean of the per-source means
/// an unbiased estimator of the average distance over connected pairs. Because
/// acceptance requires `w` to be reached, an accepted sample always has
/// `reached >= 1`, so the mean below is never a division by zero.
fn sample<G: RandomAccessGraph>(
    index: usize,
    graph: &G,
    exact: bool,
    distribution: &Uniform<usize>,
    rng: &mut impl Rng,
    rejections: &AtomicUsize,
) -> Result<Totals> {
    let started = Instant::now();

    let visit = if exact {
        bfs(index, graph)
    } else {
        let mut draws = 0;

        loop {
            draws += 1;
            ensure!(
                draws <= MAX_DRAWS_PER_SAMPLE,
                "Could not draw a connected pair in {MAX_DRAWS_PER_SAMPLE} attempts: \
                 the graph may have too few connected pairs to sample"
            );

            let v = rng.sample(distribution);
            let w = rng.sample(distribution);

            if v == w {
                continue;
            }

            let visit = bfs(v, graph);

            if visit.seen.get(w) {
                break visit;
            }

            let total = rejections.fetch_add(1, Ordering::Relaxed) + 1;
            if total & REJECTION_LOG_MASK == 0 {
                log::debug!("((rejected {total}) (last source {v}) (reached {}))", visit.reached);
            }
        }
    };

    // In exact mode a sink reaches nobody; it contributes nothing to the mean,
    // and `distance_sum / reached` would be a NaN, so it is skipped.
    let mean = if visit.reached > 0 {
        (visit.distance_sum as f64) / (visit.reached as f64)
    } else {
        0.0
    };

    log::debug!(
        "((avgdist {:.6}) (elapsed {:?}) (eccentricity {}) (reached {}))",
        mean,
        started.elapsed(),
        visit.eccentricity,
        visit.reached
    );

    Ok(Totals {
        diameter: visit.eccentricity,
        distance_sum: visit.distance_sum as u128,
        reached: visit.reached as u128,
        mean_sum: mean,
    })
}

/// Returns `arg` as a string, or an error naming it.
///
/// Arguments are collected with [`env::args_os`] because [`env::args`] panics on
/// a non-UTF-8 argument, and `<basename>` is a path.
fn utf8<'a>(arg: &'a OsStr, name: &str) -> Result<&'a str> {
    arg.to_str()
        .with_context(|| format!("<{name}> is not valid UTF-8: {arg:?}"))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stderr)
        .try_init()?;

    let args: Vec<_> = env::args_os().collect();
    ensure!(
        args.len() == 5,
        "Usage: {} <basename> <num_threads> <epsilon> <exact>",
        args.first()
            .map_or("unipairs".into(), |arg| arg.to_string_lossy())
    );

    let basename = PathBuf::from(&args[1]);
    let num_threads: usize = utf8(&args[2], "num_threads")?
        .parse()
        .with_context(|| format!("Could not parse <num_threads> from {:?}", args[2]))?;
    let epsilon: f64 = utf8(&args[3], "epsilon")?
        .parse()
        .with_context(|| format!("Could not parse <epsilon> from {:?}", args[3]))?;
    let exact: bool = utf8(&args[4], "exact")?.parse().with_context(|| {
        format!("Could not parse <exact> from {:?}, expected true or false", args[4])
    })?;

    ensure!(num_threads > 0, "<num_threads> must be positive");
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "<epsilon> must be a positive finite number"
    );

    let thread_pool = rayon::ThreadPoolBuilder::default()
        .num_threads(num_threads)
        .build()
        .context("Could not create the thread pool")?;

    let graph = BvGraph::with_basename(&basename)
        // Hints to `madvise`: the visits jump all over the mapping.
        .flags(MemoryFlags::RANDOM_ACCESS | MemoryFlags::TRANSPARENT_HUGE_PAGES)
        .load()
        .with_context(|| format!("Could not load the graph {}", basename.display()))?;

    let num_nodes = graph.num_nodes();
    ensure!(
        num_nodes > 1,
        "The graph has {num_nodes} nodes: at least two are needed to sample a pair"
    );

    // The size is checked as a float: too small an epsilon overflows to
    // infinity, and the cast to `usize` would then saturate to `usize::MAX`.
    let sample_size = if exact {
        num_nodes
    } else {
        let size = (num_nodes as f64).log2() / (2.0 * epsilon.powi(2));
        ensure!(
            size.is_finite() && size >= 1.0 && size < usize::MAX as f64,
            "<epsilon> = {epsilon:e} yields a sample size of {size:e}, which is not usable"
        );
        size.ceil() as usize
    };

    log::info!(
        "((|V| {}) (|E| {}) (|S| {}) (s {}) (exact {}))",
        num_nodes,
        graph.num_arcs(),
        sample_size,
        thread_pool.current_num_threads(),
        exact
    );

    let distribution = Uniform::new(0, num_nodes);
    let rejections = AtomicUsize::new(0);

    // One shared logger, with a threshold of one so that every completed sample
    // reaches it: the sample count is small and a visit is expensive, so the
    // default buffering of `ConcurrentWrapper` would hold updates back for the
    // whole run. `display_memory` is left off on purpose — it reads through
    // `sysinfo`, which can deadlock under Rayon.
    let mut pl = ConcurrentWrapper::with_threshold(1);
    pl.item_name("sample")
        .expected_updates(sample_size)
        .log_interval(LOG_INTERVAL)
        .local_speed(true);
    pl.start("Sampling connected pairs...");

    let totals = thread_pool.install(|| {
        (0..sample_size)
            .into_par_iter()
            .map_init(
                || (rand::rngs::ThreadRng::default(), pl.clone()),
                |(rng, pl), index| -> Result<Totals> {
                    let totals = sample(index, &graph, exact, &distribution, rng, &rejections)?;
                    pl.update();
                    Ok(totals)
                },
            )
            .try_reduce(Totals::default, |a, b| Ok(a.merge(b)))
    })?;

    pl.done();

    let rejected = rejections.load(Ordering::Relaxed);
    if !exact {
        log::info!(
            "((accepted {}) (rejected {}) (acceptance {:.6}))",
            sample_size,
            rejected,
            (sample_size as f64) / ((sample_size + rejected) as f64)
        );
    }

    // In exact mode the sources are the nodes themselves, so the sums are the
    // exact totals over all connected ordered pairs. In sampled mode the
    // sources are already drawn proportionally to the size of their reachable
    // set, so the estimate is the plain mean of the per-source means — dividing
    // the summed distances by the summed counts would apply that weight twice.
    let average_distance = if exact {
        ensure!(
            totals.reached > 0,
            "No node reaches any other node: the average distance is undefined"
        );
        (totals.distance_sum as f64) / (totals.reached as f64)
    } else {
        totals.mean_sum / (sample_size as f64)
    };

    let mut out = stdout().lock();
    writeln!(
        out,
        "((average distance {:.6}) (diameter {}))",
        average_distance, totals.diameter
    )
    .context("Could not write the estimate on standard output")?;
    out.flush()
        .context("Could not flush the estimate on standard output")?;

    Ok(())
}
