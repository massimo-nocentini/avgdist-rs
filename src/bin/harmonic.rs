//! Estimation of the harmonic-like centrality
//!
//! ```text
//! c(u) = (1 / |S|) * sum_{v in S, v != u} 1 / (1 + d(u, v))
//! ```
//!
//! over a sample `S` of sources, by one breadth-first visit per source. A source
//! that does not reach `u` contributes nothing, which is the `1 / (1 + inf) = 0`
//! convention. `d(u, v)` is a distance in the *original* graph, and passing the
//! transposed basename is the caller's job — the binary cannot tell which it was
//! handed. Every invocation under `data/` passes a `-t` basename, so a forward
//! visit from `v` yields `d(u, v)` at every node `u` it reaches, i.e. as shipped
//! this is the outgoing variant of the measure.
//!
//! ```text
//! harmonic <basename> <num_threads> <epsilon> <exact>
//! ```
//!
//! * `basename` basename of a BvGraph (`.graph`, `.properties`, `.ef`);
//! * `num_threads` size of the Rayon thread pool;
//! * `epsilon` additive error driving the sample size,
//!   `|S| = ceil(log2(|V|) / (2 epsilon^2))`;
//! * `exact` `true` uses every node as a source instead of sampling.
//!
//! Standard output carries **only** one `node<TAB>centrality` line per node with
//! a non-zero centrality, ordered by decreasing centrality, which is the format
//! of the `data/*/result/harmonic.out` files. Graph statistics, progress and
//! timings go through the `log` crate — hence to standard error — driven by a
//! [`dsi_progress_logger`], exactly as the `webgraph` binaries do. Progress is
//! reported at most every [`LOG_INTERVAL`], and only when an item completes: the
//! logger has no background ticker, so the sort between the two phases is
//! silent. `RUST_LOG=debug` additionally logs each individual visit.
//!
//! Nothing large is sent between threads: every visit folds its own
//! contributions straight into one shared array of atomics, so the resident set
//! is one `f64` per node plus one visited bit vector per running visit. Earlier
//! versions shipped a vector holding one entry per reached node through an
//! unbounded channel to a single consumer, which is what made the 2·10⁹-node
//! runs die.

use anyhow::{Context, Result, ensure};
use dsi_progress_logger::prelude::*;
use rand::Rng;
use rand::distributions::Uniform;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::env;
use std::ffi::OsStr;
use std::io::{BufWriter, Write, stdout};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use sux::bits::BitVec;
use webgraph::prelude::*;

/// How often progress is reported.
const LOG_INTERVAL: Duration = Duration::from_secs(30);

/// Adds `delta` to the [`f64`] stored, as bits, in `cell`.
///
/// Every node is reached at most once per visit, so a given cell is written
/// once per source: contention is bounded by the number of visits running at
/// the same instant and the loop below effectively never spins. Accumulating in
/// `f64` rather than in fixed point keeps exactly the arithmetic of the
/// single-threaded version, up to the summation order — which was already
/// unspecified when the contributions arrived over a channel.
#[inline]
fn add_f64(cell: &AtomicU64, delta: f64) {
    let mut current = cell.load(Ordering::Relaxed);
    loop {
        let updated = (f64::from_bits(current) + delta).to_bits();
        match cell.compare_exchange_weak(current, updated, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(actual) => current = actual,
        }
    }
}

/// Visits `graph` breadth-first from `source`, adding `1 / (1 + d(u, source))`
/// to `centralities[u]` for every node `u` it reaches.
///
/// The `seen` bit of a node is set when the node is *enqueued*, so every node is
/// enqueued at most once and the level at which it is enqueued is its distance
/// from `source`. The source itself is marked as seen before the loop and hence
/// never contributes to its own centrality, which is what the sum over `v != u`
/// asks for; it also correctly excludes a source lying on a cycle.
///
/// Returns the eccentricity of `source` and the number of nodes it reaches.
fn bfs<G: RandomAccessGraph>(
    source: usize,
    graph: &G,
    centralities: &[AtomicU64],
) -> (usize, usize) {
    let mut eccentricity = 0;
    let mut reached = 0;
    // `BitVec::new` goes through `vec![0usize; _]`, hence `alloc_zeroed`, hence
    // a fresh anonymous mapping the kernel zeroes page by page on first touch.
    // A visit reaching few nodes therefore pays almost nothing; one reaching the
    // giant component pays the equivalent of a full memset either way, so
    // hoisting the bit vector out would only help if we tracked the words to
    // clear.
    let mut seen = BitVec::new(graph.num_nodes());
    let mut frontier = vec![source];

    seen.set(source, true);

    let mut distance = 0;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        distance += 1;
        let contribution = 1.0 / ((1 + distance) as f64);

        for node in frontier {
            for succ in graph.successors(node) {
                if !seen.get(succ) {
                    seen.set(succ, true);
                    eccentricity = distance;
                    reached += 1;
                    add_f64(&centralities[succ], contribution);
                    next.push(succ);
                }
            }
        }

        frontier = next;
    }

    (eccentricity, reached)
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
            .map_or("harmonic".into(), |arg| arg.to_string_lossy())
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
    ensure!(num_nodes > 0, "The graph has no nodes");

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

    // One `f64`, held as bits, per node. This is the whole of the state shared
    // by the visits.
    let centralities: Vec<AtomicU64> = (0..num_nodes).map(|_| AtomicU64::new(0)).collect();
    let distribution = Uniform::new(0, num_nodes);

    // One shared logger, with a threshold of one so that every completed visit
    // reaches it: a visit is expensive, so the default buffering of
    // `ConcurrentWrapper` would hold updates back for the whole run.
    // `display_memory` is left off on purpose — it reads through `sysinfo`,
    // which can deadlock under Rayon.
    let mut pl = ConcurrentWrapper::with_threshold(1);
    pl.item_name("source")
        .expected_updates(sample_size)
        .log_interval(LOG_INTERVAL)
        .local_speed(true);
    pl.start("Visiting the sampled sources...");

    thread_pool.install(|| {
        (0..sample_size).into_par_iter().for_each_init(
            || (rand::rngs::ThreadRng::default(), pl.clone()),
            |(rng, pl), index| {
                // Sources are drawn with replacement, which keeps the samples
                // independent and hence the estimator unbiased.
                let source = if exact { index } else { rng.sample(distribution) };

                let started = Instant::now();
                let (eccentricity, reached) = bfs(source, &graph, &centralities);
                log::debug!(
                    "((source {source}) (elapsed {:?}) (eccentricity {eccentricity}) (reached {reached}))",
                    started.elapsed()
                );
                pl.update();
            },
        )
    });

    pl.done();

    // c(u) is the mean over the |S| sources, not over the sources that happened
    // to reach u: conditioning on having been reached would rank a node reached
    // once at distance one above a hub reached by every source.
    let normalization = (sample_size as f64).recip();
    let sum = |node: usize| f64::from_bits(centralities[node].load(Ordering::Relaxed));
    // The value actually printed. Sorting on this rather than on the raw sum
    // matters: `x -> x * normalization` is not injective in `f64`, so two nodes
    // whose raw sums differ by one ulp can print the same line, and ordering
    // them by the raw sum would order identical lines by bits that never appear
    // in the output.
    let centrality = |node: usize| sum(node) * normalization;

    // Every contribution is `1 / (1 + d)` with `d >= 1`, hence in `(0, 0.5]`:
    // it cannot underflow to zero, go negative, or become a NaN. So a sum of
    // exactly zero means "reached by no source", and such a node is left out of
    // the output, as in the `data/*/result/harmonic.out` files. Sorting a
    // permutation rather than `(node, centrality)` pairs keeps the scratch space
    // to one `usize` per node.
    let mut order: Vec<usize> = (0..num_nodes).filter(|&node| sum(node) > 0.0).collect();

    if order.is_empty() {
        log::warn!("No node was reached by any source; the output is empty");
        return Ok(());
    }

    log::info!("((nodes reached {}) (of {}))", order.len(), num_nodes);

    // The sort has no per-item hook, so this logger only brackets the phase: no
    // `expected_updates`, which would advertise a percentage that can never be
    // shown. It is the one silent stretch of the run.
    let mut pl = progress_logger![log_interval = LOG_INTERVAL];
    pl.start("Sorting the centralities...");
    thread_pool.install(|| {
        // Descending by centrality, ascending by node within a tie, which is
        // what the original's stable sort over node-ordered pairs produced.
        order.par_sort_unstable_by(|&a, &b| centrality(b).total_cmp(&centrality(a)).then(a.cmp(&b)))
    });
    pl.done_with_count(order.len());

    log::info!(
        "((max {}) (min {}))",
        centrality(order[0]),
        centrality(order[order.len() - 1])
    );

    // `update` rather than `light_update`: the latter consults the clock only
    // every 2^20 items, so any output shorter than that would be written in
    // complete silence, however slow the consumer.
    let mut pl = progress_logger![
        item_name = "node",
        expected_updates = order.len(),
        log_interval = LOG_INTERVAL,
    ];
    pl.start("Writing the centralities...");

    let mut out = BufWriter::with_capacity(1 << 20, stdout().lock());
    for &node in &order {
        writeln!(out, "{}\t{}", node, centrality(node))
            .context("Could not write the centralities on standard output")?;
        pl.update();
    }
    out.flush()
        .context("Could not flush the centralities on standard output")?;

    pl.done();

    Ok(())
}
