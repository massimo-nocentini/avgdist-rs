use rand::Rng;
use std::env;
use std::io::{self, Write};
use std::ops::{Div, SubAssign};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;
use sux::bits::BitVec;
use webgraph::prelude::*;

fn bfs<T: RandomAccessGraph>(start: usize, graph: &Arc<T>) -> (usize, usize, usize, BitVec) {
    let mut frontier = Vec::new();
    let mut distance = 0usize;
    let mut diameter = 0usize;
    let mut count = 0usize;
    let mut seen = BitVec::new(graph.num_nodes());

    seen.set(start, true);

    frontier.push((start, 0));

    while !frontier.is_empty() {
        let mut frontier_next = Vec::new();

        for (current_node, l) in frontier {
            let ll = l + 1;

            for succ in graph.successors(current_node) {
                if !seen.get(succ) {
                    diameter = std::cmp::max(diameter, ll);
                    seen.set(succ, true);
                    count += 1;
                    distance += ll;
                    frontier_next.push((succ, ll));
                }
            }
        }

        frontier = frontier_next;
    }

    (diameter, distance, count, seen)
}

fn sample<T: RandomAccessGraph + Send + Sync + 'static>(
    thread_pool: &rayon::ThreadPool,
    k: usize,
    agraph: Arc<T>,
    exact_computation: bool,
) -> (usize, usize, usize, f64, usize) {
    let (tx, rx) = std::sync::mpsc::channel();

    let num_nodes = agraph.num_nodes();
    let remaining = Arc::new(AtomicUsize::new(k));
    let distr = rand::distributions::Uniform::new(0, num_nodes);

    for each in 0..k {
        let agraph = agraph.clone();
        let tx = tx.clone();
        let remaining = remaining.clone();

        thread_pool.spawn(move || {
            let instant = Instant::now();

            let current_avgdist = if exact_computation {
                let (dia, dist, count, _seen) = bfs(each, &agraph);
                tx.send((dia, dist, count, each)).unwrap();
                (dist as f64) / (count as f64)
            } else {
                let mut r = rand::rngs::ThreadRng::default();

                loop {
                    let v = r.sample(distr);
                    let w = r.sample(distr);

                    if v == w {
                        continue;
                    }

                    let (dia, dist, count, seen) = bfs(v, &agraph);

                    if seen.get(w) {
                        tx.send((dia, dist, count, v)).unwrap();

                        break (dist as f64) / (count as f64);
                    }
                }
            };

            {
                let rem = remaining.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

                println!(
                    "((avgdist {:.6}) (eta {:?}) (remaining {}))",
                    current_avgdist,
                    instant.elapsed(),
                    rem
                );
                io::stdout().flush().unwrap();
            }
        });
    }

    drop(tx);

    let mut tx = (0usize, 0usize, 0usize, 0.0, 0);

    while let Ok((dia, sum, count, _v)) = rx.recv() {
        tx.0 = std::cmp::max(tx.0, dia);
        tx.1 += sum;
        tx.2 += count;
        if count > 0 {
            tx.3 += (sum as f64) / (count as f64);
            tx.4 += 1;
        }
    }

    tx
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    // let mut slot: usize = args[2].parse().unwrap();
    let epsilon: f64 = args[3].parse().unwrap();
    let exact_computation: bool = args[4].parse().unwrap();

    let thread_pool = rayon::ThreadPoolBuilder::default()
        .build()
        .expect("Failed to create thread pool");

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let num_nodes = graph.num_nodes();
    let k = (num_nodes as f64).log2().div(2.0 * epsilon.powi(2)).ceil() as usize;

    let sample_size = if exact_computation { num_nodes } else { k };

    println!(
        "((|V| {}) (|E| {}) (|S| {}) (s {}))",
        num_nodes,
        graph.num_arcs(),
        sample_size,
        thread_pool.current_num_threads()
    );

    let ag = Arc::new(graph);

    let mut D = 0usize; // maximum diameter
    let mut S = 0usize; // sum of distances
    let mut C = 0usize; // count of pairs
    let mut R = 0.0; // ratio of pairs
    let mut Rc = 0usize; // count of pairs

    let instant = Instant::now();

    let (dia, sum, count, ratio, c) =
        sample(&thread_pool, sample_size, ag.clone(), exact_computation);

    D = std::cmp::max(D, dia);
    S += sum;
    C += count;
    R += ratio;
    Rc += c;

    println!(
        "\n((average distance {:.6}) (diameter {}) (eta {:?}))",
        if exact_computation {
            (S as f64) / (C as f64)
        } else {
            R / (sample_size as f64)
        },
        D,
        instant.elapsed()
    );
}
