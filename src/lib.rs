use rand::Rng;
use std::io::{self, Write};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;
use sux::bits::BitVec;
use webgraph::prelude::*;

fn avgdist_bfs<T: RandomAccessGraph>(
    start: usize,
    graph: &Arc<T>,
) -> (usize, usize, usize, BitVec) {
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

pub fn avgdist_sample<T: RandomAccessGraph + Send + Sync + 'static>(
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
                let (dia, dist, count, _seen) = avgdist_bfs(each, &agraph);
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

                    let (dia, dist, count, seen) = avgdist_bfs(v, &agraph);

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

fn hc_bfs<T: RandomAccessGraph>(
    start: usize,
    graph: &Arc<T>,
) -> (usize, usize, usize, BitVec, Vec<(usize, usize)>) {
    let mut frontier = Vec::new();
    let mut distance = 0usize;
    let mut diameter = 0usize;
    let mut count = 0usize;
    let mut seen = BitVec::new(graph.num_nodes());
    let mut finite_distances = Vec::new();

    seen.set(start, true);

    frontier.push((start, 0));

    while !frontier.is_empty() {
        let mut frontier_next = Vec::new();

        for (current_node, l) in frontier {
            let ll = l + 1;

            for (_i, succ) in graph.successors(current_node).into_iter().enumerate() {
                if !seen.get(succ) {
                    diameter = std::cmp::max(diameter, ll);
                    seen.set(succ, true);
                    count += 1;
                    distance += ll;
                    frontier_next.push((succ, ll));
                    finite_distances.push((succ, ll));
                }
            }
        }

        frontier = frontier_next;
    }

    (diameter, distance, count, seen, finite_distances)
}

pub fn hc_sample<T: RandomAccessGraph + Send + Sync + 'static>(
    thread_pool: &rayon::ThreadPool,
    sample_size: usize,
    agraph: Arc<T>,
    exact_computation: bool,
) -> (usize, usize, usize, Vec<usize>, Vec<Option<f64>>) {
    let num_nodes = agraph.num_nodes();
    let distr = rand::distributions::Uniform::new(0, num_nodes);
    let (tx, rx) = std::sync::mpsc::channel();

    for each in 0..sample_size {
        let agraph = agraph.clone();
        let tx = tx.clone();
        thread_pool.spawn(move || {
            let instant = Instant::now();

            let vertex = if exact_computation {
                each
            } else {
                rand::rngs::ThreadRng::default().sample(distr)
            };

            let (dia, dist, count, _seen, finite_dist) = hc_bfs(vertex, &agraph);
            tx.send((dia, dist, count, vertex, finite_dist)).unwrap();

            print!(">: {:?} | ", instant.elapsed());
            io::stdout().flush().unwrap();
        });
    }

    drop(tx);

    let mut tx = (0usize, 0usize, 0usize);

    let mut sizes = vec![0usize; num_nodes];
    let mut finite_ds = vec![None; num_nodes];

    while let Ok((dia, sum, count, _v, finite_dist)) = rx.recv() {
        tx.0 = std::cmp::max(tx.0, dia);
        tx.1 += sum;
        tx.2 += count;

        for (node, dist) in finite_dist {
            sizes[node] += 1;

            let dist_inv = 1.0 / ((1 + dist) as f64);

            finite_ds[node] = match finite_ds[node] {
                None => Some(dist_inv),
                Some(existing_dist) => Some(existing_dist + dist_inv),
            };
        }
    }

    (tx.0, tx.1, tx.2, sizes, finite_ds)
}
