use rand::Rng;
use std::io::{self, Write};
use std::ops::Div;
use std::sync::Arc;
use std::time::Instant;
use std::{env, thread};
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

            for (_i, succ) in graph.successors(current_node).into_iter().enumerate() {
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
    k: usize,
    agraph: Arc<T>,
    exact_computation: bool,
) -> (usize, usize, usize, f64, usize) {
    let (tx, rx) = std::sync::mpsc::channel();

    let thread_pool = rayon::ThreadPoolBuilder::default()
        .build()
        .expect("Failed to create thread pool");

    for each in 0..k {
        let agraph = agraph.clone();
        let tx = tx.clone();

        thread_pool.install(|| {
            let instant = Instant::now();
            let mut r = rand::thread_rng();

            if exact_computation {
                let (dia, dist, count, _seen) = bfs(each, &agraph);
                tx.send((dia, dist, count, each)).unwrap();
            } else {
                let num_nodes = agraph.num_nodes();

                loop {
                    let v = r.gen_range(0..num_nodes);
                    let w = r.gen_range(0..num_nodes);

                    if v == w {
                        continue;
                    }

                    let (dia, dist, count, seen) = bfs(v, &agraph);

                    if seen.get(w) {
                        tx.send((dia, dist, count, v)).unwrap();
                        break;
                    }
                }
            }

            print!(">: {:?} | ", instant.elapsed());
            io::stdout().flush().unwrap();
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
    let mut slot: usize = args[2].parse().unwrap();
    let epsilon: f64 = args[3].parse().unwrap();
    let exact_computation: bool = args[4].parse().unwrap();

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let num_nodes = graph.num_nodes();
    let k = (num_nodes as f64).log2().div(2.0 * epsilon.powi(2)).ceil() as usize;
    
    let sample_size = if exact_computation { num_nodes } else { k };

    println!(
        "|V| = {}, |E| = {}, |S| = {}, s = {}.",
        num_nodes,
        graph.num_arcs(),
        sample_size,
        slot
    );

    let ag = Arc::new(graph);

    let mut remaining = sample_size;
    let mut iteration = 1usize;

    let mut D = 0usize; // maximum diameter
    let mut S = 0usize; // sum of distances
    let mut C = 0usize; // count of pairs
    let mut R = 0.0; // ratio of pairs
    let mut Rc = 0usize; // count of pairs

    let instant = Instant::now();

    slot = if exact_computation { remaining } else { k };

    println!(
        "\n*** iteration {}, batch size {}, remaining {}.",
        iteration,
        slot,
        remaining - slot
    );

    let instant = Instant::now();
    let (dia, sum, count, ratio, c) = sample(slot, ag.clone(), exact_computation);

    D = std::cmp::max(D, dia);
    S += sum;
    C += count;
    R += ratio;
    Rc += c;

    println!("sampled in {:?}", instant.elapsed());

    remaining -= slot;
    iteration += 1;

    println!(
        "\n((average distance {:.6}) (diameter {}) (eta {:?}))",
        if exact_computation {
            (S as f64) / (C as f64)
        } else {
            (R / (sample_size as f64))
        },
        D,
        instant.elapsed()
    );
}
