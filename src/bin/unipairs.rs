use rand::rngs::ThreadRng;
use rand::Rng;
use std::io::{self, Write};
use std::ops::Div;
use std::sync::{Arc, Mutex};
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
                    diameter = diameter.max(ll);
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
    _: &mut ThreadRng,
) -> Vec<usize> {
    let num_nodes = agraph.num_nodes();

    let (tx, rx) = std::sync::mpsc::channel();

    for _ in 0..k {
        let agraph = agraph.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let instant = Instant::now();

            let mut r = rand::thread_rng();
            let mut v = 0usize;
            let mut connected = false; // still to be proved.

            while !connected {
                v = r.gen_range(0..num_nodes);
                let mut w = r.gen_range(0..num_nodes);

                if v == w {
                    w = (v + 1) % num_nodes;
                }

                let (_, _, _, seen) = bfs(v, &agraph);

                connected = seen.get(w);
            }

            tx.send(v).unwrap();

            print!(">: {:?} | ", instant.elapsed());
            io::stdout().flush().unwrap();
        });
    }

    drop(tx);

    let mut sampled = vec![0usize; k];

    while let Ok(v) = rx.recv() {
        sampled.push(v);
    }

    sampled
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let mut slot: usize = args[2].parse().unwrap();
    let epsilon: f64 = args[3].parse().unwrap();
    let truth: bool = args[4].parse().unwrap();
    let dummy: bool = args[5].parse().unwrap();

    let mut r = rand::thread_rng();
    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let num_nodes = graph.num_nodes();
    let k = (num_nodes as f64).log2().div(2.0 * epsilon.powi(2)).ceil() as usize;

    println!(
        "|V| = {}, |E| = {}, |S| = {}, s = {}.",
        num_nodes,
        graph.num_arcs(),
        k,
        slot
    );

    let ag = Arc::new(graph);

    let mut averages_dist = Vec::new();
    let mut averages_diameter = Vec::new();

    let mut remaining = k;
    let mut iteration = 1usize;

    let instant = Instant::now();

    while remaining > 0 {
        slot = slot.min(remaining);

        println!(
            "\n*** iteration {}, batch size {}, remaining {}.",
            iteration,
            slot,
            remaining - slot
        );

        let instant = Instant::now();
        let sampled: Vec<usize> = if truth {
            slot = remaining;
            (0..num_nodes).collect()
        } else {
            if dummy {
                (0..slot).map(|_j| r.gen_range(0..num_nodes)).collect()
            } else {
                sample(slot, ag.clone(), &mut r)
            }
        };
        println!("sampled in {:?}", instant.elapsed());

        let triple = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
        let mut handles = Vec::new();
        let instant = Instant::now();
        for v in sampled {
            let ag = ag.clone();
            let triple = triple.clone();

            let handle = thread::spawn(move || {
                let instant = Instant::now();
                let (dia, sum, count, _) = bfs(v, &ag);
                {
                    let mut tx = triple.lock().unwrap();

                    tx.0 = tx.0.max(dia);
                    tx.1 += sum;
                    tx.2 += count;
                }
                print!("<: {:?} | ", instant.elapsed());
                io::stdout().flush().unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let (dia, sum, count) = triple.lock().unwrap().clone();

        println!("bfses in {:?}", instant.elapsed());

        if count > 0 {
            let adist = (sum as f64) / ((count * (num_nodes - 1)) as f64);
            let adia = dia as f64;

            println!("\naverages: distance {:.3}, diameter {:.3}.", adist, adia);

            averages_dist.push(adist);
            averages_diameter.push(adia);
        }

        let n = averages_dist.len() as f64;
        let avgdist: f64 = averages_dist.iter().sum::<f64>() / n;
        let avgdia: f64 = averages_diameter.iter().sum::<f64>() / n;

        let avgdist_var = averages_dist
            .iter()
            .map(|x| (x - avgdist).powi(2))
            .sum::<f64>()
            / (n - 1.0);

        let avgdia_var = averages_diameter
            .iter()
            .map(|x| (x - avgdia).powi(2))
            .sum::<f64>()
            / (n - 1.0);

        println!(
            "average of averages: distance {:.9} (norm {:.3}), std {:.9} (norm {:.3}), diameter {:.3} (std {:.3}).",
            avgdist,
            avgdist * ((num_nodes - 1) as f64),
            avgdist_var.sqrt(),
            avgdist_var.sqrt() * ((num_nodes - 1) as f64),
            avgdia,
            avgdia_var.sqrt()
        );

        remaining -= slot;
        iteration += 1;
    }

    println!("\nTotal time: {:?}", instant.elapsed());
}
