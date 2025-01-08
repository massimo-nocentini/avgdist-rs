use rand::rngs::ThreadRng;
use rand::Rng;
use std::io::{self, Write};
use std::ops::{Deref, Div};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{env, thread};
use sux::bits::BitVec;
use webgraph::prelude::*;
use webgraph_algo::prelude::breadth_first::EventNoPred;
use webgraph_algo::traits::Parallel;

use dsi_progress_logger::no_logging;
use std::ops::ControlFlow::Continue;
use webgraph::prelude::BvGraph;
use webgraph::traits::RandomAccessGraph;
use webgraph::traits::SequentialLabeling;
use webgraph_algo::threads;

fn visit<G>(root: usize, graph: G) -> (BitVec, usize, usize, usize)
where
    G: RandomAccessGraph + Send + Sync,
{
    let num_nodes = graph.num_nodes();
    let mut distance = 0usize;
    let mut diameter = 0usize;
    let mut count = 0usize;
    let mut d = Vec::with_capacity(num_nodes);

    for _ in 0..num_nodes {
        d.push(AtomicUsize::new(0usize));
    }

    let mut visit = webgraph_algo::algo::visits::breadth_first::ParFairNoPred::new(graph, 1);

    visit
        .par_visit(
            root,
            |event| {
                if let EventNoPred::Unknown { curr, distance, .. } = event {
                    d[curr].store(distance, Ordering::Relaxed);
                }

                Continue::<G>(())
            },
            &threads![],
            no_logging![],
        )
        .continue_value();

    let mut seen = BitVec::new(num_nodes);

    for (v, au) in d.iter().enumerate() {
        let dist = au.load(Ordering::Relaxed);

        if dist < 1 {
            continue;
        }

        distance += dist;
        count += 1;
        diameter = diameter.max(dist);

        seen.set(v, true);
    }

    (seen, diameter, distance, count)
}

fn bfs(
    start: usize,
    channel: Sender<BitVec>,
    graph_filename: Arc<String>,
) -> (usize, usize, usize) {
    let graph = BvGraph::with_basename(graph_filename.deref())
        .load()
        .unwrap();

    let (seen, diameter, distance, count) = visit(start, graph);

    channel.send(seen).unwrap();

    (diameter, distance, count)
}

fn sample(k: usize, num_nodes: usize, agraph: Arc<String>, r: &mut ThreadRng) -> Vec<usize> {
    let (tx, rx) = std::sync::mpsc::channel();

    for _ in 0..k {
        let v = r.gen_range(0..num_nodes);
        let agraph = agraph.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let instant = Instant::now();
            bfs(v, tx, agraph);
            print!(">: {:?} | ", instant.elapsed());
            io::stdout().flush().unwrap();
        });
    }

    drop(tx);

    let mut cross = vec![0usize; num_nodes];

    while let Ok(seen) = rx.recv() {
        seen.iter_ones().for_each(|v| cross[v] += 1);
    }

    for i in 1..num_nodes {
        cross[i] += cross[i - 1];
    }

    let maxc = *cross.last().unwrap();

    let mut sampled = vec![0usize; k];

    for i in 0..k {
        let c = r.gen_range(0..=maxc);
        let mut l = 0usize;
        let mut h = num_nodes - 1;

        while l < h {
            let m = (h + l) >> 1;
            if cross[m] < c {
                l = m + 1;
            } else {
                h = m;
            }
        }

        sampled[i] = l;
    }

    sampled
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = args[1].clone();
    let graph_filename_t = args[2].clone();
    let mut slot: usize = args[3].parse().unwrap();
    let epsilon: f64 = args[4].parse().unwrap();
    let truth: bool = args[5].parse().unwrap();
    let dummy: bool = args[6].parse().unwrap();

    let mut r = rand::thread_rng();
    let graph = BvGraph::with_basename(&graph_filename).load().unwrap();

    let num_nodes = graph.num_nodes();
    let k = (num_nodes as f64).log2().div(epsilon.powi(2)).ceil() as usize;

    println!(
        "|V| = {}, |E| = {}, |S| = {}, s = {}.",
        num_nodes,
        graph.num_arcs(),
        k,
        slot
    );

    let ag = Arc::new(graph_filename);
    let ag_t = Arc::new(graph_filename_t);

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
                sample(slot, num_nodes, ag_t.clone(), &mut r)
            }
        };
        println!("sampled in {:?}", instant.elapsed());

        let triple = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
        let mut handles = Vec::new();
        let instant = Instant::now();
        let (tx, _rx) = std::sync::mpsc::channel();
        for v in sampled {
            let ag = ag.clone();
            let triple = triple.clone();
            let tx = tx.clone();
            let handle = thread::spawn(move || {
                let instant = Instant::now();
                let (dia, sum, count) = bfs(v, tx, ag);
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

        let adist = (sum as f64) / (count as f64);
        let adia = dia as f64;

        println!("\naverages: distance {:.3}, diameter {:.3}.", adist, adia);

        averages_dist.push(adist);
        averages_diameter.push(adia);

        let avgdist: f64 = averages_dist.iter().sum();
        let avgdia: f64 = averages_diameter.iter().sum();
        let n = averages_dist.len() as f64;

        println!(
            "average of averages: distance {:.3}, diameter {:.3}.",
            avgdist / n,
            avgdia / n
        );

        remaining -= slot;
        iteration += 1;
    }

    println!("\nTotal time: {:?}", instant.elapsed());
}
