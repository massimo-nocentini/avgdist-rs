use rand::rngs::ThreadRng;
use rand::Rng;
use std::io::{self, Write};
use std::ops::Div;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{env, thread};
use sux::bits::BitVec;
use webgraph::prelude::*;

fn bfs<T: RandomAccessGraph>(
    start: usize,
    channel: Option<&Arc<Mutex<Vec<usize>>>>,
    graph: Arc<T>,
) -> (usize, usize, usize) {
    let mut frontier = Vec::new();
    let mut distance = 0usize;
    let mut diameter = 0usize;
    let mut count = 0usize;
    let mut seen = BitVec::new(graph.num_nodes());

    seen.set(start, true);

    frontier.push((start, 0));

    while !frontier.is_empty() {
        let mut frontier_next = Vec::new();

        for (current_node, l) in frontier.iter() {
            let ll = *l + 1;

            for succ in graph.successors(*current_node) {
                if !seen.get(succ) {
                    diameter = diameter.max(ll);
                    seen.set(succ, true);
                    count += 1;
                    distance += ll;
                    frontier_next.push((succ, ll));
                }
            }
        }

        if !frontier_next.is_empty() {
            if let Some(ch) = channel {
                let mut ch = ch.lock().unwrap();
                for (v, d) in frontier_next.iter() {
                    ch[*v] += *d;
                }
            }
        }

        frontier = frontier_next;
    }

    (diameter, distance, count)
}

fn sample<T: RandomAccessGraph + Send + Sync + 'static>(
    k: usize,
    agraph: Arc<T>,
    r: &mut ThreadRng,
) -> Vec<usize> {
    let num_nodes = agraph.num_nodes();

    let across = Arc::new(Mutex::new(vec![0usize; num_nodes]));

    let mut handles = Vec::new();

    for _ in 0..k {
        let v = r.gen_range(0..num_nodes);
        let agraph = Arc::clone(&agraph);
        let across = across.clone();
        let handle = thread::spawn(move || {
            let instant = Instant::now();
            bfs(v, Some(&across), agraph);
            print!(">: {:?} | ", instant.elapsed());
            io::stdout().flush().unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let mut cross = across.lock().unwrap();

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

    let graph_filename = &args[1];
    let graph_filename_t = &args[2];
    let mut slot: usize = args[3].parse().unwrap();
    let epsilon: f64 = args[4].parse().unwrap();
    let truth: bool = args[5].parse().unwrap();
    let dummy: bool = args[6].parse().unwrap();

    let mut r = rand::thread_rng();
    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let num_nodes = graph.num_nodes();
    let k = (num_nodes as f64).log2().div(epsilon.powi(2)).ceil() as usize;

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
                let graph_t = BvGraph::with_basename(graph_filename_t).load().unwrap();
                let ag_t = Arc::new(graph_t);
                sample(slot, ag_t, &mut r)
            }
        };
        println!("sampled in {:?}", instant.elapsed());

        let triple = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
        let mut handles = Vec::new();
        let instant = Instant::now();
        for v in sampled {
            let ag = Arc::clone(&ag);
            let tx = triple.clone();
            let handle = thread::spawn(move || {
                let instant = Instant::now();
                let (dia, sum, count) = bfs(v, None, ag);
                {
                    let mut tx = tx.lock().unwrap();

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
