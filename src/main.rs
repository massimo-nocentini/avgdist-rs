
use lender::for_;
use rand::rngs::ThreadRng;
use rand::Rng;

use std::sync::{mpsc, Arc};
use std::{env, thread};
use std::{
    io::{self, Write},
    ops::Div,
};
use sux::bits::BitVec;
use webgraph::prelude::*;

fn bfs_layered(start: usize, graph: Arc<Vec<Vec<usize>>>) -> (Vec<(usize, usize)>, usize) {
    let mut distances = Vec::new();
    let mut frontier = Vec::new();
    let mut diameter = 0usize;

    let mut seen = BitVec::new(graph.len());

    seen.set(start, true);

    frontier.push(start);

    while !frontier.is_empty() {
        let mut frontier_next = Vec::new();
        diameter = diameter + 1;

        for current_node in frontier.iter() {
            for each in graph[*current_node].iter() {
                let succ = *each;

                match seen.get(succ) {
                    false => {
                        seen.set(succ, true);
                        distances.push((succ, diameter));
                        frontier_next.push(succ);
                    }
                    _ => {}
                }
            }
        }

        frontier = frontier_next;
    }

    (distances, diameter)
}

fn bfs(start: usize, graph: Arc<Vec<Vec<usize>>>) -> (Vec<(usize, usize)>, usize) {
    bfs_layered(start, graph)
}

fn sample(k: usize, agraph: &Arc<Vec<Vec<usize>>>, r: &mut ThreadRng) -> (Vec<usize>, usize) {
    let num_nodes = agraph.len();
    let mut cross = vec![0usize; num_nodes];
    let mut diameter = 0usize;

    let (tx, rx) = mpsc::channel();

    for _ in 0..k {
        let v = r.gen_range(0..num_nodes);
        let tx1 = tx.clone();
        let agraph = Arc::clone(agraph);
        thread::spawn(move || {
            let tup = bfs(v, agraph);
            tx1.send(tup).unwrap();
            print!(">");
            io::stdout().flush().expect("Unable to flush stdout");
        });
    }

    drop(tx);

    for (distances, dia) in rx {
        diameter = diameter.max(dia);

        for (v, d) in distances {
            cross[v] += 1;
        }
    }

    print!("|");

    for i in 1..num_nodes {
        cross[i] += cross[i - 1];
    }

    let maxc = *cross.last().unwrap();
    let (tx, rx) = mpsc::channel();
    let across = Arc::new(cross);

    for _ in 0..k {
        let tx1 = tx.clone();
        let across = Arc::clone(&across);
        let c = r.gen_range(0..maxc);
        thread::spawn(move || {
            let mut l = 0usize;
            let mut h = num_nodes - 1;

            while l < h {
                let m = (h + l) >> 1;
                if across[m] < c {
                    l = m + 1;
                } else {
                    h = m;
                }
            }

            tx1.send(l).unwrap();
        });
    }

    println!("s");

    drop(tx);

    (rx.iter().collect(), diameter)
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
    let graph_t = BvGraph::with_basename(graph_filename_t).load().unwrap();

    let num_nodes = graph.num_nodes();
    let k = (num_nodes as f64).log2().div(epsilon.powi(2)).ceil() as usize;

    let mut g = vec![Vec::new(); num_nodes];
    for_!((src, succ) in graph {
        g[src].extend(succ);
    });

    let mut g_t = vec![Vec::new(); num_nodes];
    for_!((src, succ) in graph_t {
        g_t[src].extend(succ);
    });

    println!(
        "|V| = {}, |E| = {}, |S| = {}, s = {}.",
        num_nodes,
        graph.num_arcs(),
        k,
        slot
    );

    let ag = Arc::new(g);
    let ag_t = Arc::new(g_t);

    let mut averages_dist = Vec::new();
    let mut averages_diameter = Vec::new();

    let mut remaining = k;
    let mut iteration = 1usize;

    while remaining > 0 {
        slot = slot.min(remaining);

        println!(
            "\n*** iteration {}, batch size {}, remaining {}.",
            iteration,
            slot,
            remaining - slot
        );

        let sampled: Vec<usize> = if truth {
            slot = remaining;
            (0..num_nodes).collect()
        } else {
            if dummy {
                (0..slot).map(|_j| r.gen_range(0..num_nodes)).collect()
            } else {
                let (sampled, _) = sample(slot, &ag_t, &mut r);
                sampled
            }
        };

        let mut sum = 0usize;
        let mut count = 0usize;
        let mut dia = 0;

        let (tx, rx) = mpsc::channel();

        for v in sampled {
            let tx1 = tx.clone();
            let ag = Arc::clone(&ag);
            thread::spawn(move || {
                let tup = bfs(v, ag);
                tx1.send(tup).unwrap();
                print!("<");
                io::stdout().flush().expect("Unable to flush stdout");
            });
        }

        drop(tx);

        for (distances, d) in rx {
            dia = dia.max(d);

            for (_v, d) in distances {
                sum = sum + d;
                count = count + 1;
            }
        }

        println!("|");

        let adist = (sum as f64) / (count as f64);
        let adia = dia as f64;

        println!("averages: distance {}, diameter {}.", adist, adia);

        averages_dist.push(adist);
        averages_diameter.push(adia);

        let avgdist: f64 = averages_dist.iter().sum();
        let avgdia: f64 = averages_diameter.iter().sum();
        let n = averages_dist.len() as f64;

        println!(
            "average of averages: distance {}, diameter {}.",
            avgdist / n,
            avgdia / n
        );

        remaining -= slot;
        iteration += 1;
    }
}
