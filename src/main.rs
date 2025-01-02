use lender::for_;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::{env, thread};
use std::{
    io::{self, Write},
    ops::Div,
};
use sux::bits::BitVec;
use webgraph::prelude::*;

fn bfs(
    start: usize,
    across: Arc<Mutex<Vec<usize>>>,
    graph: Arc<Vec<Vec<usize>>>,
) -> (usize, usize, usize) {
    let mut frontier = Vec::new();
    let mut diameter = 0usize;
    let mut distances = Vec::new();
    let mut seen = BitVec::new(graph.len());

    seen.set(start, true);

    frontier.push(start);

    while !frontier.is_empty() {
        let mut frontier_next = Vec::new();
        diameter = diameter + 1;

        for current_node in frontier.iter() {
            for each in graph[*current_node].iter() {
                let succ = *each;

                if !seen.get(succ) {
                    seen.set(succ, true);
                    distances.push((succ, diameter));
                    frontier_next.push(succ);
                }
            }
        }

        frontier = frontier_next;
    }

    let mut distance = 0usize;
    {
        let mut cross = across.lock().unwrap();
        for (succ, d) in distances.iter() {
            cross[*succ] += *d;
            distance += *d;
        }
    }

    (diameter, distance, distances.len())
}

fn sample(k: usize, agraph: &Arc<Vec<Vec<usize>>>, r: &mut ThreadRng) -> Vec<usize> {
    let num_nodes = agraph.len();
    let across = Arc::new(Mutex::new(vec![0usize; num_nodes]));

    let mut handles = Vec::new();

    for _ in 0..k {
        let v = r.gen_range(0..num_nodes);
        let agraph = Arc::clone(agraph);
        let across = Arc::clone(&across);
        let handle = thread::spawn(move || {
            bfs(v, across, agraph);

            print!(">");
            io::stdout().flush().expect("Unable to flush stdout");
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    print!("|");

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

    println!("s");

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
                sample(slot, &ag_t, &mut r)
            }
        };

        let mut handles = Vec::new();
        let triple = Arc::new(Mutex::new((0usize, 0usize, 0usize)));
        let across = Arc::new(Mutex::new(vec![0usize; num_nodes]));

        for v in sampled {
            let ag = Arc::clone(&ag);
            let triple = Arc::clone(&triple);
            let across = Arc::clone(&across);
            let handle = thread::spawn(move || {
                let (dia, sum, count) = bfs(v, across, ag);

                {
                    let mut t = triple.lock().unwrap();

                    t.0 = t.0 + sum;
                    t.1 = t.1 + count;
                    t.2 = t.2.max(dia);
                }

                print!("<");
                io::stdout().flush().expect("Unable to flush stdout");
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let (sum, count, dia) = *triple.lock().unwrap();

        println!("|");

        let adist = (sum as f64) / (count as f64);
        let adia = dia as f64;

        println!("averages: distance {:.3}, diameter {:.3}.", adist, adia);

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
}
