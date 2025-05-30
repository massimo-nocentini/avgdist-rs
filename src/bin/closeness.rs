use rand::rngs::ThreadRng;
use rand::Rng;
use std::io::{self, Write};
use std::ops::Div;
use std::sync::Arc;
use std::time::Instant;
use std::{env, thread};
use sux::bits::BitVec;
use webgraph::prelude::*;

fn bfs<T: RandomAccessGraph>(
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

fn sample<T: RandomAccessGraph + Send + Sync + 'static>(
    k: usize,
    agraph: Arc<T>,
    exact_computation: bool,
) -> (usize, usize, usize, Vec<usize>, Vec<usize>) {
    let num_nodes = agraph.num_nodes();

    let (tx, rx) = std::sync::mpsc::channel();

    let n = if exact_computation { num_nodes } else { k };

    for each in 0..n {
        let agraph = agraph.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let instant = Instant::now();
            let mut r = rand::thread_rng();

            if exact_computation {
                // If exact computation is requested, we sample from all nodes
                let (dia, dist, count, _, finite_dist) = bfs(each, &agraph);

                tx.send((dia, dist, count, each, finite_dist)).unwrap();
            } else {
                loop {
                    let v = r.gen_range(0..num_nodes);
                    let w = r.gen_range(0..num_nodes);

                    if v == w {
                        continue;
                    }

                    let (dia, dist, count, seen, finite_dist) = bfs(v, &agraph);

                    if seen.get(w) {
                        tx.send((dia, dist, count, v, finite_dist)).unwrap();
                        break;
                    }
                }
            }

            print!(">: {:?} | ", instant.elapsed());
            io::stdout().flush().unwrap();
        });
    }

    drop(tx);

    let mut tx = (0usize, 0usize, 0usize);

    let mut sizes = vec![0usize; num_nodes];
    let mut finite_ds = vec![0usize; num_nodes];

    while let Ok((dia, sum, count, _v, finite_dist)) = rx.recv() {
        tx.0 = std::cmp::max(tx.0, dia);
        tx.1 += sum;
        tx.2 += count;

        for (node, dist) in finite_dist {
            sizes[node] += 1;
            finite_ds[node] += dist;
        }
    }

    (tx.0, tx.1, tx.2, sizes, finite_ds)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let mut slot: usize = args[2].parse().unwrap();
    let epsilon: f64 = args[3].parse().unwrap();
    let exact_computation: bool = args[4].parse().unwrap();

    let mut r = rand::thread_rng();
    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let num_nodes = graph.num_nodes();
    let k = 6.907.div(2.0 * epsilon.powi(2)).ceil() as usize;

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

    let mut sizes = vec![0usize; num_nodes];
    let mut finite_ds = vec![0usize; num_nodes];

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
        let (dia, sum, count, v_sizes, v_finite_ds) = sample(slot, ag.clone(), exact_computation);

        println!("sampled in {:?}", instant.elapsed());

        for i in 0..num_nodes {
            sizes[i] += v_sizes[i];
            finite_ds[i] += v_finite_ds[i];
        }

        if count > 0 {
            let adist = (sum as f64) / (count as f64);
            let adia = dia as f64;

            // println!("\naverages: distance {:.3}, diameter {:.3}.", adist, adia);

            averages_dist.push(adist);
            averages_diameter.push(adia);
        }

        if exact_computation {
            break;
        }

        // let n = averages_dist.len() as f64;
        // let avgdist: f64 = averages_dist.iter().sum::<f64>() / n;
        // let avgdia: f64 = averages_diameter.iter().sum::<f64>() / n;

        // let avgdist_var = averages_dist
        //     .iter()
        //     .map(|x| (x - avgdist).powi(2))
        //     .sum::<f64>()
        //     / (n - 1.0);

        // let avgdia_var = averages_diameter
        //     .iter()
        //     .map(|x| (x - avgdia).powi(2))
        //     .sum::<f64>()
        //     / (n - 1.0);

        // println!(
        //     "average of averages: distance {:.9}, std {:.9}, diameter {:.3} (std {:.3}).",
        //     avgdist,
        //     avgdist_var.sqrt(),
        //     avgdia,
        //     avgdia_var.sqrt()
        // );

        remaining -= slot;
        iteration += 1;
    }

    let mut centralities: Vec<(usize, f64)> = (0..num_nodes)
        .filter_map(|node| {
            let reach = sizes[node];
            let dist_sum = finite_ds[node];
            if reach > 0 && dist_sum > 0 {
                Some((
                    node,
                    // (reach as f64).powf(2.0)                        / ((dist_sum * (if exact_computation { num_nodes } else { k })) as f64),
                    1.0 / ((dist_sum * (if exact_computation { num_nodes } else { k })) as f64),
                ))
            } else {
                None
            }
        })
        .collect();

    // Group centralities by their floored value (to 6 decimal places)
    let mut grouped = std::collections::BTreeMap::<u64, Vec<usize>>::new();
    let b = 1_000_000_000.0;
    for &(node, centrality) in &centralities {
        let bucket = (centrality * b).floor() as u64;
        grouped.entry(bucket).or_default().push(node);
    }

    println!("\nGrouped centralities (bucketed by {:.6}):", 1.0 / b);
    for (bucket, nodes) in grouped.iter().rev() {
        let centrality_value = *bucket as f64 / b;
        println!("{:.6}: {} nodes {:?}", centrality_value, nodes.len(), nodes);
    }

    let mut histogram = std::collections::BTreeMap::<u32, usize>::new();
    let b = 1000000.0;
    for &(_, centrality) in &centralities {
        let bucket = (centrality * b).floor() as u32;
        *histogram.entry(bucket).or_insert(0) += 1;
    }

    println!("\nHistogram of centralities (bucketed by {}):", 1.0 / b);
    for (bucket, count) in histogram.iter().rev() {
        println!(
            "{:.6} - {:.6}: {}",
            *bucket as f64 / b,
            (*bucket as f64 + 1.0) / b,
            count
        );
    }

    println!("\nCloseness centrality (node, centrality) ordered by most central vertices:");

    centralities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (node, closeness) in centralities {
        println!("{}, {:.6}", node, closeness);
    }

    println!("\nTotal time: {:?}", instant.elapsed());
}
