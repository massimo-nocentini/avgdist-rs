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

            let (dia, dist, count, _seen, finite_dist) = bfs(vertex, &agraph);
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
        "|V| = {}, |E| = {}, |S| = {}, s = {}.",
        num_nodes,
        graph.num_arcs(),
        sample_size,
        thread_pool.current_num_threads()
    );

    let ag = Arc::new(graph);

    let mut sizes = vec![0usize; num_nodes];
    let mut sums = vec![None; num_nodes];

    let instant = Instant::now();
    let (_dia, _sum, _count, v_sizes, v_finite_ds) =
        sample(&thread_pool, sample_size, ag.clone(), exact_computation);

    println!("sampled in {:?}", instant.elapsed());

    for i in 0..num_nodes {
        sizes[i] += v_sizes[i];

        if let Some(sum) = v_finite_ds[i] {
            let existing_sum = sums[i].unwrap_or(0.0);
            sums[i] = Some(existing_sum + sum);
        }
    }

    // c(u)=(1/n) * sum_{v in V} 1/(1+d(u,v))
    // c(u,U)= (1/|U|)* sum_{v in U} 1/(1+d(u,v))

    let normalization = (sample_size as f64).recip();

    let mut centralities: Vec<(usize, f64)> = (0..num_nodes)
        .filter_map(|node| {
            if let Some(sum) = sums[node] {
                Some((node, normalization * sum))
            } else {
                None
            }
        })
        .collect();

    let mut histogram = std::collections::BTreeMap::<u64, usize>::new();
    let b = 1_000_000_000.0;
    for &(_, centrality) in &centralities {
        let bucket = (centrality * b).floor() as u64;
        *histogram.entry(bucket).or_insert(0) += 1;
    }

    println!("\nHistogram of centralities (bucketed by {}):", 1.0 / b);
    for (bucket, count) in histogram.iter().rev() {
        println!(
            "{} - {}: {}",
            *bucket as f64 / b,
            (*bucket as f64 + 1.0) / b,
            count
        );
    }

    println!("\nCloseness centrality (node, centrality) ordered by most central vertices:");

    centralities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (node, closeness) in centralities {
        println!("{}\t{}", node, closeness);
    }

    println!("\nTotal time: {:?}", instant.elapsed());
}
