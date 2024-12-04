
use lender::for_;
use rand::Rng;
use std::{collections::VecDeque, ops::Div};
use webgraph::prelude::*;

fn bfs<F: RandomAccessDecoderFactory>(start: usize, graph: &BvGraph<F>) -> Vec<usize> {
    let num_nodes = graph.num_nodes();
    let mut seen = vec![0usize; num_nodes];
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while !queue.is_empty() {
        let current_node = queue.pop_front().unwrap();

        let d = seen[current_node];

        for succ in graph.successors(current_node) {
            if succ != start && seen[succ] == 0 {
                seen[succ] = d + 1;
                queue.push_back(succ);
            }
        }
    }

    seen
}

fn sample<F: RandomAccessDecoderFactory>(epsilon: f64, graph: &BvGraph<F>) -> Vec<usize> {
    let num_nodes = graph.num_nodes();
    let k = (num_nodes as f64).log2().div(epsilon.powi(2)).ceil() as usize;
    let mut r = rand::thread_rng();
    let mut pool = vec![0usize; num_nodes];
    let mut sampled = vec![0usize; k];

    let mut cross = vec![0usize; num_nodes];

    for_!((node, _) in graph {
        pool[node] = node;
    });

    for _ in 0..k {
        let mut start: usize = r.gen_range(0..pool.len());

        start = pool.remove(start);

        let distances = bfs(start, graph);

        for (i, _) in distances.iter().enumerate() {
            cross[i] += 1;
        }
    }

    for i in 1..num_nodes {
        cross[i] += cross[i - 1];
    }

    for i in 0..k {
        let c = r.gen_range(cross[0]..cross[num_nodes - 1]);

        let mut l = 0usize;
        let mut h = num_nodes - 1;

        loop {
            if l >= h {
                break;
            }

            let m = l + ((h - l) >> 1);
            if cross[m] == c {
                l = m;
                h = m;
            } else if cross[m] > c {
                h = m - 1;
            } else {
                l = m + 1;
            }
        }

        sampled[i] = h;
    }

    sampled
}

fn main() {
    let graph = BvGraph::with_basename("/home/mn/Developer/bitcoin/pg")
        .load()
        .unwrap();

    let graph_t = BvGraph::with_basename("/home/mn/Developer/bitcoin/pg-t")
        .load()
        .unwrap();

    let sampled = sample(0.1, &graph_t);

    let mut sum = 0usize;
    let mut count = 0usize;

    for s in sampled {
        let distances = bfs(s, &graph);
        for d in distances {
            if d > 0 {
                sum = sum + d;
                count = count + 1;
            }
        }
    }

    println!("{}", (sum as f64).div(count as f64));
}
