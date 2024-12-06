use rand::Rng;
use std::{
    collections::VecDeque,
    io::{self, Write},
    ops::Div,
};
use webgraph::prelude::*;

fn bfs<F: RandomAccessDecoderFactory>(
    start: usize,
    graph: &BvGraph<F>,
) -> (Vec<usize>, Vec<usize>) {
    let mut distances = vec![0usize; graph.num_nodes()];
    let mut queue = VecDeque::new();
    let mut good = Vec::new();

    queue.push_back(start);

    while !queue.is_empty() {
        let current_node = queue.pop_front().unwrap();

        let d = distances[current_node];

        for succ in graph.successors(current_node) {
            if succ != start && distances[succ] == 0 {
                distances[succ] = d + 1;
                good.push(succ);
                queue.push_back(succ);
            }
        }
    }

    (distances, good)
}

fn sample<F: RandomAccessDecoderFactory>(k: usize, graph: &BvGraph<F>) -> Vec<usize> {
    let num_nodes = graph.num_nodes();
    let mut r = rand::thread_rng();
    let mut sampled = vec![0usize; k];
    let mut cross = vec![0usize; num_nodes];

    for _ in 0..k {
        let (_, dgood) = bfs(r.gen_range(0..num_nodes), graph);

        print!(",");
        io::stdout().flush().expect("Unable to flush stdout");

        for i in dgood {
            cross[i] += 1;
        }
    }

    for i in 1..num_nodes {
        cross[i] += cross[i - 1];
    }

    let (minc, maxc) = (cross[0], cross[num_nodes - 1]);
    
    for i in 0..k {
        let c = r.gen_range(minc..maxc);

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
        assert!(l == h);
        sampled[i] = h;
    }

    sampled
}

fn main() {
    let graph = BvGraph::with_basename("/data/bitcoin/bitcoin-webgraph/pg")
        .load()
        .unwrap();

    let graph_t = BvGraph::with_basename("/data/bitcoin/bitcoin-webgraph/pg-t")
        .load()
        .unwrap();

    let num_nodes = graph.num_nodes();
    let epsilon = 0.1f64;
    let k = (num_nodes as f64).log2().div(epsilon.powi(2)).ceil() as usize;

    println!(
        "|V| = {}, |E| = {}, |S| = {}.",
        num_nodes,
        graph.num_arcs(),
        k
    );

    for j in 1..k + 1 {
        println!("*** |s| = {}", j);

        let sampled = sample(j, &graph_t);

        let mut sum = 0usize;
        let mut count = 0usize;

        for (i, &s) in sampled.iter().enumerate() {
            let (distances, good) = bfs(s, &graph);

            for d in good {
                sum = sum + distances[d];
                count = count + 1;
            }

            println!(
                "after {}, reachable pairs {},average distance {}.",
                i + 1,
                count,
                (sum as f64).div(count as f64)
            );
        }
    }
}
