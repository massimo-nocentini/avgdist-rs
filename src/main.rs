
use std::io::{self, Write};
use rand::Rng;
use std::{collections::VecDeque, ops::Div};
use webgraph::prelude::*;

fn bfs<F: RandomAccessDecoderFactory>(start: usize, graph: &BvGraph<F>) -> (Vec<usize>, Vec<usize>) {
    let num_nodes = graph.num_nodes();
    let mut seen = vec![0usize; num_nodes];
    let mut queue = VecDeque::new();
    let mut good = Vec::new ();

    queue.push_back(start);

    while !queue.is_empty() {
        let current_node = queue.pop_front().unwrap();

        let d = seen[current_node];

        for succ in graph.successors(current_node) {
            if succ != start && seen[succ] == 0 {
                seen[succ] = d + 1;
                good.push (succ);
                queue.push_back(succ);
            }
        }
    }

    (seen, good)
}

fn sample<F: RandomAccessDecoderFactory>(epsilon: f64, graph: &BvGraph<F>) -> Vec<usize> {
    let num_nodes = graph.num_nodes();
    let k = 1usize; //(num_nodes as f64).log2().div(epsilon.powi(2)).ceil() as usize;
    let mut r = rand::thread_rng();
    let mut pool = vec![0usize; num_nodes];
    let mut sampled = vec![0usize; k];
    let mut cross = vec![0usize; num_nodes];
    let mut good = Vec::new ();

    println! ("Sample size {}", k);
    
    for node in 0..num_nodes {
        pool[node] = node; // the identity permutation.
    }

    for _ in 0..k {
        let mut start: usize = r.gen_range(0..pool.len());

        start = pool.remove(start);

        let (distances, dgood) = bfs(start, graph);

        print! (".");
        io::stdout().flush().expect("Unable to flush stdout");

        for i in dgood {
            //if distances[i] > 0 {
                //cross[i] += 1;
                good.push (i);
            //}
        }
    }

    for i in 0..k {
        let c = r.gen_range(0..good.len ());
        sampled[i] = good.remove (c);
    }

    /*
    for i in 1..num_nodes {
        cross[i] += cross[i - 1];
    }

    println! ("\nCumulated");

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
    */

    sampled
}

fn main() {
    let graph = BvGraph::with_basename("/data/bitcoin/bitcoin-webgraph/pg")
        .load()
        .unwrap();

    let graph_t = BvGraph::with_basename("/data/bitcoin/bitcoin-webgraph/pg-t")
        .load()
        .unwrap();

    let sampled = sample(0.1, &graph_t);

    let mut sum = 0usize;
    let mut count = 0usize;

    for s in sampled {
        let (distances, good) = bfs(s, &graph);
        print! (".");
        io::stdout().flush().expect("Unable to flush stdout");
        for d in good {
            sum = sum + distances[d];
            count = count + 1;
        }
    }

    println!("\n{}", (sum as f64).div(count as f64));
}
