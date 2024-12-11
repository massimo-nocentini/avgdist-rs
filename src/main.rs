use lender::for_;
use rand::Rng;
use std::fs::File;
use std::io::prelude::*;
use std::{
    io::{self, Write},
    ops::Div,
};
use webgraph::prelude::*;

struct Pennant<'a, T> {
    value: &'a T,
    left: Option<&'a Pennant<'a, T>>,
    right: Option<&'a Pennant<'a, T>>,
}

impl<'a, T> Pennant<'a, T> {
    fn new(value: &'a T) -> Pennant<'a, T> {
        Pennant {
            value,
            left: None,
            right: None,
        }
    }

    fn union(&'a self, y: &'a mut Pennant<'a, T>) -> Pennant<'a, T> {
        y.right = self.left;
        // self.left = Some(y);
        Pennant {
            value: self.value,
            left: Some(y),
            right: self.right,
        }
    }

    fn len(&self) -> usize {
        let l = match self.left {
            None => 0,
            Some(p) => p.len(),
        };
        let r = match self.right {
            None => 0,
            Some(p) => p.len(),
        };
        1 + l + r
    }
}

struct Bag<'a, T> {
    spine: Vec<Option<&'a Pennant<'a, T>>>,
}

impl<'a, T> Bag<'a, T> {
    fn new() -> Bag<'a, T> {
        Bag { spine: Vec::new() }
    }

    fn len(&self) -> usize {
        let mut l = 0;

        for p in self.spine.iter() {
            l += match *p {
                None => 0,
                Some(pp) => pp.len(),
            };
        }

        l
    }

    fn push(&mut self, value: &T) {
        let mut v = Pennant::new(value);
        let mut k = 0usize;

        while k < self.len() {
            let each = self.spine[k];

            let pp = match each {
                None => break,
                Some(p) => break, // p.union(&mut v),
            };

            self.spine[k] = None;
            k += 1;
        }
    }
}

fn bfs<F: RandomAccessDecoderFactory>(
    start: usize,
    graph: &BvGraph<F>,
) -> (Vec<Option<usize>>, Vec<usize>, usize) {
    let mut distances = vec![None; graph.num_nodes()];
    let mut frontier = Vec::new();
    let mut good = Vec::new();
    let mut diameter = 0usize;

    distances[start] = Some(0usize);

    frontier.push(start);

    while !frontier.is_empty() {
        diameter = diameter + 1;

        let mut frontier_next = Vec::new();

        for current_node in frontier.iter() {
            for succ in graph.successors(*current_node) {
                match distances[succ] {
                    None => {
                        distances[succ] = Some(diameter);
                        good.push(succ);
                        frontier_next.push(succ);
                    }
                    _ => {}
                }
            }
        }

        frontier = frontier_next;
    }

    (distances, good, diameter)
}

fn sample<F: RandomAccessDecoderFactory>(k: usize, graph: &BvGraph<F>) -> (Vec<usize>, f64) {
    let num_nodes = graph.num_nodes();
    let mut r = rand::thread_rng();
    let mut sampled = vec![0usize; k];
    let mut cross = vec![0usize; num_nodes];
    let mut diameter = 0usize;

    for _ in 0..k {
        let (distances, dgood, d) = bfs(r.gen_range(0..num_nodes), graph);

        diameter += d;

        print!(",");
        io::stdout().flush().expect("Unable to flush stdout");

        for i in dgood {
            cross[i] += distances[i].unwrap();
        }
    }

    for i in 1..num_nodes {
        cross[i] += cross[i - 1];
    }

    let (minc, maxc) = (0, cross[num_nodes - 1]);

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

        sampled[i] = l;
    }

    (sampled, (diameter as f64) / (k as f64))
}

fn as_bytes(v: &[usize]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            v.as_ptr() as *const u8,
            v.len() * std::mem::size_of::<usize>(),
        )
    }
}

fn append_to_vec<F: RandomAccessDecoderFactory>(graph: &BvGraph<F>, buffer: &mut Vec<usize>) {
    buffer.push(graph.num_nodes());

    for_!((src, succ) in graph {

        buffer.push(src);
        buffer.push(succ.len());

        for dst in succ {
            buffer.push(dst);
        }
    });
}

fn main() {
    let graph = BvGraph::with_basename("/data/bitcoin/bitcoin-webgraph/pg")
        .load()
        .unwrap();

    let graph_t = BvGraph::with_basename("/data/bitcoin/bitcoin-webgraph/pg-t")
        .load()
        .unwrap();

    // {
    //     let mut file = File::create("/data/bitcoin/bitcoin-webgraph/pg.data").unwrap();
    //     let mut buffer = Vec::new();
    //     append_to_vec(&graph, &mut buffer);
    //     file.write_all(as_bytes(&buffer)).unwrap();
    // }

    // {
    //     let mut file = File::create("/data/bitcoin/bitcoin-webgraph/pg-t.data").unwrap();
    //     let mut buffer = Vec::new();
    //     append_to_vec(&graph_t, &mut buffer);
    //     file.write_all(as_bytes(&buffer)).unwrap();
    // }

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

        let (sampled, diameter) = sample(j, &graph_t);

        let mut sum = 0usize;
        let mut count = 0usize;

        for (i, &s) in sampled.iter().enumerate() {
            let (distances, good, d) = bfs(s, &graph);

            for d in good {
                sum = sum + distances[d].unwrap();
                count = count + 1;
            }

            println!(
                "after {}, reachable pairs {},average distance {}, average diameter {}.",
                i + 1,
                count,
                (sum as f64) / (count as f64),
                (diameter + (d as f64)) / 2.0
            );
        }
    }
}
