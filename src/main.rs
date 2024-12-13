use bitvec::{bitarr, bits, bitvec, BitArr};
use lender::for_;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::collections::VecDeque;
use std::fs::File;
use std::io::prelude::*;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
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

fn bfs_layered(start: usize, graph: Arc<Vec<Vec<usize>>>) -> (Vec<(usize, usize)>, usize) {
    let mut distances = Vec::new();
    let mut frontier = Vec::new();
    let mut diameter = 0usize;

    let n = graph.len();
    let mut seen = bitvec![0; n];

    seen.set(start, true);

    frontier.push(start);

    while !frontier.is_empty() {
        diameter = diameter + 1;

        let mut frontier_next = Vec::new();

        for current_node in frontier.iter() {
            for each in graph[*current_node].iter() {
                let succ = *each;

                match seen.get_mut(succ) {
                    Some(mut b) => {
                        if *b == false {
                            *b = true;
                            distances.push((succ, diameter));
                            frontier_next.push(succ);
                        }
                    }
                    _ => {}
                }
            }
        }

        frontier = frontier_next;
    }

    (distances, diameter)
}

fn bfs_queued<F: RandomAccessDecoderFactory>(
    start: usize,
    graph: &BvGraph<F>,
) -> (Vec<Option<usize>>, Vec<usize>, usize) {
    let mut distances = vec![None; graph.num_nodes()];
    let mut queue = VecDeque::new();
    let mut good = Vec::new();
    let mut diameter = 0usize;

    distances[start] = Some(diameter);

    queue.push_back(start);

    while !queue.is_empty() {
        let current_node = queue.pop_front().unwrap();

        let d = distances[current_node].unwrap();
        diameter = diameter.max(d);

        for succ in graph.successors(current_node) {
            match distances[succ] {
                None => {
                    distances[succ] = Some(d + 1);
                    good.push(succ);
                    queue.push_back(succ);
                }
                _ => {}
            }
        }
    }

    (distances, good, diameter)
}

fn bfs(start: usize, graph: Arc<Vec<Vec<usize>>>) -> (Vec<(usize, usize)>, usize) {
    bfs_layered(start, graph)
}

fn sample(k: usize, agraph: &Arc<Vec<Vec<usize>>>, r: &mut ThreadRng) -> (Vec<usize>, f64) {
    let num_nodes = agraph.len();
    let mut sampled = vec![0usize; k];

    let diameter_mutex = Arc::new(Mutex::new(0usize));
    let cross_mutex = Arc::new(Mutex::new(vec![0usize; num_nodes]));

    for _ in 0..k {
        let v = r.gen_range(0..num_nodes);
        let agraph = Arc::clone(&agraph);
        let diameter_mutex = Arc::clone(&diameter_mutex);
        let cross_mutex = Arc::clone(&cross_mutex);
        thread::spawn(move || {
            let (distances, d) = bfs(v, agraph);
            {
                let mut diameter = diameter_mutex.lock().unwrap();
                *diameter += d;
            }
            {
                let mut cross = cross_mutex.lock().unwrap();
                for (v, _d) in distances {
                    cross[v] += 1;
                }
            }
            print!(">");
            io::stdout().flush().expect("Unable to flush stdout");
        });
    }

    let mut cross = cross_mutex.lock().unwrap();

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

    let diameter = *diameter_mutex.lock().unwrap();

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
    let mut r = rand::thread_rng();
    let mut slot = 50;
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

        let (sampled, _) = sample(slot, &ag_t, &mut r);

        println!("");

        let diameter_mutex = Arc::new(Mutex::new(0usize));
        let sum_mutex = Arc::new(Mutex::new(0usize));
        let count_mutex = Arc::new(Mutex::new(0usize));

        for v in sampled {
            let ag = Arc::clone(&ag);
            let diameter_mutex = Arc::clone(&diameter_mutex);
            let sum_mutex = Arc::clone(&sum_mutex);
            let count_mutex = Arc::clone(&count_mutex);
            thread::spawn(move || {
                let (distances, dia) = bfs(v, ag);
                {
                    let mut diameter = diameter_mutex.lock().unwrap();
                    *diameter += dia;
                }
                {
                    let mut count = count_mutex.lock().unwrap();
                    *count += distances.len();
                }
                {
                    let mut sum = sum_mutex.lock().unwrap();
                    for (_v, d) in distances {
                        *sum += d;
                    }
                }

                print!("<");
                io::stdout().flush().expect("Unable to flush stdout");
            });
        }

        let sum = *sum_mutex.lock().unwrap();
        let count = *count_mutex.lock().unwrap();
        let dia = *diameter_mutex.lock().unwrap();

        let adist = (sum as f64) / (count as f64);
        let adia = (dia as f64) / (slot as f64);

        println!("\naverages: distance {}, diameter {}.", adist, adia);

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
