use bitvec::bitvec;
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

fn bfs_layered(start: usize, graph: Arc<Vec<Vec<usize>>>) -> (Vec<(usize, usize)>, usize) {
    let mut distances = Vec::new();
    let mut frontier = Vec::new();
    let mut diameter = 0usize;

    let mut seen = bitvec![0; graph.len()];

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

fn sample(k: usize, agraph: &Arc<Vec<Vec<usize>>>, r: &mut ThreadRng) -> Vec<usize> {
    let num_nodes = agraph.len();
    let cross = Arc::new(Mutex::new(vec![0usize; num_nodes]));

    for _ in 0..k {
        let v = r.gen_range(0..num_nodes);
        let agraph = Arc::clone(&agraph);
        let cross = Arc::clone(&cross);
        thread::spawn(move || {
            let (distances, _) = bfs(v, agraph);
            let mut c = cross.lock().unwrap();
            for (v, d) in distances {
                c[v] += 1;
            }
            print!(">");
            io::stdout().flush().expect("Unable to flush stdout");
        });
    }

    print!("|");
    {
        let mut cross = cross.lock().unwrap();
        for i in 1..num_nodes {
            cross[i] += cross[i - 1];
        }
    }

    let cross = cross.lock().unwrap();
    let maxc = cross[num_nodes - 1];

    let sampled = (0..k)
        .map(|j| {
            let c = r.gen_range(0..maxc);
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

            l
        })
        .collect();

    println!("s");

    sampled
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
        let graph = BvGraphSeq::with_basename("hello").load().unwrap();
        buffer.push(src);
        buffer.push(succ.len());

        for dst in succ {
            buffer.push(dst);
        }
    });
}

fn main() {
    let mut r = rand::thread_rng();
    let mut slot = 61;
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

        let sampled = sample(slot, &ag_t, &mut r);

        let tup = Arc::new(Mutex::new((0usize, 0usize, 0usize)));

        for v in sampled {
            let ag = Arc::clone(&ag);
            let tup = Arc::clone(&tup);
            thread::spawn(move || {
                let (distances, dia) = bfs(v, ag);
                let (mut s, mut c, mut d) = *tup.lock().unwrap();

                d += dia;
                for (_v, dist) in distances {
                    s += dist;
                    c += 1;
                }

                print!("<");
                io::stdout().flush().expect("Unable to flush stdout");
            });
        }

        println!("|");

        let (sum, count, dia) = *tup.lock().unwrap();

        let adist = (sum as f64) / (count as f64);
        let adia = (dia as f64) / (slot as f64);

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
