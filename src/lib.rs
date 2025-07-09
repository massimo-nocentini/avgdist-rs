use core::panic;
use rand::Rng;
use std::io::{self, Write};
use std::ops::Neg;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;
use sux::bits::BitVec;
use webgraph::prelude::*;

fn avgdist_bfs<T: RandomAccessGraph>(
    start: usize,
    graph: &Arc<T>,
) -> (usize, usize, usize, BitVec) {
    let mut frontier = Vec::new();
    let mut distance = 0usize;
    let mut diameter = 0usize;
    let mut count = 0usize;
    let mut seen = BitVec::new(graph.num_nodes());

    seen.set(start, true);

    frontier.push((start, 0));

    while !frontier.is_empty() {
        let mut frontier_next = Vec::new();

        for (current_node, l) in frontier {
            let ll = l + 1;

            for succ in graph.successors(current_node) {
                if !seen.get(succ) {
                    diameter = std::cmp::max(diameter, ll);
                    seen.set(succ, true);
                    count += 1;
                    distance += ll;
                    frontier_next.push((succ, ll));
                }
            }
        }

        frontier = frontier_next;
    }

    (diameter, distance, count, seen)
}

pub fn avgdist_sample<T: RandomAccessGraph + Send + Sync + 'static>(
    thread_pool: &rayon::ThreadPool,
    k: usize,
    agraph: Arc<T>,
    exact_computation: bool,
) -> (usize, usize, usize, f64, usize) {
    let (tx, rx) = std::sync::mpsc::channel();

    let num_nodes = agraph.num_nodes();
    let remaining = Arc::new(AtomicUsize::new(k));
    let distr = rand::distributions::Uniform::new(0, num_nodes);

    for each in 0..k {
        let agraph = agraph.clone();
        let tx = tx.clone();
        let remaining = remaining.clone();

        thread_pool.spawn(move || {
            let instant = Instant::now();

            let current_avgdist = if exact_computation {
                let (dia, dist, count, _seen) = avgdist_bfs(each, &agraph);
                tx.send((dia, dist, count, each)).unwrap();
                (dist as f64) / (count as f64)
            } else {
                let mut r = rand::rngs::ThreadRng::default();

                loop {
                    let v = r.sample(distr);
                    let w = r.sample(distr);

                    if v == w {
                        continue;
                    }

                    let (dia, dist, count, seen) = avgdist_bfs(v, &agraph);

                    if seen.get(w) {
                        tx.send((dia, dist, count, v)).unwrap();

                        break (dist as f64) / (count as f64);
                    }
                }
            };

            {
                let rem = remaining.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

                println!(
                    "((avgdist {:.6}) (eta {:?}) (remaining {}))",
                    current_avgdist,
                    instant.elapsed(),
                    rem
                );
                io::stdout().flush().unwrap();
            }
        });
    }

    drop(tx);

    let mut tx = (0usize, 0usize, 0usize, 0.0, 0);

    while let Ok((dia, sum, count, _v)) = rx.recv() {
        tx.0 = std::cmp::max(tx.0, dia);
        tx.1 += sum;
        tx.2 += count;
        if count > 0 {
            tx.3 += (sum as f64) / (count as f64);
            tx.4 += 1;
        }
    }

    tx
}

fn hc_bfs<T: RandomAccessGraph>(
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

pub fn hc_sample<T: RandomAccessGraph + Send + Sync + 'static>(
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

            let (dia, dist, count, _seen, finite_dist) = hc_bfs(vertex, &agraph);
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

const BASE: usize = 32;
const LOG_MEMSIZE: usize = BASE - 5;
const MEMSIZE: usize = 1 << LOG_MEMSIZE;
const LOG_HTSIZE: usize = BASE - 7;
const HTSIZE: usize = 1 << LOG_HTSIZE;

struct Simpath {
    n: usize,
    mem: Vec<usize>,
    tail: usize,
    boundary: usize,
    head: usize,
    htable: Vec<usize>,
    htid: usize,
    htcount: usize,
    wrap: usize,
    vert: Vec<usize>,
    num: Vec<usize>,
    arcto: Vec<usize>,
    firstarc: Vec<usize>,
    mate: Vec<usize>,
    serial: usize,
    newserial: usize,
}

fn trunc(addr: usize) -> usize {
    addr & (MEMSIZE - 1)
}

fn printstate(simpath: &mut Simpath, j: usize, jj: usize, ll: usize) -> usize {
    let mut h: usize;
    let mut hh: usize;
    let mut t: usize;
    let mut tt: usize;
    let mut hash: usize;

    t = j;
    while t < jj {
        if simpath.mate[t] > 0 && simpath.mate[t] != t {
            break;
        }
        t += 1;
    }

    let ret;

    if t < jj {
        ret = 0;
    } else if ll < jj {
        ret = 0;
    } else {
        let ss = ll + 1 - jj;
        if simpath.head + ss - simpath.tail > MEMSIZE {
            panic!(
                "Oops, I'm out of memory (memsize={}, serial={})!",
                MEMSIZE, simpath.serial
            );
        }

        t = jj;
        h = trunc(simpath.head);
        hash = 0;
        while t <= ll {
            simpath.mem[h] = simpath.mate[t];
            hash = hash * 31415926525 + simpath.mate[t];

            t += 1;
            h = trunc(h + 1)
        }

        hash = hash & (HTSIZE - 1);
        loop {
            hh = simpath.htable[hash];

            if (hh ^ simpath.htid) >= MEMSIZE {
                simpath.htcount += 1;
                if simpath.htcount > (HTSIZE >> 1) {
                    panic!(
                        "Sorry, the hash table is full (htsize={}, serial={})!",
                        HTSIZE, simpath.serial
                    );
                }
                hh = trunc(simpath.head);
                simpath.htable[hash] = simpath.htid + hh;
                simpath.head += ss;
                break;
            }

            hh = trunc(hh);
            t = hh;
            h = trunc(simpath.head);
            tt = trunc(t + ss - 1);

            let should_continue = loop {
                if simpath.mem[t] != simpath.mem[h] {
                    break true;
                }

                if t == tt {
                    break false;
                }

                t = trunc(t + 1);
                h = trunc(h + 1);
            };

            if should_continue {
                hash = (hash + 1) & (HTSIZE - 1);
            } else {
                break;
            }
        }

        h = trunc(hh - simpath.boundary) / ss;
        ret = simpath.newserial + h;
    }

    return ret;
}

impl Simpath {
    pub fn new<T: RandomAccessGraph>(graph: &T) -> Self {
        let num_nodes = graph.num_nodes();
        Simpath {
            n: num_nodes,
            mem: vec![0; MEMSIZE],
            tail: 0,
            boundary: 0,
            head: 0,
            htable: vec![0; HTSIZE],
            htid: 0,
            htcount: 0,
            wrap: 1,
            vert: vec![0; num_nodes + 1],
            num: vec![0; num_nodes],
            arcto: vec![0; graph.num_arcs().try_into().unwrap()],
            firstarc: vec![0; num_nodes + 2],
            mate: vec![0; num_nodes + 3],
            serial: 0,
            newserial: 0,
        }
    }
}

pub fn simpath<T: RandomAccessGraph>(graph: &T, source: usize, target: usize) {
    let mut simpath = Simpath::new(graph);

    if source == 0 {
        let mut k = 0;
        while k < simpath.n {
            let k_succ = k + 1;
            simpath.num[k] = k_succ;
            simpath.vert[k_succ] = k;
            k = k_succ;
        }
    } else {
        simpath.vert[1] = source;
        simpath.num[source] = 1;

        let mut k = 1;
        let mut j = 0;
        while j < k {
            j += 1;
            let v = simpath.vert[j];
            for u in graph.successors(v) {
                if simpath.num[u] == 0 {
                    k += 1;
                    simpath.num[u] = k;
                    simpath.vert[k] = u;
                }
            }
        }

        if simpath.num[target] == 0 {
            eprintln!("Vertices reachables from {}:", source);

            for v in simpath.num.iter().filter(|&each| *each > 0) {
                eprintln!("{} ", simpath.vert[*v]);
            }

            panic!(
                "Sorry, there's no path from {} to {} in the graph!",
                source, target
            );
        }

        if k < simpath.n {
            eprintln!("The graph isn't connected ({} < {})! But that's OK; I'll work with the component of {}.", k, simpath.n, source);
            simpath.n = k;
        }
    }

    let mut m = 0;
    let mut k = 1;
    while k <= simpath.n {
        simpath.firstarc[k] = m;

        let v = simpath.vert[k];
        let v_successors: Vec<usize> = graph.successors(v).into_iter().collect();

        for (_ui, u) in v_successors.iter().enumerate() {
            let u_num = simpath.num[*u];
            if u_num > k {
                simpath.arcto[m] = u_num;
                m += 1;
            }
        }
        k += 1;
    }
    simpath.firstarc[k] = m;

    for t in 2..=simpath.n {
        simpath.mate[t] = t;
    }
    let target_num = simpath.num[target];
    simpath.mate[target_num] = 1;
    simpath.mate[1] = target_num;

    let mut jj = 1;
    let mut ll = 1;
    simpath.mem[0] = simpath.mate[1];
    simpath.tail = 0;
    simpath.head = 1;
    simpath.serial = 2;

    let mut firstnode = Vec::new();
    let mut lo = Vec::new();
    let mut hi = Vec::new();

    firstnode.push(0usize);
    lo.push(-1isize);
    lo.push(-1isize);
    hi.push(0isize);
    hi.push(0isize);

    for i in 0..m {
        let i_succ = i + 1;

        firstnode.push(lo.len());

        simpath.boundary = simpath.head;
        simpath.htcount = 0;
        simpath.htid = (i + simpath.wrap) << LOG_MEMSIZE;

        if simpath.htid == 0 {
            for hash in 0..HTSIZE {
                simpath.htable[hash] = 0;
            }
            simpath.wrap += 1;
            simpath.htid = 1 << LOG_MEMSIZE;
        }

        simpath.newserial = simpath.serial + ((simpath.head - simpath.tail) / (ll + 1 - jj));

        let j = jj;
        k = simpath.arcto[i];
        let l = ll;
        while jj <= simpath.n && simpath.firstarc[jj + 1] == i_succ {
            jj += 1;
        }
        ll = if k > l { k } else { l };
        while simpath.tail < simpath.boundary {
            simpath.serial += 1;

            for t in j..=l {
                simpath.mate[t] = simpath.mem[trunc(simpath.tail)];
                if simpath.mate[t] > l {
                    let i = simpath.mate[t];
                    simpath.mate[i] = t;
                }
                simpath.tail += 1;
            }

            let left = printstate(&mut simpath, j, jj, ll);

            let jm = simpath.mate[j];
            let km = simpath.mate[k];

            let right: usize;

            if jm == 0 || km == 0 {
                right = 0;
            } else if jm == k {
                let mut t = j + 1;
                while t <= ll {
                    if t != k {
                        if simpath.mate[t] > 0 && simpath.mate[t] != t {
                            break;
                        }
                    }
                    t += 1;
                }

                right = if t > ll { 1 } else { 0 };
            } else {
                simpath.mate[j] = 0;
                simpath.mate[k] = 0;
                simpath.mate[jm] = km;
                simpath.mate[km] = jm;
                right = printstate(&mut simpath, j, jj, ll);
                simpath.mate[jm] = j;
                simpath.mate[km] = k;
                simpath.mate[j] = jm;
                simpath.mate[k] = km;
            }

            lo.push(left as isize);
            hi.push(right as isize);
        }
    }

    assert!(lo.len() == hi.len());

    firstnode.push(lo.len());

    bdd_reduce(&firstnode, &mut lo, &mut hi);
}

fn bdd_reduce(firstnode: &Vec<usize>, lo: &mut Vec<isize>, hi: &mut Vec<isize>) {
    assert!(lo[0] == -1 && lo[1] == -1);

    for t in (1..firstnode.len() - 1).rev() {
        let mut head = 0isize;

        for k in firstnode[t]..firstnode[t + 1] {
            {
                let q = lo[k];
                assert!(q >= 0);

                let lo_q = lo[q as usize];
                if lo_q >= 0 {
                    lo[k] = lo_q
                }
            }

            {
                let mut q = hi[k];
                assert!(q >= 0);
                let lo_q = lo[q as usize];
                if lo_q >= 0 {
                    q = lo_q;
                    hi[k] = lo_q;
                }
                if q != 0 {
                    assert!(q >= 0);
                    let hi_q = hi[q as usize];
                    if hi_q >= 0 {
                        hi[k] = head.neg();
                        head = q;
                    } else {
                        hi[k] = hi_q.neg();
                    }
                    hi[q as usize] = (k as isize).neg();
                }
            }
        }

        let mut p = head;

        while p != 0 {
            assert!(p >= 0);
            let mut q = hi[p as usize].neg();
            while q > 0 {
                let r = lo[q as usize];
                assert!(r >= 0);
                if lo[r as usize] <= 0 {
                    println!("{:#x}: (~{}?{:#x}:{:#x})", q, t, r, p);

                    lo[r as usize] = q;
                    lo[q as usize] = r.neg() - 1;
                } else {
                    lo[q as usize] = lo[r as usize];
                }

                q = hi[q as usize]
            }
            q = hi[p as usize].neg();
            hi[p as usize] = 0;
            let mut r = 0isize;
            while q > 0 {
                assert!(q >= 0);
                r = lo[q as usize];
                if r < 0 {
                    lo[(r.neg() - 1) as usize] = -1;
                }

                r = q;
                assert!(r >= 0);
                q = hi[r as usize];
            }
            hi[r as usize] = 0;

            p = q.neg();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simpath_star() {
        let graph = BvGraph::with_basename("data/star/star").load().unwrap();
        simpath(&graph, 0, 10000 + 1);
        // assert_eq!(result, 4);
    }
}
