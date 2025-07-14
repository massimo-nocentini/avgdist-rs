use core::panic;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::env;
use std::io::{self, Write};
use std::ops::{Neg, Not};
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

pub struct Simpath {
    n: usize,
    m: usize,
    mem: Vec<usize>,
    tail: usize,
    boundary: usize,
    head: usize,
    htable: Vec<usize>,
    htid: usize,
    htcount: usize,
    wrap: usize,
    vert: Vec<usize>,
    num: HashMap<usize, usize>,
    arcto: Vec<usize>,
    firstarc: Vec<usize>,
    mate: Vec<usize>,
    serial: usize,
    newserial: usize,
    log_memsize: usize,
    memsize: usize,
    htsize: usize,
}

impl Simpath {
    fn trunc(&self, addr: usize) -> usize {
        addr & (self.memsize - 1)
    }

    fn printstate(&mut self, j: usize, jj: usize, ll: usize) -> usize {
        let mut h: usize;
        let mut hh: usize;
        let mut t: usize;
        let mut tt: usize;
        let mut hash: usize;

        t = j;
        while t < jj {
            if self.mate[t] > 0 && self.mate[t] != t {
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
            if self.head + ss - self.tail > self.memsize {
                panic!(
                    "Oops, I'm out of memory (memsize={}, serial={})!",
                    self.memsize, self.serial
                );
            }

            t = jj;
            h = self.trunc(self.head);
            hash = 0;
            while t <= ll {
                self.mem[h] = self.mate[t];
                hash = hash * 31415926525 + self.mate[t];

                t += 1;
                h = self.trunc(h + 1)
            }

            hash = hash & (self.htsize - 1);
            loop {
                hh = self.htable[hash];

                if (hh ^ self.htid) >= self.memsize {
                    self.htcount += 1;
                    if self.htcount > (self.htsize >> 1) {
                        panic!(
                            "Sorry, the hash table is full (htsize={}, serial={})!",
                            self.htsize, self.serial
                        );
                    }
                    hh = self.trunc(self.head);
                    self.htable[hash] = self.htid + hh;
                    self.head += ss;
                    break;
                }

                hh = self.trunc(hh);
                t = hh;
                h = self.trunc(self.head);
                tt = self.trunc(t + ss - 1);

                let should_continue = loop {
                    if self.mem[t] != self.mem[h] {
                        break true;
                    }

                    if t == tt {
                        break false;
                    }

                    t = self.trunc(t + 1);
                    h = self.trunc(h + 1);
                };

                if should_continue {
                    hash = (hash + 1) & (self.htsize - 1);
                } else {
                    break;
                }
            }

            h = self.trunc(hh - self.boundary) / ss;
            ret = self.newserial + h;
        }

        return ret;
    }

    pub fn init_num_arcto_repr<T: RandomAccessGraph>(
        &mut self,
        graph: &T,
        source: usize,
        target: usize,
        subgraph: &Option<HashSet<usize>>,
    ) {
        let instant = Instant::now();
        // if source == 0 {
        //     let mut k = 0;
        //     while k < self.n {
        //         let k_succ = k + 1;
        //         self.num[k] = k_succ;
        //         self.vert[k_succ] = k;
        //         k = k_succ;
        //     }
        // } else
        {
            self.vert.push(0);
            self.vert.push(source);
            self.num.insert(source, 1);

            let mut k = 1;
            let mut j = 0;
            while j < k {
                j += 1;
                let v = self.vert[j];

                for u in graph.successors(v) {
                    if (subgraph.is_none())
                        || (subgraph.is_some() && subgraph.as_ref().unwrap().contains(&u))
                            && self.num.contains_key(&u).not()
                    {
                        k += 1;
                        self.vert.push(u);
                        self.num.insert(u, k);
                    }
                }
            }

            if self.num.contains_key(&target).not() {
                let reachable = self
                    .num
                    .iter()
                    .filter(|&(each, _)| *each > 0)
                    .map(|(each, _)| self.vert[*each])
                    .collect::<Vec<usize>>();

                eprintln!("Vertices reachables from {}: {:?}", source, reachable);

                panic!(
                    "Sorry, there's no path from {} to {} in the graph!",
                    source, target
                );
            }

            if k < self.n {
                eprintln!("The graph isn't connected! But that's OK; I'll work with the {}'s component of size {} (|V| = {}).", source, k, self.n);
                self.n = k;
            } else {
                eprintln!("The graph is connected ({} vertices)!", k);
            }
        }

        let mut m = 0;
        let mut k = 1;
        self.firstarc.push(0);
        while k <= self.n {
            self.firstarc.push(m);

            let v = self.vert[k];

            for u in graph.successors(v) {
                if subgraph.is_none() || subgraph.as_ref().unwrap().contains(&u) {
                    match self.num.get(&u) {
                        Some(&u_num) => {
                            if u_num > k {
                                self.arcto.push(u_num);
                                m += 1;
                            }
                        }
                        None => panic!("Vertex {} not found in num map!", u),
                    }
                }
            }
            k += 1;
        }

        self.firstarc.push(m);
        assert!(
            self.firstarc.len() - 1 == k,
            "firstarc length mismatch: {} != {}",
            self.firstarc.len() - 1,
            k
        );
        self.m = m;

        eprintln!(
            "Finished reading the component ({} vertices, {} arcs) in {:?}.",
            k - 1,
            m,
            instant.elapsed()
        );
    }

    pub fn from_webgraph<T: RandomAccessGraph>(graph: &T) -> Self {
        let default_memsiz: usize = 30; // Default memory size if not set in the environment
        let base_memsiz: usize = match env::var("BASE_MEMSIZE") {
            Ok(var) => match var.parse::<usize>() {
                Ok(size) => size,
                Err(_) => {
                    eprintln!(
                        "BASE_MEMSIZE is not a valid usize, using default value {}.",
                        default_memsiz
                    );
                    default_memsiz
                }
            },
            Err(_) => {
                eprintln!(
                    "BASE_MEMSIZE is not a valid usize, using default value {}.",
                    default_memsiz
                );
                default_memsiz
            }
        };

        let num_nodes = graph.num_nodes();

        let log_memsize: usize = base_memsiz;
        let memsize: usize = 1 << log_memsize;

        eprintln!(
            "Allocating two vectors of memory size {} GB.",
            (memsize * size_of::<usize>()) as f64 / 1e9
        );

        Simpath {
            n: num_nodes,
            m: 0,
            mem: vec![0; memsize],
            tail: 0,
            boundary: 0,
            head: 0,
            htable: vec![0; memsize],
            htid: 0,
            htcount: 0,
            wrap: 1,
            vert: Vec::new(),
            num: HashMap::new(),
            arcto: Vec::new(),
            firstarc: Vec::new(),
            mate: Vec::new(),
            serial: 0,
            newserial: 0,
            log_memsize,
            memsize,
            htsize: memsize,
        }
    }

    pub fn to_zdd(&mut self, target: usize) -> (Vec<(usize, usize, usize, usize)>, usize) {
        assert!(
            self.num.contains_key(&target),
            "Target vertex {} not found in num map!",
            target
        );

        let target_num = *self.num.get(&target).unwrap();

        assert!(
            target_num < self.n,
            "Target vertex {} is out of bounds (n = {})",
            target_num,
            self.n
        );

        self.mate.push(0);
        self.mate.push(target_num);
        for t in 2..=self.n {
            self.mate.push(t);
        }
        self.mate[target_num] = 1;

        let mut jj = 1;
        let mut ll = 1;
        self.mem[0] = self.mate[1];
        self.tail = 0;
        self.head = 1;
        self.serial = 2;

        eprintln!("Setup mates complete.");

        let mut firstnode = Vec::new();
        let mut lo = Vec::new();
        let mut hi = Vec::new();

        firstnode.push(0usize);
        lo.push(-1isize);
        lo.push(-1isize);
        hi.push(0isize);
        hi.push(0isize);

        for i in 0..self.m {
            let i_succ = i + 1;

            firstnode.push(lo.len());

            self.boundary = self.head;
            self.htcount = 0;
            self.htid = (i + self.wrap) << self.log_memsize;

            if self.htid == 0 {
                eprintln!("Initializing the hash table (wrap = {})...", self.wrap);
                for hash in 0..self.htsize {
                    self.htable[hash] = 0;
                }
                self.wrap += 1;
                self.htid = 1 << self.log_memsize;
            }

            self.newserial = self.serial + ((self.head - self.tail) / (ll + 1 - jj));

            let j = jj;
            let k = self.arcto[i];
            let l = ll;
            while jj <= self.n && self.firstarc[jj + 1] == i_succ {
                jj += 1;
            }
            ll = if k > l { k } else { l };

            let bdd_nodes = lo.len();

            while self.tail < self.boundary {
                self.serial += 1;

                for t in j..=l {
                    self.mate[t] = self.mem[self.trunc(self.tail)];
                    if self.mate[t] > l {
                        let i = self.mate[t];
                        self.mate[i] = t;
                    }
                    self.tail += 1;
                }

                let left = self.printstate(j, jj, ll);

                let jm = self.mate[j];
                let km = self.mate[k];

                let right: usize;

                if jm == 0 || km == 0 {
                    right = 0;
                } else if jm == k {
                    let mut t = j + 1;
                    while t <= ll {
                        if t != k {
                            if self.mate[t] > 0 && self.mate[t] != t {
                                break;
                            }
                        }
                        t += 1;
                    }

                    right = if t > ll { 1 } else { 0 };
                } else {
                    self.mate[j] = 0;
                    self.mate[k] = 0;
                    self.mate[jm] = km;
                    self.mate[km] = jm;
                    right = self.printstate(j, jj, ll);
                    self.mate[jm] = j;
                    self.mate[km] = k;
                    self.mate[j] = jm;
                    self.mate[k] = km;
                }

                lo.push(left as isize);
                hi.push(right as isize);
            }

            eprintln!(
                "Finished processing arc {} of {} (added {} bdd nodes).",
                i + 1,
                self.m,
                lo.len() - bdd_nodes
            );
        }

        assert!(lo.len() == hi.len());

        eprintln!("Setup BDD vectors complete of length {}.", lo.len());

        firstnode.push(lo.len());

        let zdd = bdd_reduce(&firstnode, &mut lo, &mut hi);

        eprintln!(
            "Finished BDD reduction, resulting in {} ZDD nodes.",
            zdd.0.len()
        );

        zdd
    }

    pub fn zdd_all_sols(
        &mut self,
        zdd: &Vec<(usize, usize, usize, usize)>,
        varsize: usize,
    ) -> Vec<Vec<usize>> {
        let mut root = 0usize;
        let mut mem = HashMap::new();

        let mut minv = varsize;
        let mut present = HashSet::new();

        for (ii1, i2, i3, i4) in zdd.iter() {
            let i1 = *ii1 as usize;
            // if lo(mem[i1]) != 0 || hi(mem[i1]) != 0 {
            //     panic!(
            //         "! clobbered node in the tuple: ({}, {}, {}, {})",
            //         i1,
            //         v(mem[i1]),
            //         lo(mem[i1]),
            //         hi(mem[i1])
            //     );
            // }

            if *i2 < minv {
                minv = *i2;
                root = i1;
            }

            mem.insert(i1, (*i2, *i3, *i4));

            present.insert(*i2);
        }

        eprintln!(
            "There are {} ZDD nodes and {} ZDD variables.",
            zdd.len(),
            present.len()
        );

        let mut paths = Vec::new();
        let mut stack = Vec::new();

        mem.insert(0, (varsize, 0, 0));
        mem.insert(1, (varsize, 0, 0));
        if root > 0 {
            self.zdd_paths(root, &mut stack, &mem, &mut paths);
        }

        paths
    }

    fn zdd_paths(
        &mut self,
        p: usize,
        stack: &mut Vec<usize>,
        mem: &HashMap<usize, (usize, usize, usize)>,
        paths: &mut Vec<Vec<usize>>,
    ) {
        if p <= 1 {
            paths.push(stack.iter().map(|each| self.vert[*each]).collect());
        } else {
            let tup = *mem.get(&p).unwrap();
            let mut q = lo(tup);
            if q > 0 {
                self.zdd_paths(q, stack, mem, paths);
            }

            q = hi(tup);
            if q > 0 {
                stack.push(v(tup));
                self.zdd_paths(q, stack, mem, paths);
                stack.pop();
            }
        }
    }
}

fn bdd_reduce(
    firstnode: &Vec<usize>,
    lo: &mut Vec<isize>,
    hi: &mut Vec<isize>,
) -> (Vec<(usize, usize, usize, usize)>, usize) {
    let mut zdd = Vec::new();
    let mut maxv = 0usize;

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
                    zdd.push((q as usize, t, r as usize, p as usize));
                    maxv = std::cmp::max(maxv, t);

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

    (zdd, maxv)
}

fn v(tup: (usize, usize, usize)) -> usize {
    tup.0
}

fn lo(tup: (usize, usize, usize)) -> usize {
    tup.1
}

fn hi(tup: (usize, usize, usize)) -> usize {
    tup.2
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn simpath_star() {
//         let graph = BvGraph::with_basename("data/star/star").load().unwrap();
//         simpath(&graph, 0, 10000 + 1);
//         // assert_eq!(result, 4);
//     }
// }
