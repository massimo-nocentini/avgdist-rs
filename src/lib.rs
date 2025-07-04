use rand::Rng;
use std::io::{self, Write};
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
    maxn: usize,
    maxm: u64,
    mem: Vec<usize>,
    tail: usize,
    boundary: usize,
    head: usize,
    htable: Vec<u32>,
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

fn printstate(a: usize, b: usize, c: usize) {}

pub fn simpath<T: RandomAccessGraph + Send + Sync + 'static>(
    graph: &T,
    source: usize,
    target: usize,
) {
    let num_nodes = graph.num_nodes();

    let mut simpath = Simpath {
        maxn: num_nodes,
        maxm: graph.num_arcs(),
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
    };

    for v in 0..num_nodes {
        let a: Vec<usize> = graph.successors(v).into_iter().collect();
        for (i, u) in a.iter().enumerate() {
            if *u == v {
                panic!("Self-loop detected at node {}", v);
            }

            // let b = a[if v < *u { i + 1 } else { i - 1 }];
            // if b != v {
            //     panic!(
            //         "Sorry, the graph isn't undirected! ({:?} -> {:?} has mate pointing to {:?})",
            //         v, u, b
            //     );
            // }
        }
    }

    // if (source == g->vertices)
    // {
    //     for (k = 0; k < n; k++)
    //         (g->vertices + k)->num = k + 1, vert[k + 1] = g->vertices + k;
    // }
    if source == 0 {
        for k in 0..num_nodes {
            simpath.num[k] = k + 1;
            simpath.vert[k + 1] = k;
        }
    } else {
        // for (k = 0; k < n; k++)
        //     (g->vertices + k)->num = 0;
        // for k in 0..num_nodes {
        //     simpath.num[k] = 0;
        // }

        // vert[1] = source, source->num = 1;
        simpath.vert[1] = source;
        simpath.num[source] = 1;

        // for (j = 0, k = 1; j < k; j++)
        // {
        //     v = vert[j + 1];
        //     for (a = v->arcs; a; a = a->next)
        //     {
        //         u = a->tip;
        //         if (u->num == 0)
        //             u->num = ++k, vert[k] = u;
        //     }
        // }
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

        // if (target->num == 0)
        // {
        //     fprintf(stderr, "Sorry, there's no path from %s to %s in the graph!\n",
        //             argv[2], argv[3]);
        //     exit(-8);
        // }
        for v in simpath.num.iter().filter(|&each| *each > 0) {
            println!("{} ", simpath.vert[*v]);
        }

        if simpath.num[target] == 0 {
            panic!(
                "Sorry, there's no path from {} to {} in the graph!",
                source, target
            );
        }

        // if (k < n)
        // {
        //     fprintf(stderr, "The graph isn't connected (%d<%d)!\n", k, n);
        //     fprintf(stderr, "But that's OK; I'll work with the component of %s.\n",
        //             argv[2]);
        //     n = k;
        // }
        if k < num_nodes {
            println!("The graph isn't connected ({} < {})! But that's OK; I'll work with the component of {}.", k, num_nodes, source);
        }
    }

    {
        // for (m = 0, k = 1; k <= n; k++)
        // {
        //     firstarc[k] = m;
        //     v = vert[k];
        //     printf("%ld(%s)\n", v->num, v->name);
        //     for (a = v->arcs; a; a = a->next)
        //     {
        //         u = a->tip;
        //         if (u->num > k)
        //         {
        //             arcto[m++] = u->num;
        //             if (a->len == 1)
        //                 printf(" -> %ld(%s) #%d\n", u->num, u->name, m);
        //             else
        //                 printf(" -> %ld(%s,%ld) #%d\n", u->num, u->name, a->len, m);
        //         }
        //     }
        // }
        // firstarc[k] = m;

        let mut m = 0usize;
        let mut k = 1;
        while k <= num_nodes {
            simpath.firstarc[k] = m;
            let v = simpath.vert[k];
            println!("{}({})", simpath.num[v], v);
            let v_successors: Vec<usize> = graph.successors(v).into_iter().collect();
            let v_successors_len = v_successors.len() - 1;
            for (ui, u) in v_successors.iter().enumerate() {
                let u_num = simpath.num[*u];
                if u_num > k {
                    simpath.arcto[m] = u_num;
                    m += 1;
                    if ui == v_successors_len {
                        println!(" -> {}({}) #{}", u_num, u, m);
                    } else {
                        println!(" -> {}({}, {}) #{}", u_num, u, v_successors_len - ui, m);
                    }
                }
            }
            k += 1;
        }
        simpath.firstarc[k] = m;

        // for (t = 2; t <= n; t++)
        //     mate[t] = t;
        // mate[target->num] = 1, mate[1] = target->num;
        for t in 2..=num_nodes {
            simpath.mate[t] = t;
        }
        let target_num = simpath.num[target];
        simpath.mate[target_num] = 1;
        simpath.mate[1] = target_num;

        // jj = ll = 1;
        // mem[0] = mate[1];
        // tail = 0, head = 1;
        // serial = 2;
        let mut jj = 1;
        let mut ll = 1;
        simpath.mem[0] = simpath.mate[1];
        simpath.tail = 0;
        simpath.head = 1;
        simpath.serial = 2;

        for i in 0..m {
            println!("#{}:", i + 1);
            println!(
                "Beginning arc {} (serial={}, head-tail={})",
                i + 1,
                simpath.serial,
                simpath.head - simpath.tail
            );

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

            simpath.newserial =
                simpath.serial + ((simpath.head - simpath.tail) / (ll + 1usize - jj));

            let j = jj;
            k = simpath.arcto[i];
            let l = ll;
            while jj <= num_nodes && simpath.firstarc[jj + 1] == i + 1 {
                jj += 1;
            }
            ll = if k > l { k } else { l };
            while simpath.tail < simpath.boundary {
                print!("{}:", simpath.serial);
                simpath.serial += 1;

                for t in j..=l {
                    simpath.mate[t] = simpath.mem[trunc(simpath.tail)];
                    if simpath.mate[t] > l {
                        let i = simpath.mate[t];
                        simpath.mate[i] = t;
                    }
                    simpath.tail += 1;
                }

                printstate(j, jj, ll);

                print!(",");

                let jm = simpath.mate[j];
                let km = simpath.mate[k];
                if jm == 0 || km == 0 {
                    print!("0");
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

                    if t > ll {
                        print!("1");
                    } else {
                        print!("0");
                    }
                } else {
                    simpath.mate[j] = 0;
                    simpath.mate[k] = 0;
                    simpath.mate[jm] = km;
                    simpath.mate[km] = jm;
                    printstate(j, jj, ll);
                    simpath.mate[jm] = j;
                    simpath.mate[km] = k;
                    simpath.mate[j] = jm;
                    simpath.mate[k] = km;
                }

                println!("");
            }
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
