use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::io::Write;
use std::ops::{Add, Div, Mul, Sub};
use std::time::Instant;
use std::{env, fs, io};
use webgraph::prelude::*;

struct LouvainBinaryGraph {
    degrees: Vec<usize>,
    arcs_weights: Vec<(usize, f64)>,
    total_weight: f64,
}

struct LouvainCommunity {
    g: LouvainBinaryGraph,
    tuples: Vec<LouvainTuple>,
    neigh_last: usize,
    min_modularity: f64,
}

impl LouvainBinaryGraph {
    fn nbnodes(&self) -> usize {
        self.degrees.len()
    }

    fn nblinks(&self) -> usize {
        self.arcs_weights.len()
    }

    fn from_webgraph<T: RandomAccessGraph>(graph: &T, directed: bool, weight: f64) -> Self {
        // Build adjacency list representation, this is equivalent to the code
        // in `graph.cpp` that loads a graph from a file.
        let mut links = vec![Vec::new(); graph.num_nodes()];

        for src in 0..links.len() {
            for dest in graph.successors(src) {
                links[src].push((dest, weight));
                if src != dest && !directed {
                    links[dest].push((src, weight));
                }
            }
        }

        // Merge multiple arcs between the same nodes, summing their weights.
        // This is equivalent to the `clean` function in the original C implementation.
        let mut num_arcs = 0usize;
        for i in links.iter_mut() {
            let mut m = std::collections::HashMap::new();

            for &(dest, weight) in i.iter() {
                let entry = m.entry(dest).or_insert(0.0);
                *entry += weight;
            }

            num_arcs += m.len();
            *i = m.into_iter().collect();
        }

        let mut degrees = Vec::with_capacity(links.len());
        let mut arcs_weights = Vec::with_capacity(num_arcs);
        let mut cumulative = 0usize;
        let mut total_weight = 0.0;
        for i in links.iter() {
            cumulative += i.len();
            degrees.push(cumulative);

            for &j in i.iter() {
                arcs_weights.push(j);
                total_weight += j.1;
            }
        }

        degrees.shrink_to_fit();
        arcs_weights.shrink_to_fit();

        LouvainBinaryGraph {
            degrees,
            arcs_weights,
            total_weight,
        }
    }

    fn nb_selfloops(&self, node: usize) -> f64 {
        self.neighbors(node)
            .iter()
            .filter(|&&(neighbor, _)| neighbor == node)
            .map(|&(_, weight)| weight)
            .sum()
    }

    fn weighted_degree(&self, node: usize) -> f64 {
        self.neighbors(node).iter().map(|&(_, weight)| weight).sum()
    }

    fn neighbors(&self, node: usize) -> &[(usize, f64)] {
        if node == 0 {
            &self.arcs_weights[0..self.degrees[0]]
        } else {
            &self.arcs_weights[self.degrees[node - 1]..self.degrees[node]]
        }
    }
}

struct LouvainTuple {
    neigh_weight: f64,
    neigh_pos: usize,
    n2c: usize,
    in_: f64,
    tot: f64,
    node: usize,
}

impl<'a> LouvainCommunity {
    fn size(&self) -> usize {
        self.tuples.len()
    }

    fn from_graph(g: LouvainBinaryGraph, minm: f64) -> Self {
        let tuples: Vec<LouvainTuple> = (0..g.nbnodes())
            .map(|i| LouvainTuple {
                neigh_weight: -1.0,
                neigh_pos: 0,
                n2c: i,
                in_: g.nb_selfloops(i),
                tot: g.weighted_degree(i),
                node: i,
            })
            .collect();

        LouvainCommunity {
            g,
            tuples,
            neigh_last: 0,
            min_modularity: minm,
        }
    }

    fn modularity(&self) -> f64 {
        let m2 = self.g.total_weight;
        self.tuples
            .iter()
            .filter(|each| each.tot > 0.0)
            .fold(0.0, |acc, i| acc + (i.in_ / m2) - (i.tot / m2).powi(2))
    }

    fn remove(&mut self, node: usize, comm: usize, dnodecomm: f64) {
        let tup = &mut self.tuples[comm];
        tup.tot -= self.g.weighted_degree(node);
        tup.in_ -= 2.0.mul(dnodecomm).add(self.g.nb_selfloops(node));

        self.tuples[node].n2c = usize::MAX;
    }

    fn insert(&mut self, node: usize, comm: usize, dnodecomm: f64) {
        let tup = &mut self.tuples[comm];

        tup.tot += self.g.weighted_degree(node);
        tup.in_ += 2.0.mul(dnodecomm).add(self.g.nb_selfloops(node));

        self.tuples[node].n2c = comm;
    }

    fn modularity_gain(&self, comm: usize, dnodecomm: f64, w_degree: f64) -> f64 {
        dnodecomm.sub(self.tuples[comm].tot.mul(w_degree).div(self.g.total_weight))
    }

    fn neigh_comm(&mut self, node: usize) {
        for i in 0..self.neigh_last {
            let j = self.tuples[i].neigh_pos;
            self.tuples[j].neigh_weight = -1.0;
        }
        self.neigh_last = 0;

        let node_n2c = self.tuples[node].n2c;
        self.tuples[0].neigh_pos = node_n2c;
        self.tuples[node_n2c].neigh_weight = 0.0;
        self.neigh_last = 1;

        for &(neigh, neigh_w) in self.g.neighbors(node).iter().filter(|&&(n, _)| n != node) {
            let neigh_comm = self.tuples[neigh].n2c;

            if self.tuples[neigh_comm].neigh_weight == -1.0 {
                self.tuples[neigh_comm].neigh_weight = 0.0;
                self.tuples[self.neigh_last].neigh_pos = neigh_comm;
                self.neigh_last += 1;
            }
            self.tuples[neigh_comm].neigh_weight += neigh_w;
        }
    }

    fn one_level(&mut self, rng: &mut ThreadRng) -> bool {
        let mut random_order: Vec<usize> = (0..self.size()).collect();
        random_order.shuffle(rng);

        let mut improvement = false;
        let mut new_mod = self.modularity();

        loop {
            let cur_mod = new_mod;
            let mut nb_moves = 0usize;

            for &node in random_order.iter() {
                let node_comm = self.tuples[node].n2c;
                let w_degree = self.g.weighted_degree(node);

                // computation of all neighboring communities of current node
                self.neigh_comm(node);
                // remove node from its current community
                self.remove(node, node_comm, self.tuples[node_comm].neigh_weight);

                // compute the nearest community for node
                // default choice for future insertion is the former community
                let mut best_comm = node_comm;
                let mut best_nblinks = 0.0;
                let mut best_increase = 0.0;

                for i in 0..self.neigh_last {
                    let neigh_pos = self.tuples[i].neigh_pos;
                    let neigh_w = self.tuples[neigh_pos].neigh_weight;

                    let increase = self.modularity_gain(neigh_pos, neigh_w, w_degree);

                    if increase > best_increase {
                        best_comm = neigh_pos;
                        best_nblinks = neigh_w;
                        best_increase = increase;
                    }
                }

                // insert node in the nearest community
                self.insert(node, best_comm, best_nblinks);

                if best_comm != node_comm {
                    nb_moves += 1;
                }
            }

            new_mod = self.modularity();

            if nb_moves > 0 {
                improvement = true;
            }

            if (nb_moves == 0) || (new_mod - cur_mod <= self.min_modularity) {
                break;
            }
        }

        improvement
    }

    fn partition2graph_binary(&self) -> LouvainBinaryGraph {
        let mut renumber: Vec<Option<usize>> = vec![None; self.size()];

        for tup in self.tuples.iter() {
            renumber[tup.n2c] = match renumber[tup.n2c] {
                Some(r) => Some(r + 1),
                None => Some(0usize),
            };
        }

        let mut final_comm = 0usize;
        for i in renumber.iter_mut().filter(|&&mut r| r.is_some()) {
            *i = Some(final_comm);
            final_comm += 1;
        }

        let mut comm_nodes = vec![Vec::new(); final_comm];
        for t in self.tuples.iter() {
            if let Some(comm) = renumber[t.n2c] {
                comm_nodes[comm].push(t.node)
            }
        }

        let mut g2_degrees = Vec::with_capacity(comm_nodes.len());
        let mut g2_arcs_weights = Vec::with_capacity(self.g.nblinks());
        let mut g2_total_weight = 0.0;
        for (icomm, comm) in comm_nodes.iter().enumerate() {
            let mut m = std::collections::HashMap::new();

            for &node in comm {
                for &(neigh, neigh_w) in self.g.neighbors(node) {
                    if let Some(neigh_comm) = renumber[self.tuples[neigh].n2c] {
                        let entry = m.entry(neigh_comm).or_insert(0.0);
                        *entry += neigh_w;
                    }
                }
            }

            g2_degrees.push(m.len() + if icomm == 0 { 0 } else { g2_degrees[icomm - 1] });

            for (&neigh_comm, &weight) in m.iter() {
                g2_total_weight += weight;
                g2_arcs_weights.push((neigh_comm, weight));
            }
        }

        g2_degrees.shrink_to_fit();
        g2_arcs_weights.shrink_to_fit();

        LouvainBinaryGraph {
            degrees: g2_degrees,
            arcs_weights: g2_arcs_weights,
            total_weight: g2_total_weight,
        }
    }

    fn display_partition(&self) {
        let mut renumber = vec![None; self.size()];
        for tup in self.tuples.iter() {
            renumber[tup.n2c] = match renumber[tup.n2c] {
                Some(r) => Some(r + 1),
                None => Some(0usize),
            };
        }

        let mut final_comm = 0usize;
        for i in renumber.iter_mut().filter(|i| i.is_some()) {
            *i = Some(final_comm);
            final_comm += 1;
        }

        for i in self.tuples.iter() {
            let n2c = match renumber[i.n2c] {
                Some(r) => r,
                None => usize::MAX,
            };
            println!("{} {}", i.node, n2c);
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let precision = args[2].parse().unwrap();
    let verbose = args[3].parse().unwrap();

    let dot_filename = &args[4];

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let instant = Instant::now();

    let mut g = LouvainBinaryGraph::from_webgraph(&graph, false, 1.0);
    let mut c = LouvainCommunity::from_graph(g, precision);

    for level in 0usize.. {
        eprint!(
            "Level {}: elapsed {:?}, nodes {}, links {}, weight {}, modularity {} ",
            level,
            instant.elapsed(),
            c.g.nbnodes(),
            c.g.nblinks(),
            c.g.total_weight,
            c.modularity()
        );

        let improvement = c.one_level(&mut rng);

        eprintln!("increased to {}.", c.modularity());

        if verbose {
            c.display_partition();
        }

        if !improvement {
            let f = fs::File::create(dot_filename).unwrap();
            let mut writer = io::BufWriter::new(f);

            writer
                .write_all(b"graph G {\n\tnode [shape=circle];\n")
                .unwrap();

            for i in 0..c.g.nbnodes() {
                for &(j, _weight) in c.g.neighbors(i).iter().filter(|&&(n, _)| i < n) {
                    writeln!(writer, "\t{} -- {}", i, j).unwrap();
                }
            }

            writer.write_all(b"}\n").unwrap();

            break;
        }

        g = c.partition2graph_binary();
        c = LouvainCommunity::from_graph(g, precision);
    }

    eprintln!("\nTotal time: {:?}", instant.elapsed());

    if !verbose {
        c.display_partition();
    }
}
