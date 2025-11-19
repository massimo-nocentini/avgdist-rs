use rand::seq::SliceRandom;
use std::env;
use std::time::Instant;
use webgraph::prelude::*;

struct LouvainBinaryGraph {
    degrees: Vec<usize>,
    arcs_weights: Vec<(usize, f64)>,
    nbnodes: usize,
    nblinks: usize,
    total_weight: f64,
}

struct LouvainCommunity {
    g: LouvainBinaryGraph,
    tuples: Vec<LouvainTuple>,
    neigh_last: usize,
    size: usize,
    nb_pass: isize,
    min_modularity: f64,
}

impl LouvainBinaryGraph {
    fn from_webgraph<T: RandomAccessGraph>(graph: &T) -> Self {
        let weight = 1.0;

        let num_nodes = graph.num_nodes();
        let mut links = vec![Vec::new(); num_nodes];

        for src in 0..num_nodes {
            for dest in graph.successors(src) {
                links[src].push((dest, weight));
                if src != dest {
                    links[dest].push((src, weight)); // undirected graph
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

            *i = m.into_iter().collect();
            num_arcs += i.len();
        }

        let mut tot = Vec::with_capacity(links.len());
        let mut arcs_weights = Vec::with_capacity(num_arcs);
        let mut cumulative = 0usize;
        let mut total_weight = 0.0;
        for i in links.iter() {
            cumulative += i.len();
            tot.push(cumulative);

            for &j in i.iter() {
                arcs_weights.push(j);
                total_weight += j.1;
            }
        }

        LouvainBinaryGraph {
            degrees: tot,
            arcs_weights,
            nbnodes: num_nodes,
            nblinks: num_arcs,
            total_weight,
        }
    }

    fn nb_neighbors(&self, node: usize) -> usize {
        if node == 0 {
            self.degrees[0]
        } else {
            self.degrees[node] - self.degrees[node - 1]
        }
    }

    fn nb_selfloops(&self, node: usize) -> f64 {
        for &(neighbor, weight) in self.neighbors(node).iter() {
            if neighbor == node {
                return weight;
            }
        }
        0.0
    }

    fn weighted_degree(&self, node: usize) -> f64 {
        self.neighbors(node).iter().map(|&(_, weight)| weight).sum()
    }

    fn neighbors(&self, node: usize) -> &[(usize, f64)] {
        if node == 0 {
            &self.arcs_weights[..self.degrees[0]]
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
}

impl<'a> LouvainCommunity {
    fn from_graph(g: LouvainBinaryGraph, nbp: isize, minm: f64) -> Self {
        let size = g.nbnodes;

        let mut tuples = Vec::with_capacity(size);

        for i in 0..size {
            tuples.push(LouvainTuple {
                neigh_weight: -1.0,
                neigh_pos: 0,
                n2c: i,
                in_: g.nb_selfloops(i),
                tot: g.weighted_degree(i),
            });
        }

        LouvainCommunity {
            g,
            tuples,
            neigh_last: 0,
            size,
            nb_pass: nbp,
            min_modularity: minm,
        }
    }

    fn modularity(&self) -> f64 {
        let mut q = 0.0;
        let m2 = self.g.total_weight;
        for i in self.tuples.iter().filter(|each| each.tot > 0.0) {
            q += i.in_ / m2 - (i.tot / m2).powi(2);
        }
        q
    }

    fn remove(&mut self, node: usize, comm: usize, dnodecomm: f64) {
        assert!(node < self.size);

        let tup = &mut self.tuples[comm];
        tup.tot -= self.g.weighted_degree(node);
        tup.in_ -= 2.0 * dnodecomm + self.g.nb_selfloops(node);

        self.tuples[node].n2c = usize::MAX;
    }

    fn insert(&mut self, node: usize, comm: usize, dnodecomm: f64) {
        assert!(node < self.size);
        let tup = &mut self.tuples[comm];

        tup.tot += self.g.weighted_degree(node);
        tup.in_ += 2.0 * dnodecomm + self.g.nb_selfloops(node);

        self.tuples[node].n2c = comm;
    }

    fn modularity_gain(&self, node: usize, comm: usize, dnodecomm: f64, w_degree: f64) -> f64 {
        assert!(node < self.size);

        let totc = self.tuples[comm].tot;
        let degc = w_degree;
        let m2 = self.g.total_weight;
        let dnc = dnodecomm;

        dnc - (totc * degc) / m2
    }

    fn neigh_comm(&mut self, node: usize) {
        for i in 0..self.neigh_last {
            self.tuples[i].neigh_weight = -1.0;
        }
        self.neigh_last = 0;

        let neighbors = self.g.neighbors(node);
        let deg = self.g.nb_neighbors(node);

        self.tuples[0].neigh_pos = self.tuples[node].n2c;
        let j0 = self.tuples[0].neigh_pos;
        self.tuples[j0].neigh_weight = 0.0;
        self.neigh_last = 1;

        for i in 0..deg {
            let (neigh, neigh_w) = neighbors[i];
            let neigh_comm = self.tuples[neigh].n2c;

            if neigh != node {
                if self.tuples[neigh_comm].neigh_weight == -1.0 {
                    self.tuples[neigh_comm].neigh_weight = 0.0;
                    self.tuples[self.neigh_last].neigh_pos = neigh_comm;
                    self.neigh_last += 1;
                }
                self.tuples[neigh_comm].neigh_weight += neigh_w;
            }
        }
    }

    fn one_level(&mut self) -> bool {
        let mut improvement = false;
        let mut nb_moves;
        let mut new_mod = self.modularity();

        let mut rng = rand::thread_rng();

        let mut random_order: Vec<usize> = (0..self.size).collect();
        random_order.shuffle(&mut rng);

        loop {
            let cur_mod = new_mod;
            nb_moves = 0usize;

            for node_tmp in 0..self.size {
                let node = random_order[node_tmp];
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
                    let increase = self.modularity_gain(
                        node,
                        self.tuples[i].neigh_pos,
                        self.tuples[i].neigh_weight,
                        w_degree,
                    );
                    if increase > best_increase {
                        best_comm = neigh_pos;
                        best_nblinks = self.tuples[neigh_pos].neigh_weight;
                        best_increase = increase;
                    }
                }

                // insert node in the nearest community
                self.insert(node, best_comm, best_nblinks);

                if best_comm != node_comm {
                    nb_moves += 1;
                }
            }

            // let mut total_tot = 0.0;
            // let mut total_in = 0.0;
            // for i in 0..self.tuples.len() {
            //     total_tot += self.tuples[i].tot;
            //     total_in += self.tuples[i].in_;
            // }

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
        let mut renumber = vec![-1isize; self.size];
        for tup in self.tuples.iter() {
            renumber[tup.n2c] += 1;
        }

        let mut final_comm = 0usize;
        for i in 0..self.size {
            if renumber[i] != -1 {
                renumber[i] = final_comm as isize;
                final_comm += 1;
            }
        }

        let mut comm_nodes = vec![Vec::new(); final_comm];
        for node in 0..self.size {
            let comm = renumber[self.tuples[node].n2c] as usize;
            comm_nodes[comm].push(node);
        }

        let mut g2_degrees = Vec::with_capacity(final_comm);
        let mut g2_arcs_weights = Vec::new();
        let mut g2_nblinks = 0usize;
        let mut g2_total_weight = 0.0;
        for comm in 0..final_comm {
            let mut m = std::collections::HashMap::new();

            for &node in comm_nodes[comm].iter() {
                let neighbors = self.g.neighbors(node);
                let deg = self.g.nb_neighbors(node);

                for i in 0..deg {
                    let (neigh, neigh_w) = neighbors[i];
                    let neigh_comm = renumber[self.tuples[neigh].n2c] as usize;

                    let entry = m.entry(neigh_comm).or_insert(0.0);
                    *entry += neigh_w;
                }
            }

            g2_nblinks += m.len();
            if comm == 0 {
                g2_degrees.push(m.len());
            } else {
                g2_degrees.push(g2_degrees[comm - 1] + m.len());
            }

            for (&neigh_comm, &weight) in m.iter() {
                g2_total_weight += weight;
                g2_arcs_weights.push((neigh_comm, weight));
            }
        }

        LouvainBinaryGraph {
            degrees: g2_degrees,
            arcs_weights: g2_arcs_weights,
            nbnodes: final_comm,
            nblinks: g2_nblinks,
            total_weight: g2_total_weight,
        }
    }

    fn update(&mut self, g: LouvainBinaryGraph, nbp: isize, minm: f64) {
        self.size = g.nbnodes;

        self.tuples.clear();
        for i in 0..self.size {
            self.tuples.push(LouvainTuple {
                neigh_weight: -1.0,
                neigh_pos: 0,
                n2c: i,
                in_: g.nb_selfloops(i),
                tot: g.weighted_degree(i),
            });
        }

        self.g = g;
        self.nb_pass = nbp;
        self.min_modularity = minm;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let precision = args[2].parse().unwrap();

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let instant = Instant::now();

    let louvain_graph = LouvainBinaryGraph::from_webgraph(&graph);
    let mut louvain_community = LouvainCommunity::from_graph(louvain_graph, -1, precision);

    let mut mod_ = louvain_community.modularity();
    let mut level = 0usize;

    loop {
        eprintln!(
            "Level {}: eta {:?}, nodes {}, links {}, weight {}.",
            level,
            instant.elapsed(),
            louvain_community.g.nbnodes,
            louvain_community.g.nblinks,
            louvain_community.g.total_weight
        );

        let improvement = louvain_community.one_level();
        let new_mod = louvain_community.modularity();

        level += 1;

        let g2 = louvain_community.partition2graph_binary();
        louvain_community.update(g2, -1, precision);

        eprintln!("  modularity increased from {} to {}", mod_, new_mod);

        mod_ = new_mod;

        if improvement {
            break;
        }
    }

    println!("\nTotal time: {:?}", instant.elapsed());
}
