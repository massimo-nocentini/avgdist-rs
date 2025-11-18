use std::env;
use std::time::Instant;
use webgraph::prelude::*;

struct LouvainBinaryGraph {
    tot: Vec<usize>,
    arcs_weights: Vec<(usize, f64)>,
}

impl LouvainBinaryGraph {
    fn from_webgraph<T: RandomAccessGraph>(agraph: &T) -> Self {
        let num_nodes = agraph.num_nodes();
        let mut links = vec![Vec::new(); num_nodes];

        for src in 0..num_nodes {
            for dest in agraph.successors(src) {
                let weight = 1.0;
                links[src].push((dest, weight));
                if src != dest {
                    links[dest].push((src, weight));
                }
            }
        }

        let mut num_arcs = 0usize;
        for i in 0..links.len() {
            let mut m = std::collections::HashMap::new();

            for &(dest, weight) in links[i].iter() {
                let entry = m.entry(dest).or_insert(0.0);
                *entry += weight;
            }

            links[i] = m.into_iter().collect();
            num_arcs += links[i].len();
        }

        let mut tot = Vec::with_capacity(links.len());
        let mut arcs_weights = Vec::with_capacity(num_arcs);
        let mut cumulative = 0usize;
        for i in links.iter() {
            cumulative += i.len();
            tot.push(cumulative);

            for &j in i.iter() {
                arcs_weights.push(j);
            }
        }

        LouvainBinaryGraph { tot, arcs_weights }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let instant = Instant::now();

    let louvain_graph = LouvainBinaryGraph::from_webgraph(&graph);

    println!("\nTotal time: {:?}", instant.elapsed());
}
