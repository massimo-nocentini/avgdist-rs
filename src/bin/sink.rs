
use std::time::Instant;
use std::env;
use webgraph::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let num_nodes = graph.num_nodes();

    let mut sinks = 0usize;

    let instant = Instant::now();

    for i in 0..num_nodes {
        if graph.successors(i).len() == 0 {
            sinks += 1;
        }
    }

    println!(
        "|V| = {}, |E| = {}, |S| = {}, eta = {:?}",
        num_nodes,
        graph.num_arcs(),
        sinks,
        instant.elapsed()
    );
}
