use avgdist_rs::simpath;
use std::env;
use std::time::Instant;
use webgraph::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let source: usize = args[2].parse().unwrap();
    let target: usize = args[3].parse().unwrap();

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let instant = Instant::now();

    simpath(&graph, source, target);

    println!("Finished in {:?}.", instant.elapsed());
}
