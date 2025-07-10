use avgdist_rs::simpath;
use std::env;
use webgraph::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let source: usize = args[2].parse().unwrap();
    let target: usize = args[3].parse().unwrap();

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let zdd = simpath(&graph, source, target);

    zdd.iter().for_each(|(q, t, r, p)| {
        println!("{:#x}: (~{}?{:#x}:{:#x})", *q, *t, *r, *p);
    });
}
