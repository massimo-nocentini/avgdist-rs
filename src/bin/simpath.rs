use avgdist_rs::{simpath, zdd_all_sols};
use std::env;
use webgraph::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let source: usize = args[2].parse().unwrap();
    let target: usize = args[3].parse().unwrap();

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let subgraph = None;

    let (zdd, varsize, maxq) = simpath(&graph, source, target, &subgraph);

    let paths = zdd_all_sols(&zdd, varsize, maxq);

    eprintln!("ZDD Paths:\n{:?}", paths);

    zdd.iter().for_each(|(q, t, r, p)| {
        println!("{:#x}: (~{}?{:#x}:{:#x})", *q, *t, *r, *p);
    });
}
