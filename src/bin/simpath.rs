use avgdist_rs::Simpath;
use std::{collections::HashSet, env};
use webgraph::prelude::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let source: usize = args[2].parse().unwrap();
    let target: usize = args[3].parse().unwrap();

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let mut simpath = Simpath::from_webgraph(&graph);

    simpath.init_num_arcto_repr(&graph, source, target, &HashSet::new());

    let (zdd, varsize) = simpath.to_zdd(target);

    let paths = simpath.zdd_all_sols(&zdd, varsize);

    eprintln!("ZDD Paths:\n{:?}", paths);

    zdd.iter().for_each(|(q, t, r, p)| {
        println!("{:#x}: (~{}?{:#x}:{:#x})", *q, *t, *r, *p);
    });
}
