use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use std::{env, io, usize};
use webgraph::prelude::*;

#[derive(Debug, Deserialize)]
struct Record {
    node: usize,
    comunity: usize,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];
    let index: usize = args[2].parse().unwrap();

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b' ')
        .has_headers(false)
        .from_reader(io::stdin());

    let mut comunity = HashMap::new();
    let mut inverse = HashMap::new();
    let mut last = 0;
    let mut count = 0;

    for result in rdr.deserialize() {
        let record: Record = result.unwrap();

        if last > record.node {
            count += 1;
        }

        if count == index {
            let e = inverse.entry(record.comunity).or_insert(Vec::new());
            e.push(record.node);
            comunity.insert(record.node, record.comunity);
        }

        last += record.node;
    }

    let mut arcs = HashSet::new();

    for (&c, nodes) in inverse.iter() {
        for &node in nodes.iter() {
            for neighbor in graph.successors(node) {
                arcs.insert((c, comunity[&neighbor]));
            }
        }
    }

    for (src, dst) in arcs.iter() {
        println!("{} {}", src, dst);
    }

    let instant = Instant::now();
}
