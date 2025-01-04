use dsi_progress_logger::no_logging;
use std::ops::ControlFlow::Continue;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use webgraph::graphs::vec_graph::VecGraph;
use webgraph::labels::proj::Left;
use webgraph::prelude::BvGraph;
use webgraph::traits::SequentialLabeling;
use webgraph_algo::algo::visits::breadth_first::{self, *};
use webgraph_algo::algo::visits::Parallel;
use webgraph_algo::{
    prelude::breadth_first::{EventNoPred, ParFairNoPred},
    threads,
};

fn main() {
    let graph_filename = "";

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let mut d = Vec::with_capacity(graph.num_nodes());

    for _ in 0..graph.num_nodes() {
        d.push(AtomicUsize::new(usize::MAX));
    }

    let mut visit = webgraph_algo::algo::visits::breadth_first::ParFairNoPred::new(graph, 1);
    visit
        .par_visit(
            0,
            |event| {
                // Set distance from 0
                if let EventNoPred::Unknown { curr, distance, .. } = event {
                    d[curr].store(distance, Ordering::Relaxed);
                }
                Continue(())
            },
            &threads![],
            no_logging![],
        )
        .continue_value();
}
 