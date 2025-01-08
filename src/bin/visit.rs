use dsi_progress_logger::no_logging;
use rand::random;
use std::env;
use std::ops::ControlFlow::Continue;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use webgraph::prelude::BvGraph;
use webgraph::traits::RandomAccessGraph;
use webgraph::traits::SequentialLabeling;
use webgraph_algo::algo::visits::Parallel;
use webgraph_algo::{prelude::breadth_first::EventNoPred, threads};

fn visit<G>(root: usize, graph: G) -> Vec<(usize, usize)>
where
    G: RandomAccessGraph + Send + Sync,
{
    let mut d = Vec::with_capacity(graph.num_nodes());

    for _ in 0..graph.num_nodes() {
        d.push(AtomicUsize::new(0usize));
    }

    let mut visit = webgraph_algo::algo::visits::breadth_first::ParFairNoPred::new(graph, 1);

    visit
        .par_visit(
            root,
            |event| {
                if let EventNoPred::Unknown { curr, distance, .. } = event {
                    d[curr].store(distance, Ordering::Relaxed);
                }

                Continue::<G>(())
            },
            &threads![],
            no_logging![],
        )
        .continue_value();

    d.iter().enumerate().map(|(v, d)| (v, d.load(Ordering::Relaxed))).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let graph_filename = &args[1];

    let graph = BvGraph::with_basename(graph_filename).load().unwrap();

    let root = (random::<f64>() * (graph.num_nodes() as f64)).floor() as usize;

    visit(root, graph);
}
