
webgraph from arcs --exact --num-nodes 1000 1k-0001p < 1k-0.001p.csv
webgraph transform transpose 1k-0001p 1k-0001p-t
webgraph build ef 1k-0001p
webgraph build ef 1k-0001p-t

webgraph from arcs --exact --num-nodes 2000 2k-0001p < 2k-0.001p.csv
webgraph transform transpose 2k-0001p 2k-0001p-t
webgraph build ef 2k-0001p
webgraph build ef 2k-0001p-t

webgraph from arcs --exact --num-nodes 5000 5k-0001p < 5k-0.001p.csv
webgraph transform transpose 5k-0001p 5k-0001p-t
webgraph build ef 5k-0001p
webgraph build ef 5k-0001p-t

