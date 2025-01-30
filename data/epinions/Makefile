# Directed graph (each unordered pair of nodes is saved once): soc-Epinions1.txt 
# Directed Epinions social network
# Nodes: 75879 Edges: 508837
# FromNodeId	ToNodeId
# From https://snap.stanford.edu/data/soc-Epinions1.html

echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 75879 epinions < epinions.csv
	webgraph transform transpose epinions epinions-t
	webgraph build ef epinions
	webgraph build ef epinions-t

run:
	mkdir -p result
	#time cargo run --release --bin avgdist-rs epinions epinions-t 222 0.1 false true > result/epinions-01e-d.out
	time cargo run --release --bin avgdist-rs ./epinions ./epinions-t 1 0.1 false false > result/epinions-01e-1k.out
	time cargo run --release --bin avgdist-rs ./epinions ./epinions-t 10 0.1 false false > result/epinions-01e-10k.out
	time cargo run --release --bin avgdist-rs ./epinions ./epinions-t 100 0.1 false false > result/epinions-01e-100k.out
	time cargo run --release --bin avgdist-rs ./epinions ./epinions-t 346 0.1 false false > result/epinions-01e-346k.out
	time cargo run --release --bin avgdist-rs ./epinions ./epinions-t 1 0.1 true false > result/epinions-01e-true.out

