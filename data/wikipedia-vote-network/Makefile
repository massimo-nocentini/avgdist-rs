
# Directed graph (each unordered pair of nodes is saved once): Wiki-Vote.txt 
# Wikipedia voting on promotion to administratorship (till January 2008). Directed edge A->B means user A voted on B becoming Wikipedia administrator.
# Nodes: 7115 Edges: 103689
# FromNodeId	ToNodeId

echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 7115 wiki < wiki.csv
	webgraph transform transpose wiki wiki-t
	webgraph build ef wiki
	webgraph build ef wiki-t

run:
	mkdir -p result
	#time cargo run --release --bin avgdist-rs wiki wiki-t 222 0.1 false true > result/wiki-01e-d.out
	time cargo run --release --bin avgdist-rs ./wiki ./wiki-t 1 0.1 false false > result/wiki-01e-1k.out
	time cargo run --release --bin avgdist-rs ./wiki ./wiki-t 10 0.1 false false > result/wiki-01e-10k.out
	time cargo run --release --bin avgdist-rs ./wiki ./wiki-t 100 0.1 false false > result/wiki-01e-100k.out
	time cargo run --release --bin avgdist-rs ./wiki ./wiki-t 346 0.1 false false > result/wiki-01e-346k.out
	time cargo run --release --bin avgdist-rs ./wiki ./wiki-t 1 0.1 true false > result/wiki-01e-true.out

