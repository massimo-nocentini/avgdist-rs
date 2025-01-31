
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 281903 stanford < stanford.csv
	webgraph transform transpose stanford stanford-t
	webgraph build ef stanford
	webgraph build ef stanford-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./stanford ./stanford-t 1 0.1 true false > result/stanford-01e-true.out
	time cargo run --release --bin avgdist-rs ./stanford ./stanford-t 1 0.1 false false > result/stanford-01e-1k.out
	time cargo run --release --bin avgdist-rs ./stanford ./stanford-t 10 0.1 false false > result/stanford-01e-10k.out
	time cargo run --release --bin avgdist-rs ./stanford ./stanford-t 100 0.1 false false > result/stanford-01e-100k.out
	time cargo run --release --bin avgdist-rs ./stanford ./stanford-t 346 0.1 false false > result/stanford-01e-346k.out

