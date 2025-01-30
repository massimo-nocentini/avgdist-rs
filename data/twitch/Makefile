
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 9498 twich < twich.csv
	webgraph transform transpose twich twich-t
	webgraph build ef twich
	webgraph build ef twich-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./twich ./twich-t 1 0.1 true false >> result/twich-01e-true.out
	time cargo run --release --bin avgdist-rs ./twich ./twich-t 1 0.1 false false > result/twich-01e-1k.out
	time cargo run --release --bin avgdist-rs ./twich ./twich-t 10 0.1 false false > result/twich-01e-10k.out
	time cargo run --release --bin avgdist-rs ./twich ./twich-t 100 0.1 false false > result/twich-01e-100k.out
	time cargo run --release --bin avgdist-rs ./twich ./twich-t 346 0.1 false false > result/twich-01e-346k.out

