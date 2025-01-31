
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 36692 enron < enron.csv
	webgraph transform transpose enron enron-t
	webgraph build ef enron
	webgraph build ef enron-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./enron ./enron-t 1 0.1 true false >> result/enron-01e-true.out
	time cargo run --release --bin avgdist-rs ./enron ./enron-t 1 0.1 false false > result/enron-01e-1k.out
	time cargo run --release --bin avgdist-rs ./enron ./enron-t 10 0.1 false false > result/enron-01e-10k.out
	time cargo run --release --bin avgdist-rs ./enron ./enron-t 100 0.1 false false > result/enron-01e-100k.out
	time cargo run --release --bin avgdist-rs ./enron ./enron-t 346 0.1 false false > result/enron-01e-346k.out

