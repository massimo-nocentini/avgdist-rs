
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 22687 brightkite < brightkite.csv
	webgraph transform transpose brightkite brightkite-t
	webgraph build ef brightkite
	webgraph build ef brightkite-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./brightkite ./brightkite-t 1 0.1 true false >> result/brightkite-01e-true.out
	time cargo run --release --bin avgdist-rs ./brightkite ./brightkite-t 1 0.1 false false > result/brightkite-01e-1k.out
	time cargo run --release --bin avgdist-rs ./brightkite ./brightkite-t 10 0.1 false false > result/brightkite-01e-10k.out
	time cargo run --release --bin avgdist-rs ./brightkite ./brightkite-t 100 0.1 false false > result/brightkite-01e-100k.out
	time cargo run --release --bin avgdist-rs ./brightkite ./brightkite-t 346 0.1 false false > result/brightkite-01e-346k.out

