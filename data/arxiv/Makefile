
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 12008 arxiv < arxiv.csv
	webgraph transform transpose arxiv arxiv-t
	webgraph build ef arxiv
	webgraph build ef arxiv-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./arxiv ./arxiv-t 1 0.1 true false >> result/arxiv-01e-true.out
	time cargo run --release --bin avgdist-rs ./arxiv ./arxiv-t 1 0.1 false false > result/arxiv-01e-1k.out
	time cargo run --release --bin avgdist-rs ./arxiv ./arxiv-t 10 0.1 false false > result/arxiv-01e-10k.out
	time cargo run --release --bin avgdist-rs ./arxiv ./arxiv-t 100 0.1 false false > result/arxiv-01e-100k.out
	time cargo run --release --bin avgdist-rs ./arxiv ./arxiv-t 346 0.1 false false > result/arxiv-01e-346k.out

