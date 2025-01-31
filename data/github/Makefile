
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 22687 github < github.csv
	webgraph transform transpose github github-t
	webgraph build ef github
	webgraph build ef github-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./github ./github-t 1 0.1 true false >> result/github-01e-true.out
	time cargo run --release --bin avgdist-rs ./github ./github-t 1 0.1 false false > result/github-01e-1k.out
	time cargo run --release --bin avgdist-rs ./github ./github-t 10 0.1 false false > result/github-01e-10k.out
	time cargo run --release --bin avgdist-rs ./github ./github-t 100 0.1 false false > result/github-01e-100k.out
	time cargo run --release --bin avgdist-rs ./github ./github-t 346 0.1 false false > result/github-01e-346k.out

