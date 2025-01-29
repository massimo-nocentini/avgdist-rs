
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --exact --num-nodes 875713 web < web.csv
	webgraph transform transpose web web-t
	webgraph build ef web
	webgraph build ef web-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./web ./web-t 1 0.1 false true > result/web-01e-d.out
	time cargo run --release --bin avgdist-rs ./web ./web-t 1 0.1 false false > result/web-01e.out
	time cargo run --release --bin avgdist-rs ./web ./web-t 1 0.1 true false >> result/web-01e.out

