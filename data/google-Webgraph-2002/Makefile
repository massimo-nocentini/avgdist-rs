
# From: https://snap.stanford.edu/data/web-Google.html

echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 875713 web < web.csv
	webgraph transform transpose web web-t
	webgraph build ef web
	webgraph build ef web-t

run:
	mkdir -p result
	#time cargo run --release --bin avgdist-rs web web-t 222 0.1 false true > result/web-01e-d.out
	time cargo run --release --bin avgdist-rs ./web ./web-t 1 0.1 false false > result/web-01e-1k.out
	time cargo run --release --bin avgdist-rs ./web ./web-t 10 0.1 false false > result/web-01e-10k.out
	time cargo run --release --bin avgdist-rs ./web ./web-t 100 0.1 false false > result/web-01e-100k.out
	time cargo run --release --bin avgdist-rs ./web ./web-t 346 0.1 false false > result/web-01e-346k.out

