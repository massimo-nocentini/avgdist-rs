
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 475 twitter < twitter.csv
	webgraph transform transpose twitter twitter-t
	webgraph build ef twitter
	webgraph build ef twitter-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./twitter ./twitter-t 1 0.1 true false >> result/twitter-01e-true.out
	time cargo run --release --bin avgdist-rs ./twitter ./twitter-t 1 0.1 false false > result/twitter-01e-1k.out
	time cargo run --release --bin avgdist-rs ./twitter ./twitter-t 10 0.1 false false > result/twitter-01e-10k.out
	time cargo run --release --bin avgdist-rs ./twitter ./twitter-t 100 0.1 false false > result/twitter-01e-100k.out
	time cargo run --release --bin avgdist-rs ./twitter ./twitter-t 346 0.1 false false > result/twitter-01e-346k.out

