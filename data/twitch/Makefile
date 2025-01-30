
echo:
	echo "Invoke with a rule."

webgraph:
	webgraph from arcs --num-nodes 9498 twitch < twitch.csv
	webgraph transform transpose twitch twitch-t
	webgraph build ef twitch
	webgraph build ef twitch-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./twitch ./twitch-t 1 0.1 true false >> result/twitch-01e-true.out
	time cargo run --release --bin avgdist-rs ./twitch ./twitch-t 1 0.1 false false > result/twitch-01e-1k.out
	time cargo run --release --bin avgdist-rs ./twitch ./twitch-t 10 0.1 false false > result/twitch-01e-10k.out
	time cargo run --release --bin avgdist-rs ./twitch ./twitch-t 100 0.1 false false > result/twitch-01e-100k.out
	time cargo run --release --bin avgdist-rs ./twitch ./twitch-t 346 0.1 false false > result/twitch-01e-346k.out

