
echo:
	echo "Invoke with a rule."

generate:
	python generate.py > star.csv

webgraph:
	webgraph from arcs --exact --num-nodes 2001 star < star.csv
	webgraph transform transpose star star-t
	webgraph build ef star
	webgraph build ef star-t

run:
	mkdir -p result
	time cargo run --release --bin avgdist-rs ./star ./star-t 60 0.1 false true > result/star-01e-d.out
	time cargo run --release --bin avgdist-rs ./star ./star-t 60 0.1 false false > result/star-01e.out
	time cargo run --release --bin avgdist-rs ./star ./star-t 60 0.1 true false >> result/star-01e.out

