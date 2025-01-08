
echo:
	echo "Invoke with a rule."

generate:
	python generate.py 1000 10 > 1k-10m.csv
	python generate.py 2000 10 > 2k-10m.csv
	python generate.py 5000 10 > 5k-10m.csv

webgraph:
	webgraph from arcs --exact --num-nodes 1000 1k-10m < 1k-10m.csv
	webgraph from arcs --exact --source-column 1 --target-column 0 --num-nodes 1000 1k-10m-t < 1k-10m.csv
	webgraph build ef 1k-10m
	webgraph build ef 1k-10m-t

	webgraph from arcs --exact --num-nodes 1000 2k-10m < 2k-10m.csv
	webgraph from arcs --exact --source-column 1 --target-column 0 --num-nodes 1000 2k-10m-t < 2k-10m.csv
	webgraph build ef 2k-10m
	webgraph build ef 2k-10m-t

	webgraph from arcs --exact --num-nodes 1000 5k-10m < 5k-10m.csv
	webgraph from arcs --exact --source-column 1 --target-column 0 --num-nodes 1000 5k-10m-t < 5k-10m.csv
	webgraph build ef 5k-10m
	webgraph build ef 5k-10m-t

run:
	mkdir -p result

	time cargo run --release --bin avgdist-rs ./1k-10m ./1k-10m-t 60 0.1 false true > result/1k-10m-01e-d.out
	time cargo run --release --bin avgdist-rs ./1k-10m ./1k-10m-t 60 0.1 false false > result/1k-10m-01e.out
	time cargo run --release --bin avgdist-rs ./1k-10m ./1k-10m-t 60 0.1 true false >> result/1k-10m-01e.out

	time cargo run --release --bin avgdist-rs ./2k-10m ./2k-10m-t 60 0.1 false true > result/2k-10m-01e-d.out
	time cargo run --release --bin avgdist-rs ./2k-10m ./2k-10m-t 60 0.1 false false > result/2k-10m-01e.out
	time cargo run --release --bin avgdist-rs ./2k-10m ./2k-10m-t 60 0.1 true false >> result/2k-10m-01e.out

	time cargo run --release --bin avgdist-rs ./5k-10m ./5k-10m-t 60 0.1 false true > result/5k-10m-01e-d.out
	time cargo run --release --bin avgdist-rs ./5k-10m ./5k-10m-t 60 0.1 false false > result/5k-10m-01e.out
	time cargo run --release --bin avgdist-rs ./5k-10m ./5k-10m-t 60 0.1 true false >> result/5k-10m-01e.out