
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
