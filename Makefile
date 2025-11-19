
run:
	cargo run --bin louvain --release -- ../data/soc-karate.mtx 0.0001

build:
	cargo build --release

upstream-compile:
	cd louvain && make clean all

upstream-convert:
	cd louvain && ./convert -i sample_networks/arxiv.txt -o sample_networks/arxiv.bin
	cd louvain && ./convert -i sample_networks/example.txt -o sample_networks/example.bin
	cd louvain && ./convert -i sample_networks/karate.txt -o sample_networks/karate.bin

upstream-community:
	cd louvain && ./community sample_networks/example.bin -l -1 -v -q 0.0001 > sample_networks/example.tree
	cd louvain && ./community sample_networks/karate.bin -l -1 -v -q 0.0001 > sample_networks/karate.tree
	cd louvain && ./community sample_networks/arxiv.bin -l -1 -v -q 0.0001 > sample_networks/arxiv.tree

upstream-hierarchy:
	cd louvain && ./hierarchy sample_networks/example.tree > sample_networks/example.hier
	cd louvain && ./hierarchy sample_networks/karate.tree > sample_networks/karate.hier
	cd louvain && ./hierarchy sample_networks/arxiv.tree > sample_networks/arxiv.hier