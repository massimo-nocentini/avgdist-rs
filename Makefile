
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