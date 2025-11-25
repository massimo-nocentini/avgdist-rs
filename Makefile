
echo:
	echo "Makefile for Louvain implementation in Rust"

webgraph-all:
	cargo run --bin louvain --release -- louvain/sample_networks/example/example 0.00001 true louvain/sample_networks/example/example.webgraph.dot > louvain/sample_networks/example/example.webgraph.tree
	cargo run --bin louvain --release -- louvain/sample_networks/karate/karate 0.00001 true louvain/sample_networks/karate/karate.webgraph.dot > louvain/sample_networks/karate/karate.webgraph.tree
	cargo run --bin louvain --release -- louvain/sample_networks/arxiv/arxiv 0.00001 true louvain/sample_networks/arxiv/arxiv.webgraph.dot > louvain/sample_networks/arxiv/arxiv.webgraph.tree
	cd louvain/sample_networks/example && ../../hierarchy example.webgraph.tree > example.webgraph.hier
	cd louvain/sample_networks/karate && ../../hierarchy karate.webgraph.tree > karate.webgraph.hier
	cd louvain/sample_networks/arxiv && ../../hierarchy arxiv.webgraph.tree > arxiv.webgraph.hier

build:
	cargo build --release

upstream-compile:
	cd louvain && make clean all

upstream-convert:
	cd louvain/sample_networks/example && ../../convert -i example.txt -o example.bin
	cd louvain/sample_networks/karate && ../../convert -i karate.txt -o karate.bin
	cd louvain/sample_networks/arxiv && ../../convert -i arxiv.txt -o arxiv.bin
	
upstream-community:
	cd louvain/sample_networks/example && ../../community example.bin -l -1 -v -q 0.00001 > example.tree
	cd louvain/sample_networks/karate && ../../community karate.bin -l -1 -v -q 0.00001 > karate.tree
	cd louvain/sample_networks/arxiv && ../../community arxiv.bin -l -1 -v -q 0.00001 > arxiv.tree

upstream-hierarchy:
	cd louvain/sample_networks/example && ../../hierarchy example.tree > example.hier
	cd louvain/sample_networks/karate && ../../hierarchy karate.tree > karate.hier
	cd louvain/sample_networks/arxiv && ../../hierarchy arxiv.tree > arxiv.hier

upstream-webgraph:
	cd louvain/sample_networks/example && webgraph from arcs --labels --separator " " example < example.txt && webgraph build ef example
	cd louvain/sample_networks/karate && webgraph from arcs --labels --separator " " karate < karate.txt && webgraph build ef karate
	cd louvain/sample_networks/arxiv && webgraph from arcs --labels --separator " " arxiv < arxiv.txt && webgraph build ef arxiv

upstream-all: upstream-compile upstream-convert upstream-community upstream-hierarchy upstream-webgraph