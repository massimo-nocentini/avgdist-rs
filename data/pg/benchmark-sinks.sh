

mkdir -p result


time cargo run --bin sinks --release -- /data/bitcoin/bitcoin-webgraph/pg 222 0.1  > result/sinks.out