

mkdir -p result

time cargo run --bin sink --release -- /data/bitcoin/bitcoin-webgraph/pg > result/sinks.out
time cargo run --bin sink --release -- /data/bitcoin/bitcoin-webgraph/pg-t >> result/sinks.out
