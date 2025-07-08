

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/bitcoin-webgraph/pg 112 0.1 false > result/avgdist-uni.out
time cargo run --bin harmonic --release -- /data/bitcoin/bitcoin-webgraph/pg-t 32 0.1 false > result/harmonic.out

