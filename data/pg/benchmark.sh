

mkdir -p result


time cargo run --bin avgdist-rs --release -- /data/bitcoin/bitcoin-webgraph/pg /data/bitcoin/bitcoin-webgraph/pg-t 222 0.1 false false > result/pg-222.out
time cargo run --bin avgdist-rs --release -- /data/bitcoin/bitcoin-webgraph/pg /data/bitcoin/bitcoin-webgraph/pg-t 222 0.1 false true > result/pg-d-222.out
# don't do exact computation
