

mkdir -p result


time cargo run --release -- /data/bitcoin/bitcoin-webgraph/pg /data/bitcoin/bitcoin-webgraph/pg-t 61 0.1 false true > result/pg-d.out
time cargo run --release -- /data/bitcoin/bitcoin-webgraph/pg /data/bitcoin/bitcoin-webgraph/pg-t 61 0.1 false false > result/pg.out
# don't do exact computation
