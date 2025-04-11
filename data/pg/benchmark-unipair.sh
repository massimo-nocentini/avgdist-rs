

mkdir -p result


time cargo run --bin unipairs --release -- /data/bitcoin/bitcoin-webgraph/pg 222 0.1  > result/pg-uni-222.out
#time cargo run --bin unipairs --release -- /data/bitcoin/bitcoin-webgraph/pg 222 0.1 false true > result/pg-d-222.out
# don't do exact computation
