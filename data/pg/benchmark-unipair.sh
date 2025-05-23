

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/bitcoin-webgraph/pg 222 0.1  > result/pg-uni-222.out
time cargo run --bin closeness --release -- /data/bitcoin/bitcoin-webgraph/pg-t 62 0.1 true > result/pg-uni-222-closeness.out

