

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/bitcoin-webgraph/ug 222 0.1  > result/ug-uni-222.out
time cargo run --bin closeness --release -- /data/bitcoin/bitcoin-webgraph/ug-t 32 0.1 > result/ug-uni-222-closeness.out

