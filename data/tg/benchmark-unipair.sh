

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/bitcoin-webgraph/tg 222 0.1  > result/tg-uni-222.out
time cargo run --bin closeness --release -- /data/bitcoin/bitcoin-webgraph/tg-t 32 0.1 > result/tg-uni-222-closeness.out

