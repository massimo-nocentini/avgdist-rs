

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/address-transaction-graph/atg 222 0.1 > result/atg-uni-222.out
time cargo run --bin closeness --release -- /data/bitcoin/address-transaction-graph/atg-t 32 0.1 > result/atg-uni-222-closeness.out
