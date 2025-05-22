

mkdir -p result


time cargo run --bin unipairs --release -- /data/bitcoin/address-transaction-graph/atg 32 0.1 false > result/atg-uni-222.out
time cargo run --bin unipairs --release -- /data/bitcoin/address-transaction-graph/atg-t 32 0.1 true > result/atg-uni-222-closeness.out
