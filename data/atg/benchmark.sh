

mkdir -p result


time cargo run --bin avgdist-rs --release -- /data/bitcoin/address-transaction-graph/atg /data/bitcoin/address-transaction-graph/atg-t 61 0.1 false false > result/atg.out
time cargo run --bin avgdist-rs --release -- /data/bitcoin/address-transaction-graph/atg /data/bitcoin/address-transaction-graph/atg-t 61 0.1 false true > result/atg-d.out

