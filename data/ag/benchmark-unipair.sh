

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/ag/ag 222 0.1 > result/ag-uni-222.out
time cargo run --bin closeness --release -- /data/bitcoin/ag/ag-t 32 0.1 > result/ag-uni-222-closeness.out
