

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/ug/ug 112 0.1 false > result/avgdist-uni.out
time cargo run --bin harmonic --release -- /data/bitcoin/ug/ug-t 32 0.1 false > result/harmonic.out

