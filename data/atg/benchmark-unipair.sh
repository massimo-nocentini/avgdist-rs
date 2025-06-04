

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/atg/atg 112 0.1 false > result/avgdist-uni.out
time cargo run --bin harmonic --release -- /data/bitcoin/atg/atg-t 62 0.1 false > result/harmonic.out
