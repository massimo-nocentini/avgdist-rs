

mkdir -p result

time cargo run --bin unipairs --release -- /data/bitcoin/tg/tg 0 0.1 false > result/avgdist-uni.out
time cargo run --bin harmonic --release -- /data/bitcoin/tg/tg-t 0 0.1 false > result/harmonic.out

