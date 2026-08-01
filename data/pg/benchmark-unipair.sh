

mkdir -p result

#time cargo run --bin unipairs --release -- /data/bitcoin/2022/tg/tg 112 0.1 false > /data/bitcoin/2022/sampling/tg/avgdist-uni-2022-tg.out
#time cargo run --bin unipairs --release -- /data/bitcoin/2022/atg/atg 112 0.1 false > /data/bitcoin/2022/sampling/atg/avgdist-uni-2022-atg.out
time cargo run --bin harmonic --release -- /data/bitcoin/2022/tg/tg-t 8 0.1 false > /data/bitcoin/2022/sampling/tg/harmonic-2022-tg.out
#time cargo run --bin harmonic --release -- /data/bitcoin/2022/atg/atg-t 8 0.1 false > /data/bitcoin/2022/sampling/atg/harmonic-2022-atg.out

