

mkdir -p result


# 1k

## epsilon = 0.2
time cargo run --release --bin avgdist-rs ./1k-0001p ./1k-0001p-t 60 0.2 false true > result/1k-0001p-02e-d.out
time cargo run --release --bin avgdist-rs ./1k-0001p ./1k-0001p-t 60 0.2 false false > result/1k-0001p-02e.out
time cargo run --release --bin avgdist-rs ./1k-0001p ./1k-0001p-t 60 0.2 true false >> result/1k-0001p-02e.out


## epsilon = 0.1
time cargo run --release --bin avgdist-rs ./1k-0001p ./1k-0001p-t 60 0.1 false true > result/1k-0001p-01e-d.out
time cargo run --release --bin avgdist-rs ./1k-0001p ./1k-0001p-t 60 0.1 false false > result/1k-0001p-01e.out
time cargo run --release --bin avgdist-rs ./1k-0001p ./1k-0001p-t 60 0.1 true false >> result/1k-0001p-01e.out

# 2k

## epsilon = 0.2
time cargo run --release --bin avgdist-rs ./2k-0001p ./2k-0001p-t 60 0.2 false true > result/2k-0001p-02e-d.out
time cargo run --release --bin avgdist-rs ./2k-0001p ./2k-0001p-t 60 0.2 false false > result/2k-0001p-02e.out
time cargo run --release --bin avgdist-rs ./2k-0001p ./2k-0001p-t 60 0.2 true false >> result/2k-0001p-02e.out


## epsilon = 0.1
time cargo run --release --bin avgdist-rs ./2k-0001p ./2k-0001p-t 60 0.1 false true > result/2k-0001p-01e-d.out
time cargo run --release --bin avgdist-rs ./2k-0001p ./2k-0001p-t 60 0.1 false false > result/2k-0001p-01e.out
time cargo run --release --bin avgdist-rs ./2k-0001p ./2k-0001p-t 60 0.1 true false >> result/2k-0001p-01e.out


# 5k

## epsilon = 0.2
time cargo run --release --bin avgdist-rs ./5k-0001p ./5k-0001p-t 60 0.2 false true > result/5k-0001p-02e-d.out
time cargo run --release --bin avgdist-rs ./5k-0001p ./5k-0001p-t 60 0.2 false false > result/5k-0001p-02e.out
time cargo run --release --bin avgdist-rs ./5k-0001p ./5k-0001p-t 60 0.2 true false >> result/5k-0001p-02e.out

## epsilon = 0.1
time cargo run --release --bin avgdist-rs ./5k-0001p ./5k-0001p-t 60 0.1 false true > result/5k-0001p-01e-d.out
time cargo run --release --bin avgdist-rs ./5k-0001p ./5k-0001p-t 60 0.1 false false > result/5k-0001p-01e.out
time cargo run --release --bin avgdist-rs ./5k-0001p ./5k-0001p-t 60 0.1 true false >> result/5k-0001p-01e.out

