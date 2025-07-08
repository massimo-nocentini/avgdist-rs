
unipairs:
	time cargo run --bin unipairs --release -- 1k-0001p 1 0.1 false > result/1k-0001p-uni.out
	time cargo run --bin unipairs --release -- 1k-0001p 100 0.1 false > result/1k-0001p-uni-slot.out
	time cargo run --bin unipairs --release -- 1k-0001p 1 0.1 true > result/1k-0001p-uni-exact.out


simpath:
	RUST_BACKTRACE=full time cargo run --bin simpath --release -- 1k-0001p 3 198 