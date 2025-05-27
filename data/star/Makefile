
echo:
	echo "Invoke with a rule."

generate:
	python generate.py > star.csv

webgraph:
	webgraph from arcs --exact --num-nodes 2001 star < star.csv
	webgraph transform transpose star star-t
	webgraph build ef star
	webgraph build ef star-t

run:
	mkdir -p result
	#time cargo run --release --bin avgdist-rs ./star ./star-t 1 0.1 false true > result/star-01e-d.out
	#time cargo run --release --bin avgdist-rs ./star ./star-t 1 0.1 false false > result/star-01e.out
	time cargo run --release --bin avgdist-rs ./star ./star-t 1 0.1 true false > result/star-01e-true.out
	time cargo run --release --bin avgdist-rs ./star ./star-t 1 0.1 false false > result/star-01e-1k.out
	time cargo run --release --bin avgdist-rs ./star ./star-t 10 0.1 false false > result/star-01e-10k.out
	time cargo run --release --bin avgdist-rs ./star ./star-t 100 0.1 false false > result/star-01e-100k.out
	time cargo run --release --bin avgdist-rs ./star ./star-t 346 0.1 false false > result/star-01e-346k.out

unipairs:
	time cargo run --bin unipairs --release -- ./star 222 0.1 > result/star-uni.out

closeness:
	#time cargo run --bin closeness --release -- ./star-t 32 0.05 false > result/star-closeness.out
	#time cargo run --bin closeness --release -- ./star-t 32 0.05 true > result/star-closeness-true.out

harmonic:
	time cargo run --bin harmonic --release -- ./star-t 32 0.05 false > result/star-harmonic.out
	time cargo run --bin harmonic --release -- ./star-t 32 0.1 true > result/star-harmonic-true.out
