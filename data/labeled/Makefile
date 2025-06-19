
generate:
	echo "a\tb\nb\tc\nc\ta" > graph.tsv
	# convert to bvgraph
	cat graph.tsv | webgraph from arcs graph
	webgraph transform transpose graph graph-t
	webgraph build ef graph
	webgraph build ef graph-t
