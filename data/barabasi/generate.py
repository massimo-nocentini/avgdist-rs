
import sys
from networkx import barabasi_albert_graph

# getting values
nodes = int(sys.argv[1])
m = int(sys.argv[2])

G = barabasi_albert_graph (nodes, m)

for u, v in G.edges():
    print (str(u) + "," + str(v))
