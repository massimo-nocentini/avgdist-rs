
import sys
from networkx import dual_barabasi_albert_graph

# getting values
nodes = int(sys.argv[1])
m = int(sys.argv[2])

G = dual_barabasi_albert_graph (nodes, m, 3, 0.6)

for u, v in G.edges():
    print (str(v) + "," + str(u))
