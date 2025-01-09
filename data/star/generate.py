


edges = []

center = 0

for i in range(1, 1001):
    edges.append((center, i))
    edges.append((i, 1000 + i))

for u, v in edges:
    print (str(u) + "," + str(v))
