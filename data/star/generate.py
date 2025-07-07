


edges = []

center = 0
n = 10

for i in range(1, n + 1):
    edges.append((center, i))
    edges.append((i, n + i))

for u, v in edges:
    print (str(u) + "," + str(v))
