import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance

# Load Zachary's network and visualize it
z=nx.read_gml("karate.gml")

pos=nx.spring_layout(z)
plt.figure(1)
plt.axis('off')
nx.draw_networkx(z,pos)
plt.draw()

# Compute the average shortest path length
path_length=nx.all_pairs_shortest_path_length(z)
n = len(z.nodes())
distances=np.zeros((n,n))

for u,p in path_length.iteritems():
    for v,d in p.iteritems():
        distances[int(u)-1][int(v)-1] = d

# Apply hierarchical  clustering using linkage criteria
hier = hierarchy.average(distances)

# Show dendrogram
plt.figure(2)
hierarchy.dendrogram(hier)
plt.xlabel("Node id's")
plt.draw()
plt.show()

