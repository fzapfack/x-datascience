#!/usr/bin/env python


"""
Graph Mining and Analysis with Python - MVA-MATHBIGDATA - March 2015
"""

#%%
# Import modules
from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

G=nx.read_edgelist("ca-GrQc.txt", comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())


############## Question 2
# Network Characteristics
print 'Number of nodes:', G.number_of_nodes() 
print 'Number of edges:', G.number_of_edges() 
print 'Number of connected components:', nx.number_connected_components(G)

# Connected components
GCC=list(nx.connected_component_subgraphs(G))[0]

# Fraction of nodes and edges in GCC 
print "Fraction of nodes in GCC: ", GCC.number_of_nodes() / G.number_of_nodes()
print "Fraction of edges in GCC: ", GCC.number_of_edges() / G.number_of_edges()

#%%
############## Question 3
# Degree
degree_sequence = G.degree().values()
print "Min degree ", np.min(degree_sequence)
print "Max degree ", np.max(degree_sequence)
print "Median degree ", np.median(degree_sequence)
print "Mean degree ", np.mean(degree_sequence)

# Degree distribution
y=nx.degree_histogram(G)
plt.figure(1)
plt.plot(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
#f.savefig("degree.png",format="png")

plt.figure(2)
plt.loglog(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
#s.savefig("degree_loglog.png",format="png")


#%%
############## Question 4
# (I) Triangles
# Number of triangles
t=nx.triangles(G)
t_total = sum(t.values())/3
print "Total number of triangles ", t_total

# Distribution of triangle participation (similar to the degree distribution, i.e., in how many triangles each node participates to)
t_values = sorted(set(t.values()))
t_hist = [t.values().count(x) for x in t_values]
plt.figure(3)
plt.loglog(t_values, t_hist, 'bo')
plt.xlabel('Number of triangles')
plt.ylabel('Count')
plt.draw()
plt.show()
#plt.savefig("triangles.png",format="png")

# Average clustering coefficient
avg_clust_coef = nx.average_clustering(G)
print "Average clustering coefficient ", avg_clust_coef


#%%
############## Question 5
# Centrality measures

# Degree centrality
deg_centrality = nx.degree_centrality(G)

# Eigenvector centrality
eig_centrality = nx.eigenvector_centrality(G)

# Sort centrality values
sorted_deg_centrality = sorted(deg_centrality.items())
sorted_eig_centrality = sorted(eig_centrality.items())

# Extract centralities
deg_data=[b for a,b in sorted_deg_centrality]
eig_data=[b for a,b in sorted_eig_centrality]

# Compute Pearson correlation coefficient
from scipy.stats.stats import pearsonr
print "Pearson correlation coefficient ", pearsonr(deg_data, eig_data)

# Plot correlation between degree and eigenvector centrality
plt.figure(4)
plt.plot(deg_data, eig_data, 'ro')
plt.xlabel('Degree centrality')
plt.ylabel('Eigenvector centrality')
plt.draw()
plt.show()
#plt.savefig("deg_eig_correlation.png",format="png")



#%%
############## Question 6
# Generate a random graph with 200 nodes
R = nx.fast_gnp_random_graph(30, 0.9)

degree_sequence_random = R.degree().values()
print "Min degree ", np.min(degree_sequence_random)
print "Max degree ", np.max(degree_sequence_random)
print "Median degree ", np.median(degree_sequence_random)
print "Mean degree ", np.mean(degree_sequence_random)

r=nx.degree_histogram(R)

plt.figure(5)
plt.plot(r,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
#plt.savefig("random_degree.png",format="png")

nx.draw_networkx(R,nx.spring_layout(R))


#%%
# (I) Triangles - Random graph
# Number of triangles in the random graph
t_R=nx.triangles(R)
t_total_R = sum(t_R.values())/3
print "Total number of triangles ", t_total_R

# Distribution of triangle participation (similar to the degree distribution, i.e., in how many triangles each node participates to)
t_R_values = sorted(set(t_R.values()))
t_R_hist = [t_R.values().count(x) for x in t_R_values]
plt.figure(6)
plt.loglog(t_R_values, t_R_hist, 'bo')
plt.xlabel('Number of nodes')
plt.ylabel('Count')
plt.draw()
plt.show()
#plt.savefig("triangles_random.png",format="png")

# Average clustering coefficient - Random graph
avg_clust_coef_R = nx.average_clustering(R)
print "Average clustering coefficient ", avg_clust_coef_R




#%%
############## Question 7
# Kronecker graphs

# Initiator matrix
A_1 = np.array([[0.9, 0.5], [0.5, 0.53]])
A_G = A_1

# Get Kronecker matrix after k=10 iterations
for i in range(10):
    A_G = np.kron(A_G, A_1)

# Create adjacency matrix
for i in range(A_G.shape[0]):
    for j in range(A_G.shape[1]):
        if random.random() <= A_G[i][j]:
            A_G[i][j] = 1
        else:
            A_G[i][j] = 0

print A_G.shape
print A_G
   
# Convert adjacency matrix to graph
G_kron = nx.from_numpy_matrix(A_G, create_using=nx.Graph())

degree_sequence_random = G_kron.degree().values()
print "Min degree ", np.min(degree_sequence_random)
print "Max degree ", np.max(degree_sequence_random)
print "Median degree ", np.median(degree_sequence_random)
print "Mean degree ", np.mean(degree_sequence_random)

# Plot degree distribution
y=nx.degree_histogram(G_kron)
plt.figure(7)
plt.title("Kronecker graph")
plt.loglog(y,'b-',marker='o')
plt.ylabel("Frequency")
plt.xlabel("Degree")
plt.draw()
plt.show()
