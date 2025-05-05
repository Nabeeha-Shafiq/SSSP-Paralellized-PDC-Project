# SCRIPT 1: Generate a weighted undirected graph and save to 3-column edge list file
import numpy as np
import networkx as nx
import random

# Parameters
num_nodes = 1000
num_edges = 10000

# Generate undirected synthetic graph using G(n, m)
G = nx.gnm_random_graph(n=num_nodes, m=num_edges, directed=False)

# Assign random weights (1 to 10) to each edge
for u, v in G.edges():
    G[u][v]['weight'] = random.randint(1, 10)

# Save as 3-column edge list: u v weight
with open("rmat_weighted_edges.txt", "w") as f:
    for u, v, data in G.edges(data=True):
        f.write(f"{u} {v} {data['weight']}\n")

print("Weighted edge list saved to rmat_weighted_edges.txt")
