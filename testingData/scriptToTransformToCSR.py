# SCRIPT 2: Read weighted edge list and convert to CSR format
import numpy as np
from collections import defaultdict

# Step 1: Load the 3-column weighted edge list
edge_list = []
with open("rmat_weighted_edges.txt", "r") as f:
    for line in f:
        u, v, w = map(int, line.strip().split())
        edge_list.append((u, v, w))
        edge_list.append((v, u, w))  # Add reverse for undirected graph

# Step 2: Build adjacency list with weights
adjacency = defaultdict(list)
for u, v, w in edge_list:
    adjacency[u].append((v, w))

# Step 3: Sort and convert to CSR
vertices = sorted(adjacency.keys())
row_ptr = [0]
col_idx = []
values = []

for v in vertices:
    neighbors = sorted(adjacency[v], key=lambda x: x[0])  # sort by neighbor index
    for neighbor, weight in neighbors:
        col_idx.append(neighbor)
        values.append(weight)
    row_ptr.append(len(col_idx))

# Step 4: Save CSR arrays
np.savetxt("csr_row_ptr.txt", row_ptr, fmt='%d')
np.savetxt("csr_col_idx.txt", col_idx, fmt='%d')
np.savetxt("csr_values.txt", values, fmt='%d')

print("CSR arrays saved as csr_row_ptr.txt, csr_col_idx.txt, and csr_values.txt")
