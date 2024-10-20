import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

def convert_txt_to_mtx(input_file, output_file):
    # Load the edge list from the .txt file
    edges = np.loadtxt(input_file, dtype=int)

    # Create a sparse adjacency matrix
    num_nodes = np.max(edges) + 1
    adjacency_matrix = csr_matrix((num_nodes, num_nodes), dtype=int)

    # Populate the sparse matrix
    rows = edges[:, 0]
    cols = edges[:, 1]
    adjacency_matrix[rows, cols] = 1
    adjacency_matrix[cols, rows] = 1  # For undirected graphs

    # Write the sparse matrix to a .mtx file
    mmwrite(output_file, adjacency_matrix)

# Usage
convert_txt_to_mtx('CA-AstroPh.txt', 'CA-AstroPh.mtx')
