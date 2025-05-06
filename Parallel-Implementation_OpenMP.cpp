// This code will be run for each process and then further divided into more sub-sub graphs
// each thread done by the OpenMP section of the code.

//////////////////////////////////////////////////
//
//   2 Parts will be Parallelized
//
//   1. Detecting affected vertexes
//   3. Updating the affected vertexes
//
///////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <omp.h>
#include <fstream>
#include <climits>
#include <string>
#include <queue>
#include <unordered_set>
using namespace std;

// Initiliazing the infinity for the djikstra's algorithm part.
const int INF = INT_MAX;

// This function loads the CSR Graph data
vector<int> loadPointers(const string& filename) {
    vector<int> vec;
    ifstream infile(filename);
    int value;

    // Error Checks
    if (!infile) {
        cerr << "Error: Unable to open " << filename << endl;
        exit(1);
    }
    while (infile >> value) {
        vec.push_back(value);
    }
    return vec;
}

// Loading the Weights files (becuz they can be floating values too)
vector<float> loadWeights(const string& filename) {
    vector<float> vec;
    ifstream infile(filename);
    float value;

    // Error checks
    if (!infile) {
        cerr << "Error: Unable to open " << filename << endl;
        exit(1);
    }

    while (infile >> value) {
        vec.push_back(value);
    }
    return vec;
}

// Main function to load the CSR data from the files
void loadCSRGraph(const string& row_file, const string& col_file, const string& val_file, vector<int>& row_ptr, vector<int>& col_ind, vector<float>& weights) {
    
    // Setting the CSR dataset
    row_ptr = loadPointers(row_file);
    col_ind = loadPointers(col_file);
    weights = loadWeights(val_file);

    // Error checks
    if (col_ind.size() != weights.size()) {
        cerr << "Error: Mismatch in number of columns and weights!" << endl;
        exit(1);
    }

    // Debugging
    cout << "CSR Graph Loaded:" << endl;
    cout << " - Vertices: " << row_ptr.size() - 1 << endl;
    cout << " - Edges: " << col_ind.size() << endl;
}

// Both of the steps are combined to improve cache locality
vector<float> sssp_parallel(const vector<int>& row_ptr, const vector<int>& col_ind, const vector<float>& weights, int num_vertices, int source) {
    
    // Setting the distance for each vertex to infinity and source to 0
    vector<float> distance(num_vertices, INF);
    distance[source] = 0;

    // Making frontier to keep tracking of current nodes with affected boolean variable
    vector<bool> in_frontier(num_vertices, false);
    vector<int> frontier;

    // Start by pushing the source vertex
    frontier.push_back(source);
    in_frontier[source] = true;

    // Loop till no vertex is left
    while (!frontier.empty()) {
        vector<int> next_frontier;
        vector<bool> next_in_frontier(num_vertices, false);

        // 1 Parallelization: Identify the affected vertices
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < frontier.size(); i++) {
            int u = frontier[i];
            for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
                int v = col_ind[j];
                float weight = weights[j];

                // 2 Parallelization: Update the affected vertices
                if (distance[u] + weight < distance[v]) {
                    int new_dist = distance[u] + weight;

                    #pragma omp critical
                    {
                        if (new_dist < distance[v]) {
                            distance[v] = new_dist;
                            if (!next_in_frontier[v]) {
                                next_in_frontier[v] = true;
                                next_frontier.push_back(v);
                            }
                        }
                    }
                }
            }
        }

        frontier = next_frontier;
        in_frontier = next_in_frontier;
    }

    // Print results
    for (int i = 0; i < num_vertices; i++) {
        cout << "Vertex " << i << ": ";
        if (distance[i] == INF)
            cout << "Unreachable" << endl;
        else
            cout << distance[i] << endl;
    }

    return distance;
}

  
// Main Function ---------------------------------------------
int main() {

    // Setting the variables
    vector<int> row_ptr, col_ind;
    vector<float> weights;
    string row_file = "testingData/csr_row_ptr.txt";
    string col_file = "testingData/csr_col_idx.txt";
    string val_file = "testingData/csr_values.txt";
    int source = 0;

    // Calling the CSR loading function
    loadCSRGraph(row_file, col_file, val_file, row_ptr, col_ind, weights);

    // Calling the OpenMP code
    int num_vertices = row_ptr.size() - 1;
    vector<float> final_distance = sssp_parallel(row_ptr, col_ind, weights, num_vertices, source);

    // Converting the distance into proper file.txt
    ofstream outfile("output_distances.txt");

    for (int i = 0; i < final_distance.size(); ++i) {
        if (final_distance[i] == INF)
            outfile << "Vertex " << i << ": Unreachable" << endl;
        else
            outfile << "Vertex " << i << ": " << final_distance[i] << endl;
    }
    outfile.close();
    cout << "Distances written to output_distances.txt" << endl;

    return 0;
}
