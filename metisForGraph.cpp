//to execute me ---> g++ metisForGraph.cpp -lmetis -o bb
//give numeber of sub graphs u wantas argument when compiling -->./bb 3
//to execute me ---> g++ metisForGraph.cpp -lmetis -o bb
//give numeber of sub graphs u wantas argument when compiling -->./bb 3
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h> // For mkdir
#include <errno.h>    // For errno

#ifdef _MSC_VER
#include <direct.h> // For _mkdir on Windows
#define mkdir(path, mode) _mkdir(path)
#endif

extern "C" {
#include <metis.h>
}

// Function to read an array of idx_t from a file (space or newline separated)
int read_idx_array_from_file(const char* filename, idx_t** array, idx_t* size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        perror(("Error opening file: " + std::string(filename)).c_str());
        return -1; // Indicate error
    }

    std::vector<idx_t> temp_vec;
    idx_t value;
    while (file >> value) {
        temp_vec.push_back(value);
    }

    file.close();

    *size = temp_vec.size();
    *array = (idx_t*)malloc(*size * sizeof(idx_t));
    if (*array == NULL) {
        perror("Error allocating memory for array");
        return -1; // Indicate error
    }

    for (size_t i = 0; i < *size; ++i) {
        (*array)[i] = temp_vec[i];
    }

    return 0; // Indicate success
}

// Function to create a directory
int create_directory(const std::string& path) {
#ifdef _MSC_VER
    int ret = mkdir(path.c_str());
#else
    int ret = mkdir(path.c_str(), 0775); // Use appropriate permissions
#endif
    if (ret == 0 || errno == EEXIST) {
        return 0; // Success or directory already exists
    } else {
        perror(("Error creating directory: " + path).c_str());
        return -1; // Indicate error
    }
}

// Function to save subgraph CSR data (including weights) to files in a directory
// Each value is saved on a new line for consistency with input format.
void save_subgraph_csr_with_weights(const std::string& directory_path, idx_t num_vertices, idx_t num_edges, const idx_t* xadj, const idx_t* adjncy, const idx_t* values) {
    std::string row_ptr_filename = directory_path + "/row_ptr_sub_graph.txt";
    std::string col_idx_filename = directory_path + "/col_idx_sub_graph.txt";
    std::string values_filename = directory_path + "/values_sub_graph.txt";

    // Save row_ptr (xadj)
    std::ofstream row_ptr_outfile(row_ptr_filename);
    if (!row_ptr_outfile.is_open()) {
        perror(("Error opening file for writing: " + row_ptr_filename).c_str());
        return;
    }
    for (idx_t i = 0; i <= num_vertices; ++i) {
        row_ptr_outfile << xadj[i] << "\n";
    }
    row_ptr_outfile.close();

    // Save col_idx (adjncy)
    std::ofstream col_idx_outfile(col_idx_filename);
    if (!col_idx_outfile.is_open()) {
        perror(("Error opening file for writing: " + col_idx_filename).c_str());
        return;
    }
    for (idx_t i = 0; i < num_edges; ++i) {
        col_idx_outfile << adjncy[i] << "\n";
    }
    col_idx_outfile.close();

    // Save values (adjwgt for subgraph)
    std::ofstream values_outfile(values_filename);
    if (!values_outfile.is_open()) {
        perror(("Error opening file for writing: " + values_filename).c_str());
        return;
    }
    for (idx_t i = 0; i < num_edges; ++i) {
        values_outfile << values[i] << "\n";
    }
    values_outfile.close();
}


int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number_of_partitions>\n", argv[0]);
        return 1;
    }

    // Number of partitions (N)
    idx_t nparts = atoi(argv[1]);
    if (nparts <= 0) {
        fprintf(stderr, "Number of partitions must be positive.\n");
        return 1;
    }

    // --- File paths for CSR data ---
    const char* row_ptr_file = "/home/hp/Desktop/PDC_FP/testingData/csr_row_ptr.txt";
    const char* col_index_file = "/home/hp/Desktop/PDC_FP/testingData/csr_col_idx.txt";
    const char* values_file = "/home/hp/Desktop/PDC_FP/testingData/csr_values.txt";

    idx_t* xadj = NULL;
    idx_t* adjncy = NULL;
    idx_t* adjwgt = NULL; // Use adjwgt for edge weights in METIS
    idx_t* part = NULL;     // Output: partition assignment for each vertex

    idx_t xadj_size = 0;
    idx_t adjncy_size = 0;
    idx_t adjwgt_size = 0;

    // --- Read CSR data from files ---
    // The reading function assumes space or newline separated values
    if (read_idx_array_from_file(row_ptr_file, &xadj, &xadj_size) != 0) {
        return 1; // Error reading xadj file
    }

    if (read_idx_array_from_file(col_index_file, &adjncy, &adjncy_size) != 0) {
        free(xadj); // Free previously allocated memory
        return 1; // Error reading adjncy file
    }

    if (read_idx_array_from_file(values_file, &adjwgt, &adjwgt_size) != 0) {
        free(xadj);
        free(adjncy);
        return 1; // Error reading values file
    }

    // Determine number of vertices and edges
    idx_t nvtxs = xadj_size - 1;
    if (nvtxs <= 0 && xadj_size > 0) { // Handle case of empty graph (0 vertices, 1 entry in xadj which is 0)
        nvtxs = 0;
    } else if (xadj_size == 0) {
         fprintf(stderr, "Error: csr_row_ptr.txt is empty.\n");
         free(xadj);
         free(adjncy);
         free(adjwgt);
         return 1;
    }

    idx_t num_edges = adjncy_size; // Total entries in adjncy (should match adjwgt_size)

    if (num_edges != adjwgt_size) {
         fprintf(stderr, "Error: Mismatch between size of col_index (%d) and values (%d) files.\n", (int)num_edges, (int)adjwgt_size);
         free(xadj);
         free(adjncy);
         free(adjwgt);
         return 1;
    }
     if (nvtxs > 0 && xadj[nvtxs] != num_edges) {
         fprintf(stderr, "Error: Mismatch between last element of row_ptr (%d) and total number of edges (%d).\n", (int)xadj[nvtxs], (int)num_edges);
         free(xadj);
         free(adjncy);
         free(adjwgt);
         return 1;
    }


    // Allocate memory for the partition array
    part = (idx_t*)malloc(nvtxs * sizeof(idx_t));
    if (part == NULL) {
        perror("Error allocating memory for partition array");
        free(xadj);
        free(adjncy);
        free(adjwgt);
        return 1;
    }

    idx_t ncon = 1; // Number of balancing constraints (usually 1)
    idx_t *vwgt = NULL; // Vertex weights (can be NULL)
    idx_t *vsize = NULL; // Vertex sizes (can be NULL)

    // METIS options (optional)
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    // You can set specific options here, e.g., to specify objective function

    // --- Call METIS Partitioning Function ---
    idx_t objval; // Output: objective value (e.g., edge cut or communication volume)
    int ret;

    // Only partition if there are vertices
    if (nvtxs > 0) {
        ret = METIS_PartGraphKway(
            &nvtxs, &ncon, xadj, adjncy,
            vwgt, vsize, adjwgt, // Pass adjwgt for edge weights
            &nparts, NULL, NULL, options, &objval, part
        );
    } else {
        ret = METIS_OK; // Consider 0 vertices graph as successfully partitioned
        objval = 0;     // No edges to cut
        // No need to set 'part' array as there are no vertices
        printf("Graph has 0 vertices. No partitioning needed.\n");
    }


    // --- Process the results ---
    if (ret == METIS_OK) {
        printf("Graph partitioned successfully into %d parts.\n", (int)nparts);
        printf("Partitioning Objective Value (e.g., Weighted Edge Cut): %d\n", (int)objval);

        printf("Vertex Partitions:\n");
        if (nvtxs > 0) {
            for (idx_t i = 0; i < nvtxs; ++i) {
                printf("Vertex %d -> Partition %d\n", (int)i, (int)part[i]);
            }
        }

        // --- Save Subgraphs to METISPartitioning directory ---
        const std::string main_output_dir = "METISPartitioning";
        printf("\nCreating main output directory '%s' and saving subgraphs...\n", main_output_dir.c_str());

        if (create_directory(main_output_dir) != 0) {
            // Error already reported in create_directory
            // Proceeding might cause further errors, but we'll try
        }

        // --- Save the global partition array (FIX) ---
        std::string global_partition_filename = main_output_dir + "/global_partition.txt";
        std::ofstream global_partition_outfile(global_partition_filename);
        if (!global_partition_outfile.is_open()) {
            perror(("Error opening global partition file for writing: " + global_partition_filename).c_str());
            // Handle this error appropriately, maybe return an error code
        } else {
            for (idx_t i = 0; i < nvtxs; ++i) {
                global_partition_outfile << part[i] << "\n"; // Write partition ID for each vertex on a new line
            }
            global_partition_outfile.close();
            printf("Global partition array saved to %s\n", global_partition_filename.c_str());
        }


        // Create a vector of vectors to store vertices in each partition
        std::vector<std::vector<idx_t>> partitions_vertices(nparts);
        if (nvtxs > 0) {
            for (idx_t i = 0; i < nvtxs; ++i) {
                if (part[i] >= 0 && part[i] < nparts) {
                    partitions_vertices[part[i]].push_back(i);
                } else {
                    fprintf(stderr, "Warning: Vertex %d assigned to invalid partition %d\n", (int)i, (int)part[i]);
                }
            }
        }

        for (idx_t p = 0; p < nparts; ++p) {
            std::string subgraph_dir_name = main_output_dir + "/subgraph" + std::to_string(p + 1);
            if (create_directory(subgraph_dir_name) != 0) {
                 // Error already reported in create_directory
                 continue; // Skip saving this subgraph
            }

            const std::vector<idx_t>& current_partition_vertices = partitions_vertices[p];
            idx_t subgraph_nvtxs = current_partition_vertices.size();

            // Use vectors for subgraph CSR data to handle dynamic size
            std::vector<idx_t> sub_xadj_vec(subgraph_nvtxs + 1, 0);
            std::vector<idx_t> sub_adjncy_vec;
            std::vector<idx_t> sub_values_vec;
            idx_t subgraph_num_edges = 0;

            if (subgraph_nvtxs > 0) {
                 std::vector<idx_t> original_to_subgraph_map(nvtxs, -1);
                 for (idx_t i = 0; i < subgraph_nvtxs; ++i) {
                     original_to_subgraph_map[current_partition_vertices[i]] = i;
                 }

                 // Build CSR for the subgraph
                 for (idx_t i = 0; i < subgraph_nvtxs; ++i) {
                     idx_t original_u = current_partition_vertices[i];
                     sub_xadj_vec[i] = subgraph_num_edges;

                     for (idx_t j = xadj[original_u]; j < xadj[original_u + 1]; ++j) {
                         idx_t original_v = adjncy[j];
                         idx_t original_weight = adjwgt[j];

                         // Check if the neighbor is also in the current partition
                         if (original_v < nvtxs && part[original_v] == p) { // Add bounds check for original_v
                             sub_adjncy_vec.push_back(original_to_subgraph_map[original_v]);
                             sub_values_vec.push_back(original_weight);
                             subgraph_num_edges++;
                         } else if (original_v >= nvtxs) {
                              fprintf(stderr, "Warning: Original adjncy contains out-of-bounds vertex index %d for vertex %d\n", (int)original_v, (int)original_u);
                         }
                     }
                 }
                 sub_xadj_vec[subgraph_nvtxs] = subgraph_num_edges;
            }


            // Save the subgraph to files in the directory
            // Pass data pointers from vectors
            save_subgraph_csr_with_weights(
                subgraph_dir_name,
                subgraph_nvtxs,
                subgraph_num_edges,
                sub_xadj_vec.data(),
                sub_adjncy_vec.data(),
                sub_values_vec.data()
            );
            printf("Subgraph %d saved to directory %s (Vertices: %d, Edges: %d)\n", (int)p + 1, subgraph_dir_name.c_str(), (int)subgraph_nvtxs, (int)subgraph_num_edges);
        }

    } else {
        fprintf(stderr, "METIS partitioning failed with error code: %d\n", ret);
        // Refer to METIS documentation for error codes
    }

    // --- Free allocated memory ---
    free(xadj);
    free(adjncy);
    free(adjwgt);
    free(part); // part might be NULL if nvtxs is 0, but free(NULL) is safe

    return 0;
}