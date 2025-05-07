// to execute ---> mpicxx allInOne.cpp -lmetis -fopenmp -o run
// to run --> mpirun -np 3 ./run 3 0 8   
// (the first -np 3 tells code to make 3 processes)
// the ./run 3  tells code to make 3 METIS partitions
// the ending 8 is for number of threads
// keep in mind the number od partittions must be equal to number of processes , that is why 3 is repeated  

#include <iostream>
#include <vector>
#include <omp.h>
#include <fstream>
#include <climits> // For INT_MAX
#include <string>
#include <queue>
#include <unordered_set>
#include <mpi.h>
#include <sys/stat.h> // For mkdir
#include <errno.h>    // For errno
#include <numeric> // For iota
#include <tuple> // For tuple
#include <algorithm> // For std::sort
#include <cstring> // For strerror

extern "C" {
#include <metis.h>
}

#ifdef _MSC_VER
#include <direct.h> // For _mkdir on Windows
#define mkdir(path, mode) _mkdir(path)
#endif

using namespace std;

// Initiliazing the infinity for the djikstra's algorithm part.
// Keep as float to avoid overflow issues with accumulated integer weights
const float INF = 1e9;

// Function to read an array of integers from a file (space or newline separated)
vector<int> loadIntArray(const string& filename, int world_rank) {
    vector<int> vec;
    ifstream infile(filename);
    int value;

    if (!infile.is_open()) {
        cerr << "Error (Rank " << world_rank << "): Unable to open " << filename << " - " << strerror(errno) << endl;
        // For this combined script, loading local subgraph files happens on all ranks.
        // Loading global partition file happens on all ranks.
        // Loading global graph files happens only on root for METIS.
        // Abort if any process fails to open a file they need.
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
    while (infile >> value) {
        vec.push_back(value);
    }
    return vec;
}

// Function to read an array of floats from a file (space or newline separated)
// We keep this function but will use loadIntArray for weights as requested
vector<float> loadFloatArray(const string& filename, int world_rank) {
    vector<float> vec;
    ifstream infile(filename);
    float value;

    if (!infile.is_open()) {
         cerr << "Error (Rank " << world_rank << "): Unable to open " << filename << " - " << strerror(errno) << endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
    }

    while (infile >> value) {
        vec.push_back(value);
    }
    return vec;
}

// Function to save subgraph CSR data (including weights) to files in a directory
// Each value is saved on a new line for consistency with input format.
void save_subgraph_csr_with_weights(const std::string& directory_path, idx_t num_vertices, idx_t num_edges, const idx_t* xadj, const idx_t* adjncy, const idx_t* values, int world_rank) {
    std::string row_ptr_filename = directory_path + "/row_ptr_sub_graph.txt";
    std::string col_idx_filename = directory_path + "/col_idx_sub_graph.txt";
    std::string values_filename = directory_path + "/values_sub_graph.txt";

    // Save row_ptr (xadj)
    std::ofstream row_ptr_outfile(row_ptr_filename);
    if (!row_ptr_outfile.is_open()) {
        cerr << "Error (Rank " << world_rank << "): Error opening file for writing: " << row_ptr_filename << " - " << strerror(errno) << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (idx_t i = 0; i <= num_vertices; ++i) {
        row_ptr_outfile << xadj[i] << "\n";
    }
    row_ptr_outfile.close();

    // Save col_idx (adjncy)
    std::ofstream col_idx_outfile(col_idx_filename);
    if (!col_idx_outfile.is_open()) {
         cerr << "Error (Rank " << world_rank << "): Error opening file for writing: " << col_idx_filename << " - " << strerror(errno) << endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (idx_t i = 0; i < num_edges; ++i) {
        col_idx_outfile << adjncy[i] << "\n";
    }
    col_idx_outfile.close();

    // Save values (adjwgt for subgraph)
    std::ofstream values_outfile(values_filename);
    if (!values_outfile.is_open()) {
        cerr << "Error (Rank " << world_rank << "): Error opening file for writing: " << values_filename << " - " << strerror(errno) << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (idx_t i = 0; i < num_edges; ++i) {
        values_outfile << values[i] << "\n";
    }
    values_outfile.close();
}


// Function to load the CSR Graph data (local subgraph) from file
void loadLocalCSRGraph(const string& row_file, const string& col_file, const string& val_file,
                       vector<int>& row_ptr, vector<int>& col_ind, vector<int>& weights, int world_rank) { // Weights as int

    row_ptr = loadIntArray(row_file, world_rank);
    col_ind = loadIntArray(col_file, world_rank);
    weights = loadIntArray(val_file, world_rank); // Load weights as int

    if (col_ind.size() != weights.size()) {
        cerr << "Error (Rank " << world_rank << "): Mismatch in number of columns (" << col_ind.size() << ") and weights (" << weights.size() << ") for local subgraph!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
     if (!row_ptr.empty()) {
         int num_local_vertices = row_ptr.size() - 1;
         if (num_local_vertices >= 0 && row_ptr[num_local_vertices] != col_ind.size()) { // Check last element vs total edges
              cerr << "Error (Rank " << world_rank << "): Mismatch between last element of local row_ptr (" << row_ptr[num_local_vertices] << ") and total number of local edges (" << col_ind.size() << ")." << endl;
              MPI_Abort(MPI_COMM_WORLD, 1);
         } else if (num_local_vertices < 0) {
              cerr << "Error (Rank " << world_rank << "): Invalid number of local vertices calculated from row_ptr size." << endl;
              MPI_Abort(MPI_COMM_WORLD, 1);
         }
     }

}


// Data structure to store information about a ghost node connection
struct GhostConnection {
    int global_ghost_id; // Global ID of the ghost node
    int owner_rank;      // MPI rank of the process that owns the ghost node
    int weight;        // Weight of the edge connecting the local boundary vertex to the ghost node (as int)
};

// Data structure to store boundary and ghost node information for the local partition
struct BoundaryInfo {
    vector<int> local_boundary_vertices; // Local IDs of boundary vertices in this partition
    // Map: local boundary vertex ID -> list of ghost node connections
    vector<vector<GhostConnection>> boundary_to_ghost;
};


// Function to identify boundary vertices and ghost nodes
BoundaryInfo identify_boundary_and_ghost_nodes(
    int world_rank,
    int num_global_vertices,
    const vector<int>& global_row_ptr,
    const vector<int>& global_col_ind,
    const vector<int>& global_adjwgt, // Global weights as int
    const vector<int>& global_part,
    const vector<int>& local_to_global // Mapping from local ID to global ID for this partition
) {
    BoundaryInfo boundary_info;
    // A set to keep track of local boundary vertices to avoid duplicates
    unordered_set<int> local_boundary_set;

    // Resize boundary_to_ghost to match the number of local vertices, even though
    // only boundary vertices will have non-empty lists. This simplifies indexing.
     boundary_info.boundary_to_ghost.resize(local_to_global.size());


    // Iterate through each local vertex in this partition
    for (int local_u = 0; local_u < local_to_global.size(); ++local_u) {
        int global_u = local_to_global[local_u];

        // Iterate through the neighbors of this vertex in the GLOBAL graph
        if (global_u >= 0 && global_u < num_global_vertices) { // Bounds check for global_u
            for (int i = global_row_ptr[global_u]; i < global_row_ptr[global_u + 1]; ++i) {
                int global_v = global_col_ind[i];
                int edge_weight = global_adjwgt[i]; // Read weight as int

                // Check if the neighbor is in a different partition
                if (global_v >= 0 && global_v < num_global_vertices && global_part[global_v] != world_rank) {
                    // global_u is a boundary vertex
                    local_boundary_set.insert(local_u);

                    // global_v is a ghost node
                    GhostConnection gc;
                    gc.global_ghost_id = global_v;
                    gc.owner_rank = global_part[global_v];
                    gc.weight = edge_weight; // Store weight as int

                    // Add this ghost connection to the list for the local boundary vertex
                    boundary_info.boundary_to_ghost[local_u].push_back(gc);
                } else if (global_v < 0 || global_v >= num_global_vertices) {
                     cerr << "Warning (Rank " << world_rank << "): Global adjncy contains out-of-bounds vertex index " << global_v << " for global vertex " << global_u << endl;
                }
            }
        } else {
             cerr << "Warning (Rank " << world_rank << "): local_to_global map contains out-of-bounds global vertex ID: " << global_u << " for local vertex " << local_u << endl;
        }
    }

    // Populate the list of local boundary vertices
    for (int local_u : local_boundary_set) {
        boundary_info.local_boundary_vertices.push_back(local_u);
    }

    return boundary_info;
}


// SSSP function with MPI communication (Full Implementation)
vector<float> sssp_distributed(
    int world_rank,
    int world_size,
    const vector<int>& local_row_ptr,
    const vector<int>& local_col_ind,
    const vector<int>& local_weights, // Weights as int
    int num_local_vertices,
    int source_global_id,
    const vector<int>& local_to_global, // Mapping local ID to global ID
    const vector<int>& global_to_local, // Mapping global ID to local ID (if vertex is local, -1 otherwise)
    const BoundaryInfo& boundary_info,
    MPI_Comm comm) {

    vector<float> distance(num_local_vertices, INF); // Distances as float

    // Determine if the source vertex is local to this partition
    int local_source = -1;
    if (source_global_id >= 0 && source_global_id < global_to_local.size() && global_to_local[source_global_id] != -1) {
         local_source = global_to_local[source_global_id];
    }

    if (local_source != -1) {
        distance[local_source] = 0;
    }

    vector<int> frontier;
    vector<bool> in_frontier(num_local_vertices, false); // Track local vertices in frontier

    if (local_source != -1) {
        frontier.push_back(local_source);
        in_frontier[local_source] = true;
    }

    bool global_converged = false;

    // Define MPI datatype for pair<int, float> once
    MPI_Datatype MPI_PAIR_INT_FLOAT;
    int lengths[2] = {1, 1};
    MPI_Aint displacements[2];
    MPI_Datatype types[2] = {MPI_INT, MPI_FLOAT};

    MPI_Aint base_address;
    pair<int, float> dummy_pair;
    MPI_Get_address(&dummy_pair, &base_address);
    MPI_Get_address(&dummy_pair.first, &displacements[0]);
    MPI_Get_address(&dummy_pair.second, &displacements[1]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);

    MPI_Type_create_struct(2, lengths, displacements, types, &MPI_PAIR_INT_FLOAT);
    MPI_Type_commit(&MPI_PAIR_INT_FLOAT);


    while (!global_converged) {
        vector<int> next_frontier;
        // Use a temporary boolean array to track vertices added to next_frontier in parallel
        // Initialized to false automatically by vector constructor
        vector<bool> newly_added_to_next_frontier(num_local_vertices, false);

        // Phase 1: Local Relaxation (OpenMP)
        // Relax edges for vertices in the current frontier
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < frontier.size(); ++i) {
            int u_local = frontier[i]; // Local ID of vertex u

            // Relax outgoing edges to local neighbors
            if (u_local >= 0 && u_local < local_row_ptr.size() -1) { // Bounds check for local_u
                 for (int j = local_row_ptr[u_local]; j < local_row_ptr[u_local + 1]; ++j) {
                     if (j >= 0 && j < local_col_ind.size()) { // Bounds check for edge index j
                         int v_local = local_col_ind[j]; // Local ID of neighbor v (within this subgraph)
                         int weight = local_weights[j]; // Weight as int

                         // Calculate potential new distance (cast weight to float for addition)
                         float potential_new_dist = distance[u_local] != INF ? distance[u_local] + static_cast<float>(weight) : INF;

                         if (v_local >= 0 && v_local < num_local_vertices) { // Bounds check for v_local
                             if (potential_new_dist < distance[v_local]) {
                                 // Use atomic or critical section for updating distance and the shared next_frontier
                                 #pragma omp critical
                                 {
                                      if (potential_new_dist < distance[v_local]) { // Double-check in critical section
                                          distance[v_local] = potential_new_dist;
                                          if (!newly_added_to_next_frontier[v_local]) {
                                              newly_added_to_next_frontier[v_local] = true;
                                              next_frontier.push_back(v_local);
                                          }
                                      }
                                 }
                             }
                         } else {
                              cerr << "Warning (Rank " << world_rank << "): Local adjncy contains out-of-bounds vertex index: " << v_local << " for local vertex " << u_local << endl;
                         }
                     } else {
                          cerr << "Warning (Rank " << world_rank << "): Local row_ptr/col_ind mismatch at edge index: " << j << " for local vertex " << u_local << endl;
                     }
                 }
            } else {
                 cerr << "Warning (Rank " << world_rank << "): Frontier contains out-of-bounds local vertex ID: " << u_local << endl;
            }
        }
         // next_frontier now contains local vertices whose distance improved from local neighbors

        // Phase 2: Identify and Prepare Updates to Send to Ghost Nodes
        // Iterate through vertices *in the current frontier* that are also boundary vertices.
        // Collect updates to send to ghost neighbors.

        // Data structure to hold updates to be sent: {destination_rank, global_ghost_node_id, new_distance}
        // Organize by destination rank for efficient packing
        vector<vector<pair<int, float>>> updates_by_dest_rank(world_size);

        // Only iterate through boundary vertices that were in the current frontier
        // We can potentially parallelize this loop
        #pragma omp parallel for schedule(static)
        for (size_t local_u_idx = 0; local_u_idx < boundary_info.local_boundary_vertices.size(); ++local_u_idx) {
             int local_u = boundary_info.local_boundary_vertices[local_u_idx];

            // Check if this boundary vertex was in the current frontier and is reachable
             // Use the 'in_frontier' boolean array.
             if (local_u >= 0 && local_u < in_frontier.size() && in_frontier[local_u] && distance[local_u] != INF) {
                // This boundary vertex was active and is reachable. Collect updates for its ghost neighbors.
                if (local_u >= 0 && local_u < boundary_info.boundary_to_ghost.size()) { // Bounds check
                     for (const auto& gc : boundary_info.boundary_to_ghost[local_u]) {
                         // Calculate potential new distance (cast weight to float)
                         float potential_new_dist = distance[local_u] + static_cast<float>(gc.weight);

                         // Add the update to the list for the correct destination rank
                         // Need a critical section here as multiple threads might write to the same updates_by_dest_rank vector
                         #pragma omp critical
                         {
                            updates_by_dest_rank[gc.owner_rank].push_back({gc.global_ghost_id, potential_new_dist});
                         }
                     }
                 } // else: Should not happen if local_u comes from local_boundary_vertices list or boundary_to_ghost is not sized correctly
             } else if (local_u < 0 || local_u >= in_frontier.size()) {
                 cerr << "Warning (Rank " << world_rank << "): local_boundary_vertices contains out-of-bounds local ID: " << local_u << endl;
             }
        }


        // Phase 3: MPI Communication - Exchange Update Counts and Data

        // 3a: Exchange counts of updates to be sent/received
        vector<int> send_counts(world_size, 0);
        for (int rank = 0; rank < world_size; ++rank) {
            send_counts[rank] = updates_by_dest_rank[rank].size();
        }

        vector<int> recv_counts(world_size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, comm);

        // 3b: Prepare send buffer (flattened data)
        vector<pair<int, float>> send_buffer;
        vector<int> send_displs(world_size, 0);
        int total_send_count = 0;
        for (int rank = 0; rank < world_size; ++rank) {
            send_displs[rank] = total_send_count;
            send_buffer.insert(send_buffer.end(), updates_by_dest_rank[rank].begin(), updates_by_dest_rank[rank].end());
            total_send_count += send_counts[rank];
        }

        // 3c: Prepare receive buffer
        vector<int> recv_displs(world_size, 0);
        int total_recv_count = 0;
        for (int rank = 0; rank < world_size; ++rank) {
            recv_displs[rank] = total_recv_count;
            total_recv_count += recv_counts[rank];
        }
        vector<pair<int, float>> recv_buffer(total_recv_count);

        // 3d: Exchange update data using MPI_Alltoallv
        MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_PAIR_INT_FLOAT,
                      recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_PAIR_INT_FLOAT,
                      comm);


        // Phase 4: Apply Incoming Updates (Can be parallelized)
        // Iterate through received updates and apply improvements to local vertices
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < recv_buffer.size(); ++i) {
             const auto& incoming_update = recv_buffer[i];
             int global_v_id = incoming_update.first;
             float potential_new_dist = incoming_update.second;

            // Find the local ID of this global vertex (should be local to this process)
            // Use global_to_local map
            if (global_v_id >= 0 && global_v_id < global_to_local.size()) { // Basic bounds check
                 int v_local = global_to_local[global_v_id];

                 if (v_local != -1) { // Check if this process owns the vertex
                     if (potential_new_dist < distance[v_local]) {
                          // Use atomic or critical section for updating distance and newly_added_to_next_frontier
                          #pragma omp critical
                          {
                              if (potential_new_dist < distance[v_local]) { // Double-check
                                  distance[v_local] = potential_new_dist;
                                  if (!newly_added_to_next_frontier[v_local]) { // Check if already added locally or by another thread
                                      newly_added_to_next_frontier[v_local] = true;
                                      next_frontier.push_back(v_local);
                                  }
                              }
                          }
                     }
                 } // else: received an update for a vertex not owned by this process (shouldn't happen with correct ghost info)
            } else {
                 cerr << "Warning (Rank " << world_rank << "): Received update for out-of-bounds global vertex ID: " << global_v_id << endl;
            }
        }

        // Phase 5: Global Synchronization and Convergence Check (MPI_Allreduce)
        // The next frontier is composed of vertices updated locally OR by incoming messages.
        int local_updates_count = next_frontier.size();
        int total_updates_count;
        MPI_Allreduce(&local_updates_count, &total_updates_count, 1, MPI_INT, MPI_SUM, comm);

         if (world_rank == 0) {
             cout << "Iteration complete. Total vertices added to next frontier across all processes: " << total_updates_count << endl;
         }


        if (total_updates_count == 0) {
            global_converged = true;
        } else {
            // Only update frontier if not converged
             frontier = next_frontier; // Update frontier for the next iteration

            // Reset tracking for the next iteration
            // in_frontier needs to be updated based on the new frontier
            in_frontier.assign(num_local_vertices, false);
            for(int v : frontier) in_frontier[v] = true;
        }


    } // End while (!global_converged)

    // Free the created MPI datatype
    MPI_Type_free(&MPI_PAIR_INT_FLOAT);

    return distance; // Return local distances
}


// Function to create a directory
int create_directory(const std::string& path, int world_rank) {
#ifdef _MSC_VER
    int ret = mkdir(path.c_str());
#else
    int ret = mkdir(path.c_str(), 0775); // Use appropriate permissions
#endif
    if (ret == 0 || errno == EEXIST) {
        return 0; // Success or directory already exists
    } else {
        cerr << "Error (Rank " << world_rank << "): Error creating directory: " << path << " - " << strerror(errno) << endl;
        return -1; // Indicate error
    }
}


// Main Function combining METIS partitioning and Distributed SSSP with file I/O
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI

    int world_size; // Number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank; // Rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 4) {
        if (world_rank == 0) {
             cerr << "Usage: " << argv[0] << " <number_of_partitions> <source_vertex_global_id> <number_of_threads>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int nparts_arg = atoi(argv[1]);
    int source_global_id = atoi(argv[2]);
    int num_threads_arg = atoi(argv[3]);

    if (world_size != nparts_arg) {
        if (world_rank == 0) {
            cerr << "Error: Number of MPI processes (" << world_size << ") must match the number of partitions (" << nparts_arg << ")." << endl;
        }
        MPI_Finalize();
        return 1;
    }
    if (num_threads_arg <= 0) {
         if (world_rank == 0) {
             cerr << "Error: Number of threads must be positive." << endl;
         }
         MPI_Finalize();
         return 1;
    }

    // Set the number of OpenMP threads for this process
    omp_set_num_threads(num_threads_arg);
    if (world_rank == 0) {
        cout << "Running with " << world_size << " MPI processes and " << omp_get_max_threads() << " OpenMP threads per process." << endl;
    }


    // --- File paths for Global CSR data ---
    // Corrected file paths based on user's previous code
    const string correct_global_row_file_path = "/home/hp/Desktop/PDC_FP/BIGData/csr_row_ptr.txt";
    const string correct_global_col_file_path = "/home/hp/Desktop/PDC_FP/BIGData/csr_col_idx.txt"; // This is adjncy/col_idx
    const string correct_global_val_file_path = "/home/hp/Desktop/PDC_FP/BIGData/csr_values.txt"; // This is adjwgt/edge values


    // --- Perform METIS Partitioning and Save Files (only on root process) ---
    idx_t* metis_xadj = NULL;
    idx_t* metis_adjncy = NULL;
    idx_t* metis_adjwgt = NULL; // Use idx_t for weights as METIS expects this
    idx_t num_global_vertices = 0;
    idx_t num_global_edges = 0;
    idx_t* global_part_metis_output = NULL; // Output from METIS on root

    const std::string main_output_dir = "METISPartitioning";

    if (world_rank == 0) {
        // Create the main output directory
        if (create_directory(main_output_dir, world_rank) != 0) {
             MPI_Abort(MPI_COMM_WORLD, 1); // Abort if root can't create directory
        }

        // Load Global Graph on Root for METIS
        vector<int> global_row_ptr_vec = loadIntArray(correct_global_row_file_path, world_rank);
        vector<int> global_col_ind_vec = loadIntArray(correct_global_col_file_path, world_rank);
        vector<int> global_adjwgt_vec = loadIntArray(correct_global_val_file_path, world_rank); // Load global weights as int

        if (global_row_ptr_vec.empty() || global_col_ind_vec.empty() || global_adjwgt_vec.empty()) {
             cerr << "Error (Rank 0): Failed to load global graph files." << endl;
             MPI_Abort(MPI_COMM_WORLD, 1); // Abort if root failed to load global data
        }

        num_global_vertices = global_row_ptr_vec.size() - 1;
        num_global_edges = global_col_ind_vec.size();

        if (num_global_edges != global_adjwgt_vec.size()) {
             cerr << "Error (Rank 0): Mismatch between size of global col_index (" << global_col_ind_vec.size() << ") and values (" << global_adjwgt_vec.size() << ") files." << endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (num_global_vertices > 0 && global_row_ptr_vec[num_global_vertices] != num_global_edges) {
             cerr << "Error (Rank 0): Mismatch between last element of global row_ptr (" << global_row_ptr_vec[num_global_vertices] << ") and total number of global edges (" << num_global_edges << ")." << endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
        }


        // Convert vectors to idx_t arrays for METIS (on root)
         metis_xadj = (idx_t*)malloc((num_global_vertices + 1) * sizeof(idx_t));
         metis_adjncy = (idx_t*)malloc(num_global_edges * sizeof(idx_t));
         metis_adjwgt = (idx_t*)malloc(num_global_edges * sizeof(idx_t));
         global_part_metis_output = (idx_t*)malloc(num_global_vertices * sizeof(idx_t));


         if ((num_global_vertices > 0 && (!metis_xadj || !metis_adjncy || !metis_adjwgt || !global_part_metis_output)) || (num_global_vertices == 0 && (!metis_xadj || !global_part_metis_output))) {
              perror("Error (Rank 0) allocating memory for METIS arrays");
              free(metis_xadj); free(metis_adjncy); free(metis_adjwgt); free(global_part_metis_output);
              MPI_Abort(MPI_COMM_WORLD, 1);
         }

         for(size_t i = 0; i < global_row_ptr_vec.size(); ++i) metis_xadj[i] = global_row_ptr_vec[i];
         for(size_t i = 0; i < global_col_ind_vec.size(); ++i) metis_adjncy[i] = global_col_ind_vec[i];
         for(size_t i = 0; i < global_adjwgt_vec.size(); ++i) metis_adjwgt[i] = global_adjwgt_vec[i];


        // METIS options (optional)
        idx_t ncon = 1; // Number of balancing constraints (usually 1)
        idx_t *vwgt = NULL; // Vertex weights (can be NULL)
        idx_t *vsize = NULL; // Vertex sizes (can be NULL)
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        idx_t objval;

        int ret = METIS_ERROR; // Initialize ret to an error state

        // Only partition if there are vertices
        if (num_global_vertices > 0) {
             ret = METIS_PartGraphKway(
                 &num_global_vertices, &ncon, metis_xadj, metis_adjncy,
                 vwgt, vsize, metis_adjwgt, // Pass adjwgt for edge weights
                 &nparts_arg, NULL, NULL, options, &objval, global_part_metis_output
             );
        } else {
             ret = METIS_OK; // Consider 0 vertices graph as successfully partitioned
             objval = 0;     // No edges to cut
             printf("Rank 0: Graph has 0 vertices. No partitioning needed.\n");
        }


        if (ret == METIS_OK) {
            cout << "Rank 0: METIS partitioned graph successfully into " << nparts_arg << " parts." << endl;
            cout << "Rank 0: Partitioning Objective Value (Weighted Edge Cut): " << objval << endl;

            // --- Save the global partition array ---
            std::string global_partition_filename = main_output_dir + "/global_partition.txt";
            std::ofstream global_partition_outfile(global_partition_filename);
            if (!global_partition_outfile.is_open()) {
                perror(("Error (Rank 0) opening global partition file for writing: " + global_partition_filename).c_str());
                // Handle this error appropriately, maybe return an error code
                free(metis_xadj); free(metis_adjncy); free(metis_adjwgt); free(global_part_metis_output);
                 MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                for (idx_t i = 0; i < num_global_vertices; ++i) {
                    global_partition_outfile << global_part_metis_output[i] << "\n"; // Write partition ID for each vertex on a new line
                }
                global_partition_outfile.close();
                printf("Rank 0: Global partition array saved to %s\n", global_partition_filename.c_str());
            }

            // --- Save Subgraphs to METISPartitioning directory ---
            // Create a vector of vectors to store vertices in each partition
            std::vector<std::vector<idx_t>> partitions_vertices(nparts_arg);
            if (num_global_vertices > 0) {
                for (idx_t i = 0; i < num_global_vertices; ++i) {
                    if (global_part_metis_output[i] >= 0 && global_part_metis_output[i] < nparts_arg) {
                        partitions_vertices[global_part_metis_output[i]].push_back(i);
                    } else {
                        fprintf(stderr, "Warning (Rank 0): Vertex %d assigned to invalid partition %d\n", (int)i, (int)global_part_metis_output[i]);
                    }
                }
            }

            for (idx_t p = 0; p < nparts_arg; ++p) {
                std::string subgraph_dir_name = main_output_dir + "/subgraph" + std::to_string(p + 1);
                if (create_directory(subgraph_dir_name, world_rank) != 0) {
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
                     std::vector<idx_t> original_to_subgraph_map(num_global_vertices, -1);
                     for (idx_t i = 0; i < subgraph_nvtxs; ++i) {
                         original_to_subgraph_map[current_partition_vertices[i]] = i;
                     }

                     // Build CSR for the subgraph
                     for (idx_t i = 0; i < subgraph_nvtxs; ++i) {
                         idx_t original_u = current_partition_vertices[i];
                         sub_xadj_vec[i] = subgraph_num_edges;

                         if (original_u >= 0 && original_u < num_global_vertices) { // Bounds check for original_u
                             for (idx_t j = metis_xadj[original_u]; j < metis_xadj[original_u + 1]; ++j) {
                                 if (j >= 0 && j < num_global_edges) { // Bounds check for edge index j
                                     idx_t original_v = metis_adjncy[j];
                                     idx_t original_weight = metis_adjwgt[j];

                                     // Check if the neighbor is also in the current partition
                                     if (original_v >= 0 && original_v < num_global_vertices && global_part_metis_output[original_v] == p) {
                                         sub_adjncy_vec.push_back(original_to_subgraph_map[original_v]);
                                         sub_values_vec.push_back(original_weight);
                                         subgraph_num_edges++;
                                     } else if (original_v < 0 || original_v >= num_global_vertices) {
                                          fprintf(stderr, "Warning (Rank 0): Original adjncy contains out-of-bounds vertex index %d for vertex %d\n", (int)original_v, (int)original_u);
                                     }
                                 } else {
                                      fprintf(stderr, "Warning (Rank 0): Global xadj/adjncy mismatch at edge index: %d for vertex %d\n", (int)j, (int)original_u);
                                 }
                             }
                         } else {
                              fprintf(stderr, "Warning (Rank 0): current_partition_vertices contains out-of-bounds global ID: %d\n", (int)original_u);
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
                    sub_values_vec.data(),
                    world_rank // Pass rank for error reporting
                );
                printf("Rank 0: Subgraph %d saved to directory %s (Vertices: %d, Edges: %d)\n", (int)p + 1, subgraph_dir_name.c_str(), (int)subgraph_nvtxs, (int)subgraph_num_edges);
            }

        } else {
            fprintf(stderr, "Rank 0: METIS partitioning failed with error code: %d\n", ret);
            // Refer to METIS documentation for error codes
            free(metis_xadj); free(metis_adjncy); free(metis_adjwgt); free(global_part_metis_output);
             MPI_Abort(MPI_COMM_WORLD, ret); // Abort with METIS error code
        }

        // --- Free allocated memory on root ---
        free(metis_xadj);
        free(metis_adjncy);
        free(metis_adjwgt);
        free(global_part_metis_output);

        // Broadcast the actual number of global vertices from root to all ranks
        MPI_Bcast(&num_global_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);

    } else { // Non-root processes
         // Receive the number of global vertices from root
         MPI_Bcast(&num_global_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // 3. Synchronize all processes after root finishes partitioning and writing
    // Use MPI_COMM_WORLD as the communicator
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Load Partition and Local Subgraph Data (ON ALL RANKS) ---
    // Load global_partition.txt
    vector<int> global_part = loadIntArray(main_output_dir + "/global_partition.txt", world_rank);

    if (global_part.size() != num_global_vertices) {
         cerr << "Error (Rank " << world_rank << "): Mismatch between number of global vertices (" << num_global_vertices << ") and partition array size (" << global_part.size() << ") after loading file." << endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Basic check to ensure partition IDs are within expected range
    for (int part_id : global_part) {
        if (part_id < 0 || part_id >= world_size) {
            cerr << "Error (Rank " << world_rank << "): Invalid partition ID (" << part_id << ") found in global_partition.txt for a process count of " << world_size << "." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }


    // Load local subgraph CSR files
    string subgraph_dir = main_output_dir + "/subgraph" + to_string(world_rank + 1);
    vector<int> local_row_ptr, local_col_ind, local_weights;
    loadLocalCSRGraph(subgraph_dir + "/row_ptr_sub_graph.txt",
                      subgraph_dir + "/col_idx_sub_graph.txt",
                      subgraph_dir + "/values_sub_graph.txt",
                      local_row_ptr, local_col_ind, local_weights, world_rank);

    int num_local_vertices = local_row_ptr.size() - 1;
    if (num_local_vertices < 0 && !local_row_ptr.empty()) { // Handle case of 0 vertices correctly
         num_local_vertices = 0;
    } else if (local_row_ptr.empty()) {
         num_local_vertices = 0;
         if (world_rank == 0) cerr << "Warning: Local row_ptr is empty for rank " << world_rank << ". Assuming 0 local vertices." << endl;
    }


    // --- Create Global to Local and Local to Global Mappings (ON ALL RANKS) ---
    vector<int> global_to_local(num_global_vertices, -1);
    vector<int> local_to_global(num_local_vertices);

    int local_idx = 0;
    for (int i = 0; i < num_global_vertices; ++i) {
        if (global_part[i] == world_rank) {
            if (local_idx < num_local_vertices) { // Add bounds check
                global_to_local[i] = local_idx;
                local_to_global[local_idx] = i;
                local_idx++;
            } else {
                 cerr << "Error (Rank " << world_rank << "): More vertices assigned to this partition in global_part (" << local_idx << ") than found in local subgraph data (" << num_local_vertices << ")." << endl;
                 MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    if (local_idx != num_local_vertices) {
        cerr << "Error (Rank " << world_rank << "): Mismatch in calculated local vertices (" << local_idx << ") and loaded subgraph size (" << num_local_vertices << ") for rank " << world_rank << "." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    // --- Identify Boundary Vertices and Ghost Nodes (ON ALL RANKS) ---
    // Load Global Graph on all ranks for boundary identification
    vector<int> global_row_ptr = loadIntArray(correct_global_row_file_path, world_rank);
    vector<int> global_col_ind = loadIntArray(correct_global_col_file_path, world_rank);
    vector<int> global_adjwgt = loadIntArray(correct_global_val_file_path, world_rank); // Load global weights as int

     if (global_row_ptr.empty() || global_col_ind.empty() || global_adjwgt.empty()) {
          cerr << "Error (Rank " << world_rank << "): Failed to load global graph files for boundary identification." << endl;
          MPI_Abort(MPI_COMM_WORLD, 1); // Abort if failed to load global data
     }
     if (global_row_ptr.size() - 1 != num_global_vertices || global_col_ind.size() != global_adjwgt.size() || (num_global_vertices > 0 && global_row_ptr[num_global_vertices] != global_col_ind.size())) {
         cerr << "Error (Rank " << world_rank << "): Global graph file consistency check failed after loading for boundary identification." << endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
     }


    BoundaryInfo boundary_info = identify_boundary_and_ghost_nodes(
        world_rank,
        num_global_vertices,
        global_row_ptr,
        global_col_ind,
        global_adjwgt,
        global_part,
        local_to_global
    );

     if (world_rank == 0) {
        cout << "Boundary and ghost nodes identification complete for all processes." << endl;
     }
     cout << "Rank " << world_rank << " has " << boundary_info.local_boundary_vertices.size() << " boundary vertices." << endl;

    // --- Optional: Print Boundary and Ghost Node Information for Verification ---
    /*
    cout << "Rank " << world_rank << " Boundary and Ghost Node Details:" << endl;
    if (boundary_info.local_boundary_vertices.empty()) {
        cout << "  No boundary vertices in this partition." << endl;
    } else {
        // Sort local boundary vertices for consistent output
        sort(boundary_info.local_boundary_vertices.begin(), boundary_info.local_boundary_vertices.end());

        for (int local_u : boundary_info.local_boundary_vertices) {
            int global_u = local_to_global[local_u];
            cout << "  Local Boundary Vertex " << local_u << " (Global ID: " << global_u << "):" << endl;
            if (local_u >= 0 && local_u < boundary_info.boundary_to_ghost.size()) { // Bounds check
                 if (boundary_info.boundary_to_ghost[local_u].empty()) {
                      cout << "    No ghost node connections (should not happen if it's a boundary vertex)." << endl;
                 } else {
                     for (const auto& gc : boundary_info.boundary_to_ghost[local_u]) {
                         cout << "    -> Ghost Node Global ID: " << gc.global_ghost_id
                              << ", Owner Rank: " << gc.owner_rank
                              << ", Edge Weight: " << gc.weight << endl;
                     }
                 }
            } else {
                 cerr << "Warning (Rank " << world_rank << "): local_boundary_vertices contains out-of-bounds local ID: " << local_u << " during printing." << endl;
            }
        }
    }
    cout << "---------------------------------------------------" << endl;
    */


    // --- Run Distributed SSSP ---
    vector<float> final_local_distances = sssp_distributed(
        world_rank,
        world_size,
        local_row_ptr,
        local_col_ind,
        local_weights, // Pass local weights as int
        num_local_vertices,
        source_global_id,
        local_to_global,
        global_to_local,
        boundary_info,
        MPI_COMM_WORLD // Pass the MPI communicator
    );

    // --- Output Results (ON ALL RANKS) ---
    // Save local distances to file
    ofstream outfile("output_distances_rank_" + to_string(world_rank) + ".txt");
    if (!outfile.is_open()) {
         cerr << "Error (Rank " << world_rank << "): Unable to open output file: " << "output_distances_rank_" + to_string(world_rank) + ".txt" << " - " << strerror(errno) << endl;
    } else {
        outfile << "Distances for vertices in Partition " << world_rank << " (Original IDs):\n";
        for (int i = 0; i < num_local_vertices; ++i) {
            int global_id = local_to_global[i];
            outfile << "Vertex " << global_id << " (Local ID " << i << "): ";
            if (final_local_distances[i] >= INF) // Use >= INF for comparison
                outfile << "Unreachable" << endl;
            else
                outfile << final_local_distances[i] << endl;
        }
        outfile.close();
        cout << "Rank " << world_rank << ": Distances written to output_distances_rank_" + to_string(world_rank) + ".txt" << endl;
    }


    // --- MPI Finalize ---
    MPI_Finalize();
    return 0;
}
