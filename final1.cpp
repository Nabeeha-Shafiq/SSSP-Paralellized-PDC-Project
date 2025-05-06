//   mpicxx final1.cpp -lmetis -fopenmp -o abc
//  
// mpirun -np 3 ./abc 3 0
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

#ifdef _MSC_VER
#include <direct.h> // For _mkdir on Windows
#define mkdir(path, mode) _mkdir(path)
#endif

using namespace std;

// Initiliazing the infinity for the djikstra's algorithm part.
// Keep as float to avoid overflow issues with accumulated integer weights
const float INF = 1e9;

// Function to read an array of integers from a file (space or newline separated)
vector<int> loadIntArray(const string& filename) {
    vector<int> vec;
    ifstream infile(filename);
    int value;

    if (!infile) {
        cerr << "Error: Unable to open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI on file error
    }
    while (infile >> value) {
        vec.push_back(value);
    }
    return vec;
}

// Function to read an array of floats from a file (space or newline separated)
// We keep this function but will use loadIntArray for weights as requested
vector<float> loadFloatArray(const string& filename) {
    vector<float> vec;
    ifstream infile(filename);
    float value;

    if (!infile) {
        cerr << "Error: Unable to open " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI on file error
    }

    while (infile >> value) {
        vec.push_back(value);
    }
    return vec;
}

// Function to load the CSR Graph data (local subgraph)
void loadLocalCSRGraph(const string& row_file, const string& col_file, const string& val_file,
                       vector<int>& row_ptr, vector<int>& col_ind, vector<int>& weights) { // Weights as int

    row_ptr = loadIntArray(row_file);
    col_ind = loadIntArray(col_file);
    weights = loadIntArray(val_file); // Load weights as int

    if (col_ind.size() != weights.size()) {
        cerr << "Error: Mismatch in number of columns and weights for local subgraph!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
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
        for (int i = global_row_ptr[global_u]; i < global_row_ptr[global_u + 1]; ++i) {
            int global_v = global_col_ind[i];
            int edge_weight = global_adjwgt[i]; // Read weight as int

            // Check if the neighbor is in a different partition
            if (global_v < num_global_vertices && global_part[global_v] != world_rank) {
                // global_u is a boundary vertex
                local_boundary_set.insert(local_u);

                // global_v is a ghost node
                GhostConnection gc;
                gc.global_ghost_id = global_v;
                gc.owner_rank = global_part[global_v];
                gc.weight = edge_weight; // Store weight as int

                // Add this ghost connection to the list for the local boundary vertex
                boundary_info.boundary_to_ghost[local_u].push_back(gc);
            } else if (global_v >= num_global_vertices) {
                 cerr << "Warning (Rank " << world_rank << "): Global adjncy contains out-of-bounds vertex index " << global_v << " for global vertex " << global_u << endl;
            }
        }
    }

    // Populate the list of local boundary vertices
    for (int local_u : local_boundary_set) {
        boundary_info.local_boundary_vertices.push_back(local_u);
    }

    return boundary_info;
}


// SSSP function with MPI communication (Full Implementation Attempt)
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
    if (source_global_id < global_to_local.size() && global_to_local[source_global_id] != -1) {
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

    while (!global_converged) {
        vector<int> next_frontier;
        // Use a temporary boolean array to track vertices added to next_frontier in parallel
        vector<bool> newly_added_to_next_frontier(num_local_vertices, false);

        // Phase 1: Local Relaxation (OpenMP)
        // Relax edges for vertices in the current frontier
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < frontier.size(); ++i) {
            int u_local = frontier[i]; // Local ID of vertex u

            // Relax outgoing edges to local neighbors
            for (int j = local_row_ptr[u_local]; j < local_row_ptr[u_local + 1]; ++j) {
                int v_local = local_col_ind[j]; // Local ID of neighbor v (within this subgraph)
                int weight = local_weights[j]; // Weight as int

                // Calculate potential new distance (cast weight to float for addition)
                float potential_new_dist = distance[u_local] != INF ? distance[u_local] + static_cast<float>(weight) : INF;


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
            }
        }
         // next_frontier now contains local vertices whose distance improved

        // Phase 2: Identify and Prepare Updates to Send to Ghost Nodes
        // Iterate through vertices *in the current frontier* that are also boundary vertices.
        // Collect updates to send to ghost neighbors.

        // Data structure to hold updates to be sent: {destination_rank, global_ghost_node_id, new_distance}
        // Organize by destination rank for efficient packing
        vector<vector<pair<int, float>>> updates_by_dest_rank(world_size);

        // Only iterate through boundary vertices that were in the current frontier
        for (int local_u : frontier) {
            // Check if this local_u is a boundary vertex and has ghost connections
             if (local_u < boundary_info.boundary_to_ghost.size() && !boundary_info.boundary_to_ghost[local_u].empty()) {
                // This vertex has ghost neighbors. Iterate through them.
                if (distance[local_u] != INF) { // Only send updates from reachable vertices
                     for (const auto& gc : boundary_info.boundary_to_ghost[local_u]) {
                         // Calculate potential new distance (cast weight to float)
                         float potential_new_dist = distance[local_u] + static_cast<float>(gc.weight);

                         // Add the update to the list for the correct destination rank
                         updates_by_dest_rank[gc.owner_rank].push_back({gc.global_ghost_id, potential_new_dist});
                     }
                 }
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
        // Need to send/receive pairs of {int, float}. Define MPI datatype for pair<int, float>.
        MPI_Datatype MPI_PAIR_INT_FLOAT;
        int lengths[2] = {1, 1};
        MPI_Aint displacements[2];
        MPI_Datatype types[2] = {MPI_INT, MPI_FLOAT};

        // Calculate displacements correctly using MPI_Get_address
        MPI_Aint base_address;
        pair<int, float> dummy_pair;
        MPI_Get_address(&dummy_pair, &base_address);
        MPI_Get_address(&dummy_pair.first, &displacements[0]);
        MPI_Get_address(&dummy_pair.second, &displacements[1]);
        displacements[0] = MPI_Aint_diff(displacements[0], base_address);
        displacements[1] = MPI_Aint_diff(displacements[1], base_address);

        MPI_Type_create_struct(2, lengths, displacements, types, &MPI_PAIR_INT_FLOAT);
        MPI_Type_commit(&MPI_PAIR_INT_FLOAT);


        MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_PAIR_INT_FLOAT,
                      recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_PAIR_INT_FLOAT,
                      comm);

        // Free the created MPI datatype
        MPI_Type_free(&MPI_PAIR_INT_FLOAT);


        // Phase 4: Apply Incoming Updates
        vector<int> updates_applied_local; // Track local vertices updated by incoming messages
        vector<bool> incoming_updated_next_frontier(num_local_vertices, false); // Track for next_frontier

        // Process received updates
        for (const auto& incoming_update : recv_buffer) {
            int global_v_id = incoming_update.first;
            float potential_new_dist = incoming_update.second;

            // Find the local ID of this global vertex (should be local to this process)
            // Use global_to_local map
            if (global_v_id < global_to_local.size()) { // Basic bounds check
                 int v_local = global_to_local[global_v_id];

                 if (v_local != -1) { // Check if this process owns the vertex
                     if (potential_new_dist < distance[v_local]) {
                          // Use atomic or critical section if applying updates while local relaxation is running
                          // If applying after local relaxation (as structured here), critical is safer for distance and next_frontier
                          #pragma omp critical
                          {
                              if (potential_new_dist < distance[v_local]) { // Double-check
                                  distance[v_local] = potential_new_dist;
                                  if (!newly_added_to_next_frontier[v_local]) { // Check if already added locally
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

        if (total_updates_count == 0) {
            global_converged = true;
        }

        frontier = next_frontier; // Update frontier for the next iteration

        // Reset tracking for the next iteration
         // newly_added_to_next_frontier can be cleared automatically
         // in_frontier needs to be updated based on the new frontier
         in_frontier.assign(num_local_vertices, false);
         for(int v : frontier) in_frontier[v] = true;


         if (world_rank == 0) {
             cout << "Iteration complete. Total vertices added to next frontier across all processes: " << total_updates_count << endl;
         }


    } // End while (!global_converged)

    return distance; // Return local distances
}


// Main Function (Same as before)
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); // Initialize MPI

    int world_size; // Number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank; // Rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3) {
        if (world_rank == 0) {
             cerr << "Usage: " << argv[0] << " <number_of_partitions> <source_vertex_global_id>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int nparts_arg = atoi(argv[1]);
    int source_global_id = atoi(argv[2]);

    if (world_size != nparts_arg) {
        if (world_rank == 0) {
            cerr << "Error: Number of MPI processes (" << world_size << ") must match the number of partitions (" << nparts_arg << ")." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    // --- Load Global Graph and Partition Array (for boundary identification) ---
    // In a more scalable solution, you might distribute this data.
    // For now, each process loads the full global data.

    string global_row_file = "/home/hp/Desktop/PDC_FP/testingData/csr_row_ptr.txt";
    string global_col_file = "/home/hp/Desktop/PDC_FP/testingData/csr_col_idx.txt";
    string global_val_file = "/home/hp/Desktop/PDC_FP/testingData/csr_values.txt"; // Edge weights

    vector<int> global_row_ptr = loadIntArray(global_row_file);
    vector<int> global_col_ind = loadIntArray(global_col_file);
    vector<int> global_adjwgt = loadIntArray(global_val_file); // Load global weights as int
    int num_global_vertices = global_row_ptr.size() - 1;

     // Load the partition array (assuming it was saved by the METIS step)
    // You would need to save this 'part' array to a file in the METIS partitioning script.
    // For now, let's assume a file named 'global_partition.txt' exists in the METISPartitioning directory.
    string global_part_file = "METISPartitioning/global_partition.txt";
    vector<int> global_part = loadIntArray(global_part_file);

     if (global_part.size() != num_global_vertices) {
         if (world_rank == 0) {
             cerr << "Error: Mismatch between number of global vertices and partition array size." << endl;
         }
         MPI_Finalize();
         return 1;
     }
     // Basic check to ensure partition IDs are within expected range
     for (int part_id : global_part) {
         if (part_id < 0 || part_id >= world_size) {
             if (world_rank == 0) {
                 cerr << "Error: Invalid partition ID (" << part_id << ") found in global_partition.txt for a process count of " << world_size << "." << endl;
             }
             MPI_Finalize();
             return 1;
         }
     }


    // --- Load Local Subgraph ---
    string subgraph_dir = "METISPartitioning/subgraph" + to_string(world_rank + 1);
    string local_row_file = subgraph_dir + "/row_ptr_sub_graph.txt";
    string local_col_file = subgraph_dir + "/col_idx_sub_graph.txt";
    string local_val_file = subgraph_dir + "/values_sub_graph.txt";

    vector<int> local_row_ptr, local_col_ind;
    vector<int> local_weights; // Load local weights as int
    loadLocalCSRGraph(local_row_file, local_col_file, local_val_file, local_row_ptr, local_col_ind, local_weights);

    int num_local_vertices = local_row_ptr.size() - 1;

    // --- Create Global to Local and Local to Global Mappings ---
    // global_to_local: Maps global vertex ID to local ID (-1 if not local)
    // local_to_global: Maps local vertex ID to global ID
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
                 cerr << "Error (Rank " << world_rank << "): More vertices assigned to this partition in global_part than found in local subgraph data." << endl;
                 MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    if (local_idx != num_local_vertices) {
        if (world_rank == 0) {
             cerr << "Error: Mismatch in calculated local vertices (" << local_idx << ") and loaded subgraph size (" << num_local_vertices << ") for rank " << world_rank << "." << endl;
        }
        MPI_Finalize();
        return 1;
    }


    // --- Identify Boundary Vertices and Ghost Nodes ---
    BoundaryInfo boundary_info = identify_boundary_and_ghost_nodes(
        world_rank,
        num_global_vertices,
        global_row_ptr,
        global_col_ind,
        global_adjwgt, // Pass global weights as int
        global_part,
        local_to_global // Pass the local_to_global map
    );

     if (world_rank == 0) {
        cout << "Boundary and ghost nodes identification complete for all processes." << endl;
     }
     cout << "Rank " << world_rank << " has " << boundary_info.local_boundary_vertices.size() << " boundary vertices." << endl;

    // --- Print Boundary and Ghost Node Information for Verification ---
    // This section can be commented out after verification if desired
    cout << "Rank " << world_rank << " Boundary and Ghost Node Details:" << endl;
    if (boundary_info.local_boundary_vertices.empty()) {
        cout << "  No boundary vertices in this partition." << endl;
    } else {
        // Sort local boundary vertices for consistent output
        sort(boundary_info.local_boundary_vertices.begin(), boundary_info.local_boundary_vertices.end());

        for (int local_u : boundary_info.local_boundary_vertices) {
            int global_u = local_to_global[local_u];
            cout << "  Local Boundary Vertex " << local_u << " (Global ID: " << global_u << "):" << endl;
            if (boundary_info.boundary_to_ghost[local_u].empty()) {
                 cout << "    No ghost node connections (should not happen if it's a boundary vertex)." << endl;
            } else {
                for (const auto& gc : boundary_info.boundary_to_ghost[local_u]) {
                    cout << "    -> Ghost Node Global ID: " << gc.global_ghost_id
                         << ", Owner Rank: " << gc.owner_rank
                         << ", Edge Weight: " << gc.weight << endl;
                }
            }
        }
    }
    cout << "---------------------------------------------------" << endl;


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

    // --- Output Results (Optional: Collect and print on root or save locally) ---
    // For large graphs, collecting all distances on the root might be memory-intensive.
    // Saving local distances to a file per process is more scalable.

    // Example: Saving local distances to a file per process
    ofstream outfile("output_distances_rank_" + to_string(world_rank) + ".txt");
    if (!outfile.is_open()) {
         cerr << "Error: Unable to open output file for rank " << world_rank << endl;
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