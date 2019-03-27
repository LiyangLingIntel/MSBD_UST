/**
 * Name: Liyang Ling
 * Student id: 20527456
 * ITSC email: llingab@connect.ust.hk
*/
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>

#include "mpi_push_relabel.h"

using namespace std;

/*
 *  You can add helper functions and variables as you wish.
 */
#define ROOT (0)

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int local_n, int src) {
    dist[src] = local_n;
    for (auto v = 0; v < local_n; v++) {
        flow[utils::idx(src, v, local_n)] = cap[utils::idx(src, v, local_n)];
        flow[utils::idx(v, src, local_n)] = -flow[utils::idx(src, v, local_n)];
        excess[v] = flow[utils::idx(src, v, local_n)];
    }
}

int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow) {
    /*
     * my_rank should be between 0 to p-1
     * 
     */
    int local_src;
    int local_sink;
    int local_n;

    if (my_rank == ROOT) {
        local_n = N;
        local_src = src;
        local_sink = sink;
    }
    MPI_Bcast(&local_n, 1, MPI_INT, ROOT, comm);
    MPI_Bcast(&local_src, 1, MPI_INT, ROOT, comm);
    MPI_Bcast(&local_sink, 1, MPI_INT, ROOT, comm);

    int *dist = (int *) calloc(local_n, sizeof(int));
    int *stash_dist = (int *) calloc(local_n, sizeof(int));
    auto *excess = (int64_t *) calloc(local_n, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(local_n, sizeof(int64_t));

    int *local_cap = (int *) malloc(sizeof(int) * local_n * local_n);
    int *local_flow = (int *) malloc(sizeof(int) * local_n * local_n);
	int *stash_send = (int *) calloc(local_n * local_n, sizeof(int));

    if (my_rank == ROOT) {
        memcpy(local_cap, cap, sizeof(int)*local_n*local_n);
        memcpy(local_flow, flow, sizeof(int)*local_n*local_n);
    }
    MPI_Bcast(local_cap, local_n * local_n, MPI_INT, ROOT, comm);
    MPI_Bcast(local_flow, local_n * local_n, MPI_INT, ROOT, comm);

    // PreFlow.
    pre_flow(dist, excess, local_cap, local_flow, local_n, local_src);

    vector<int> active_nodes;
    for (auto u = 0; u < local_n; u++) {
        if (u != local_src && u != local_sink) {
            active_nodes.emplace_back(u);
        }
    }

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        // intitialize
        int avg_nodes = (active_nodes.size() + p - 1) / p; // make sure no remainder and no extra 1
        int p_start = my_rank * avg_nodes;
        int p_end = min<int>((my_rank + 1) * avg_nodes, active_nodes.size());

        // Stage 1: push.
        for (auto i=p_start; i < p_end; i++) {
            auto u = active_nodes[i];
            for (auto v = 0; v < local_n; v++) {
                auto residual_cap = local_cap[utils::idx(u, v, local_n)] - local_flow[utils::idx(u, v, local_n)];
                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                    stash_send[utils::idx(u, v, local_n)] = std::min<int64_t>(excess[u], residual_cap);
                    excess[u] -= stash_send[utils::idx(u, v, local_n)];
                }
            }
        }

        for (auto i=0; i<p; i++) {
            for (auto p_i = avg_nodes * i; p_i < min<int>(active_nodes.size(), avg_nodes * (i + 1)); p_i++) {
                auto u = active_nodes[p_i];
                MPI_Bcast(stash_send+utils::idx(u, 0, local_n), local_n, MPI_INT, i, comm);
            }
        }

        for (auto p_i = 0; p_i < active_nodes.size(); p_i++) {
            auto u = active_nodes[p_i];
            bool need_update = !(p_i>=p_start && p_i<p_end);
            for (auto v = 0; v < local_n; v++) {
                if (stash_send[utils::idx(u, v, local_n)] > 0) {
                    if (need_update) {
                        excess[u] -= stash_send[utils::idx(u, v, local_n)];
                    }
                    local_flow[utils::idx(u, v, local_n)] += stash_send[utils::idx(u, v, local_n)];
                    local_flow[utils::idx(v, u, local_n)] -= stash_send[utils::idx(u, v, local_n)];
                    stash_excess[v] += stash_send[utils::idx(u, v, local_n)];
                    stash_send[utils::idx(u, v, local_n)] = 0;
                }
            }
        }

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, local_n * sizeof(int));
        for (auto i=p_start; i < p_end; i++) {
            auto u = active_nodes[i];
            if (excess[u] > 0) {
                int min_dist = INT32_MAX;
                for (auto v = 0; v < local_n; v++) {
                    auto residual_cap = local_cap[utils::idx(u, v, local_n)] - local_flow[utils::idx(u, v, local_n)];
                    if (residual_cap > 0) {
                        min_dist = min(min_dist, dist[v]);
                        stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }

        // Stage 3: update dist.
        MPI_Allreduce(stash_dist, dist, local_n, MPI_INT, MPI_MAX, comm);

        // Stage 4: apply excess-flow changes for destination vertices.
        for (auto v = 0; v < local_n; v++) {
            if (stash_excess[v] != 0) {
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }

        // Construct active nodes.
        active_nodes.clear();
        for (auto u = 0; u < local_n; u++) {
            if (excess[u] > 0 && u != local_src && u != local_sink) {
                active_nodes.emplace_back(u);
            }
        }
        MPI_Barrier(comm);
    }

    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);
    if (my_rank == ROOT) {
        memcpy(flow, local_flow, sizeof(int)*local_n*local_n);
    }

    free(local_cap);
    free(local_flow);

    return 0;
}
