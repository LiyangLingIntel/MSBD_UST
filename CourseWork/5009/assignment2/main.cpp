/*
 * Do not change this file.
 */

#include <cassert>
#include <cstring>
#include <cmath>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <queue>

using namespace std;
using namespace std::chrono;

#include "pthread_push_relabel.h"

namespace utils {
    int N;
    int src;
    int sink;
    int *cap;
    int *flow;

    int idx(int x, int y, int n) {
        return x * n + y;
    }

    int read_file(string filename) {
        std::ifstream inputf(filename, std::ifstream::in);
        inputf >> N;
        inputf >> src;
        inputf >> sink;
        fprintf(stderr, "N: %d, src: %d, sink: %d\n", N, src, sink);

        // input matrix should be smaller than 20K * 20KB (400MB, we don't have two much memory for multi-processors)
        assert(N < (1024 * 20));
        cap = (int *) calloc(N * N, sizeof(int));
        flow = (int *) calloc(N * N, sizeof(int));
        while (inputf.good()) {
            int i, j;
            inputf >> i >> j;
            if (!inputf.good()) {
                break;
            }
            inputf >> cap[idx(i, j, N)];
        }
        return 0;
    }

    int print_result() {
        int64_t flow_sum = 0;
        for (auto u = 0; u < utils::N; u++) {
            flow_sum += utils::flow[utils::idx(u, utils::sink, utils::N)];
        }
        fprintf(stderr, "max flow:%li\n", flow_sum);

#ifdef DEBUG
        std::ofstream outputf("output.txt", std::ofstream::out);
        outputf << flow_sum << "\n";

        for (auto u = 0; u < utils::N; u++) {
            for (auto v = 0; v < utils::N; v++) {
                if (utils::flow[utils::idx(u, v, N)] > 0) {
                    outputf << u << " " << v << " " << utils::flow[utils::idx(u, v, N)] << "\n";
                }
            }
        }
        outputf << endl;
#endif
        return 0;
    }

    int verify_valid_flow() {
        for (auto u = 0; u < N; u++) {
            for (auto v = 0; v < N; v++) {
                if (flow[idx(u, v, N)] > cap[idx(u, v, N)]) {
                    fprintf(stderr, "violate (capacity constraint)\n");
                    return -1;
                }
            }
        }
        for (auto u = 0; u < N; u++) {
            for (auto v = 0; v < N; v++) {
                if (flow[idx(u, v, N)] + flow[idx(v, u, N)] != 0) {
                    fprintf(stderr, "violate (antisymmetry constraint)\n");
                    return -2;
                }
            }
        }
        int64_t flow_sum = 0;
        for (auto u = 0; u < N; u++) {
            for (auto v = 0; v < N; v++) {
                if (v != src && v != sink) {
                    flow_sum += flow[idx(u, v, N)];
                }
            }
        }
        if (flow_sum != 0) {
            fprintf(stderr, "violate (flow conservation constraint))\n");
            return -3;
        }
        return 0;
    }

    int verify_maximum_flow() {
        queue<int> q;
        vector<bool> is_visited((size_t) utils::N, false);

        // BFS to check the existence of augmenting path on the residual network.
        q.push(utils::src);
        is_visited[utils::src] = true;
        while (!q.empty()) {
            auto u = q.front();
            q.pop();
            if (u == utils::sink) {
                fprintf(stderr, "exist an augmenting path on the residual network\n");
                return -1;
            }
            for (auto v = 0; v < utils::N; v++) {
                auto residual_cap = utils::cap[utils::idx(u, v, utils::N)] -
                                    utils::flow[utils::idx(u, v, utils::N)];
                if (residual_cap > 0 && !is_visited[v]) {
                    q.push(v);
                    is_visited[v] = true;
                }
            }
        }
        return 0;
    }
}

int main(int argc, char **argv) {
    assert(argc > 1 && "Input file was not found!");
    string filename = argv[1];
    int num_threads = atoi(argv[2]);
    assert(utils::read_file(filename) == 0);

    int *cap_mat = (int *) malloc(utils::N * utils::N * sizeof(int));
    memcpy(cap_mat, utils::cap, utils::N * utils::N * sizeof(int));

    // Push relabel algorithm.
    auto start_clock = high_resolution_clock::now();
    push_relabel(num_threads, utils::N, utils::src, utils::sink, cap_mat, utils::flow);
    auto end_clock = high_resolution_clock::now();
    fprintf(stderr, "Elapsed Time: %.9lf s\n",
            duration_cast<nanoseconds>(end_clock - start_clock).count() / pow(10, 9));

    // Verify the validity of flow; Verify the maximum flow.
    int ret = utils::verify_valid_flow();
    if (ret != 0) {
        fprintf(stderr, "err code: %d\n", ret);
    } else {
        if (utils::verify_maximum_flow() == -1) {
            fprintf(stderr, "not maximum flow, exist augmenting path on the residual network\n");
        } else {
            utils::print_result();
        }
    }

    free(cap_mat);
    free(utils::cap);
    free(utils::flow);
    return 0;
}