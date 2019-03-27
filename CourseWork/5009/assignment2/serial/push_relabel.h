/*
 * Do not change this file.
 * This is a serial version of the push-relabel algorithm
 * Compile: g++ -std=c++11 main.cpp serial_push_relabel.cpp -o serial_push_relabel
 * Run: ./serial_push_relabel <input file>, you will find the output in 'output.txt' file
 */


#pragma once

/**
 * Push-relabel algorithm. Find the maximum-flow from vertex src to vertex sink.
 * @param N number of vertices
 * @param src src vertex of the maximum flow problem
 * @param sink sink vertex of the maximum flow problem
 * @param *cap capacity matrix (positive for each edge, zero for non-edge)
 * @param *flow the flow matrix
 * @attention we will use the flow matrix (my_rank==0) for the verification
*/
int push_relabel(int N, int src, int sink, int *cap, int *flow);

namespace utils {
    /*
     * translate 2-dimension coordinate to 1-dimension
     */
    int idx(int x, int y, int n);
}