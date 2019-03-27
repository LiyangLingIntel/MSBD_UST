/*
 * Do not change this file.
 * This is a mpi version of the push-relabel algorithm
 * Compile: mpic++ -std=c++11 main.cpp mpi_push_relabel_skeleton.cpp -o mpi_push_relabel
 * Run: mpiexec -n <number of processes> ./mpi_push_relabel <input file>, you will find the output in 'output.txt' file
 */


#pragma once

#include "mpi.h"

/**
 * Push-relabel algorithm. Find the maximum-flow from vertex src to vertex sink.
 * @param my_rank the rank of current process
 * @param p number of processes
 * @param comm the MPI communicator
 * @param N number of vertices
 * @param src src vertex of the maximum flow problem
 * @param sink sink vertex of the maximum flow problem
 * @param *cap capacity matrix (positive for each edge, zero for non-edge)
 * @param *flow the flow matrix
 * @attention we will use the flow matrix (my_rank==0) for the verification
*/
int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow);

namespace utils {
    /*
     * translate 2-dimension coordinate to 1-dimension
     */
    int idx(int x, int y, int n);
}