/**
 * filename: bfsStep.h
 * contents: this file contains the kernel to perform a bfs step, as well as
 * a wrapper for the functio to call from the .cpp file
 */

#ifndef BFS_STEP_H
#define BFS_STEP_H

/**
 * wrapper for cuda kernel
 */ 
void bfsStep(int threadsPerBlock,
             int numBlocks,
             bool *adjMatrix,
             int numNodes,
             int offset,
             int numLocalNodes,
             bool *visited,
             bool *done,
             long long *parentInfo,
             int rank,
             int key);

/**
 * perform a single bfs step on the graph
 */
__global__ void bfsStepKernel(bool *adjMatrix, 
                              int numNodes,
                              int offset,
                              int numLocalNodes,
                              bool *visited, 
                              bool *done,
                              long long *parentInfo,
                              int rank, 
                              int key);

#endif // BFS_STEP_H
 
