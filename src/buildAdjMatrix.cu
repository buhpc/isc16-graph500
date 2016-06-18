/**
 * filename: buildAdjMatirx.cu
 * contents: contains the cuda kernel to build the adjancecy
 * matrix from the list of edges
 */

#include "buildAdjMatrix.h"

#include <stdio.h>

/**
 * wrapper for cuda kernel
 */
void buildGraph(int threadsPerBlock,
                int numBlocks,
                int *adjMatrix,
                int numNodes,
                long long *edgeList,
                int numEdges,
                int offset,
                int graphSize,
                int rank) {
  // launch kernel
  buildAdjMatrix<<<numBlocks, threadsPerBlock>>>(adjMatrix,
                                                 numNodes,
                                                 edgeList,
                                                 numEdges,
                                                 offset,
                                                 graphSize, 
                                                 rank);
}

/**
 * constructs and adjacency matrix in device memory from the list
 * of edges
 */
__global__ void buildAdjMatrix(int *adjMatrix,
                               int numNodes,
                               long long *edgeList,
                               int numEdges,
                               int offset,
                               int graphSize,
                               int rank) {

  // each thread gets 1 edges in the edge list to build
  int index = blockIdx.x*blockDim.x + threadIdx.x;


  if (index < numEdges) {
    if (rank == 1) {
      printf("(rank, index) = (%d, %d)\n", rank, index);
    } 
    // get the two vertices to connect
    int vertA = edgeList[index];
    int vertB = edgeList[index + numEdges];
    
    // remove self edges
    if (vertA == vertB) { return; }



    // set edge in both direction
    if (vertA >= offset && vertA < (offset + graphSize)) {
      // vertA is row vertB is column
      atomicOr(&adjMatrix[vertA*numNodes+vertB], 1);
    }
    if (vertB >= offset && vertB < offset + graphSize) {
      // vert b is row vertA is column
      atomicOr(&adjMatrix[vertB*numNodes+vertA], 1);
    }
  }

}
