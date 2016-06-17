/**
 * filename: constructGraph.cpp
 * contents: this file contains the definition of the kernel 1 routine to
 * construct the graph
 */

#include "constructGraph.h"

/**
 * constructs the graph on GPU from the CPU edge list
 */
void ConstructGraph(EdgeList &edges, int rank, int np) {
  
  // edges data
  long long *edgeList = edges.edges();
  int numEdges = edges.size();

  // find number from vertices from edge list
  int numNodes = 0;
  for (int i = 0; i < numEdges; ++i) {
    if (edgeList[i] > numNodes) {
      numNodes = edgeList[i];
    }
  }
  numNodes += 1;

  /**TODO
   * 1. chunk adj matrix allocation by rank
   * 2. in kernel: convert edge list to adj matrix
   * 3. free edge list from host
   * 4. allocate adj list on dev
   * 5. launch kernel to convert adj matrix to adj list
   */

  // allocate adjacency matrix on device & set to 0's
  void *adjMatrix;
  size_t size = sizeof(bool) * numNodes * numNodes;
  CUDA_CALL(cudaMalloc(&adjMatrix, size));
  CUDA_CALL(cudaMemset(adjMatrix, 0, size));

  // allocate buffer for edge list
  void *devEdgeList;
  size = sizeof(long long) * numEdges * 2;
  CUDA_CALL(cudaMalloc(&devEdgeList, size));
  CUDA_CALL(cudaMemcpy(devEdgeList, (void *)edgeList, size, cudaMemcpyHostToDevice));


  // cleanup adj matrix
  CUDA_CALL(cudaFree(adjMatrix));
  CUDA_CALL(cudaFree(devEdgeList));
          
}
