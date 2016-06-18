/**
 * filename: constructGraph.cpp
 * contents: this file contains the definition of the kernel 1 routine to
 * construct the graph
 */

#include "constructGraph.h"

/**
 * constructs the graph on GPU from the CPU edge list
 */
pair<int, int> ConstructGraph(EdgeList &edges, int rank, int np) {
  
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
   * 1. chunk adj matrix allocation by rank [x]
   * 2. launch kernel to convert edge list to adj matrix
   * 3. free edge list from host [x]
   * 4. return offset for use in bfs [x]
   * 5. create graph object to store
   *      - offset
   *      - chunkSize
   *      - numNodes
   *      - numEdges
   *      - device ptr
   *      - visited array
   */

  // calculate the number of nodes for this procs graph & offset
  // within global graph
  int graphSize = numNodes / np;
  int leftover = numNodes % np;
  int offset;
  if (rank < leftover) { 
    graphSize += 1; 
    offset = rank * graphSize;
  }
  else {
    offset = rank * graphSize + leftover;
  }


  // allocate adjacency matrix on device & set to 0's
  void *adjMatrix;
  size_t memSize = sizeof(bool) * graphSize * numNodes;
  CUDA_CALL(cudaMalloc(&adjMatrix, memSize));
  CUDA_CALL(cudaMemset(adjMatrix, 0, memSize));

  // allocate buffer for edge list
  void *devEdgeList;
  memSize = sizeof(long long) * numEdges * 2;
  CUDA_CALL(cudaMalloc(&devEdgeList, memSize));
  CUDA_CALL(cudaMemcpy(devEdgeList, (void *)edgeList, memSize, cudaMemcpyHostToDevice));

  // launch graph contruction

  // cleanup edge list
  CUDA_CALL(cudaFree(devEdgeList));

  // return graphSize & offset for future use
  return pair<int, int>(graphSize, offset);
}
