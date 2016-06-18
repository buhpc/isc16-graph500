/**
 * filename: constructGraph.cpp
 * contents: this file contains the definition of the kernel 1 routine to
 * construct the graph
 */

#include "constructGraph.h"

/**
 * constructs the graph on GPU from the CPU edge list
 */
pair<int, int> constructGraph(EdgeList &edges, int rank, int np) {
  
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
  
  // allocate adjacency matrix on device & set to 0'se
  // note: has to be int to use atomic ops
  int *adjMatrix;
  size_t memSize = sizeof(int) * graphSize * numNodes;
  CUDA_SAFE_CALL(cudaMalloc((void**)&adjMatrix, memSize));
  CUDA_SAFE_CALL(cudaMemset(adjMatrix, 0, memSize));

  // allocate buffer for edge list
  long long *devEdgeList;
  memSize = sizeof(long long) * numEdges * 2;
  CUDA_SAFE_CALL(cudaMalloc((void**)&devEdgeList, memSize));
  CUDA_SAFE_CALL(cudaMemcpy(devEdgeList, (void *)edgeList, memSize, cudaMemcpyHostToDevice));

  // launch graph contruction
  std::cout << endl << "rank = " << rank << std::endl;
  std::cout << "offset = " << offset << std::endl;
  std::cout << "graphSize = " << graphSize << std::endl;
  std::cout << "numNodes = " << numNodes << std::endl;
  std::cout << "numEdges = " << numEdges << std::endl;
  
  // calculate num blocks & num threads per block based on numEdges

  // cleanup edge list
  CUDA_SAFE_CALL(cudaFree(devEdgeList));

  // return graphSize & offset for future use
  return pair<int, int>(graphSize, offset);
}
