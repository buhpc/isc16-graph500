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
  for (int i = 0; i < numEdges*2; ++i) {
    if (edgeList[i] > numNodes) {
      numNodes = edgeList[i];
    }
  }
  numNodes += 1;

  /**TODO
   * 1. chunk adj matrix allocation by rank [x]
   * 2. launch kernel to convert edge list to adj matrix [x]
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
  CUDA_CALL(cudaMalloc((void**)&adjMatrix, memSize));
  CUDA_CALL(cudaMemset(adjMatrix, 0, memSize));

  // allocate buffer for edge list
  long long *devEdgeList;
  memSize = sizeof(long long) * numEdges * 2;
  CUDA_CALL(cudaMalloc((void**)&devEdgeList, memSize));
  CUDA_CALL(cudaMemcpy(devEdgeList, (void *)edgeList, memSize, cudaMemcpyHostToDevice));

  // print graph chunk info
  std::cout << endl << "rank = " << rank << std::endl;
  std::cout << "offset = " << offset << std::endl;
  std::cout << "graphSize = " << graphSize << std::endl;
  std::cout << "numNodes = " << numNodes << std::endl;
  std::cout << "numEdges = " << numEdges << std::endl;
  
  // calculate num blocks & num threads per block based on numEdges
  int threadsPerBlock = 256;
  int numBlocks = ceil(float(numEdges) / threadsPerBlock);

  if( rank == 1) {
    std::cout << "blocks = " << numBlocks << std::endl;
    std::cout << "threads = " << threadsPerBlock << std::endl;
  }

  // launch kernel
  buildGraph(threadsPerBlock, 
             numBlocks, 
             adjMatrix, 
             numNodes, 
             devEdgeList,
             numEdges,
             offset,
             graphSize,
             rank);
  
  // cleanup edge list
  CUDA_CALL(cudaFree(devEdgeList));

  // DEBUG: copy graph back
  memSize = sizeof(int) * graphSize * numNodes;
  int *hostAdjMatrix = new int[memSize];
  CUDA_CALL(cudaMemcpy(hostAdjMatrix, adjMatrix, memSize, cudaMemcpyDeviceToHost));

  // inspect the graph
  for (int i = 0; i < numEdges; ++i) {
    std::cout << edgeList[i] << "\t";
  }
  std::cout << std::endl;
  for (int i = 0; i < numEdges; ++i) {
    std::cout << edgeList[i + numEdges] << "\t";
  }
  
  std::cout << std::endl;
  std::cout << "printing graph" << std::endl;
  int num = memSize / sizeof(int);
  for (int i = 0; i < num; ++i) {
    std::cout << hostAdjMatrix[i] << " ";
    
    if (!((i+1) % numNodes)) { std::cout << std::endl; }
  }
  
  
  // return graphSize & offset for future use
  return pair<int, int>(graphSize, offset);
}
