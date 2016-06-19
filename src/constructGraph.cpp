/**
 * filename: constructGraph.cpp
 * contents: this file contains the definition of the kernel 1 routine to
 * construct the graph
 */

#include "constructGraph.h"

/**
 * constructs the graph on GPU from the CPU edge list
 */
void constructGraph(EdgeList &edges, Graph &graph) {
  
  // find number from vertices from edge list
  // note: this info is in graph object but this kernel isn't provided that info
  int numNodes = 0;
  for (int i = 0; i < edges.size()*2; ++i) {
    if (edges.edges()[i] > numNodes) {
      numNodes = edges.edges()[i];
    }
  }
  numNodes += 1;
  
  // store number of nodes in whole graph
  graph.setNumGlobalNodes(numNodes);

  /**TODO
   * 1. chunk adj matrix allocation by rank [x]
   * 2. launch kernel to convert edge list to adj matrix [x]
   * 3. free edge list from host [x]
   * 4. return offset for use in bfs [x]
   * 5. create graph object to store graph data [x]
   * 6. are atomic operations neccessary? we are always setting to same thing
   */

  // break the graph into chunks for each proc
  graph.chunkGraph();

  // allocate adjacency matrix on device & set to 0's
  // note: has to be int to use atomic ops in cuda
  graph.allocateDeviceAdjMatrix();

  // allocate buffer for edge list
  long long *devEdgeList;
  size_t memSize = sizeof(long long) * edges.size() * 2;
  CUDA_CALL(cudaMalloc((void**)&devEdgeList, memSize));
  CUDA_CALL(cudaMemcpy(devEdgeList, (void *)edges.edges(), memSize, cudaMemcpyHostToDevice));

  // DEBUG: print graph chunk info
  std::cout << endl << "rank = " << graph.rank() << std::endl;
  std::cout << "nodeOffset = " << graph.nodeOffset() << std::endl;
  std::cout << "localNumNodes = " << graph.numLocalNodes() << std::endl;
  std::cout << "numNodes = " << graph.numGlobalNodes() << std::endl;
  std::cout << "numEdges = " << edges.size() << std::endl;
  
  // calculate num blocks & num threads per block based on edges.size()
  int threadsPerBlock = 256;
  int numBlocks = ceil(float(edges.size()) / threadsPerBlock);

  // launch kernel
  /*buildGraph(threadsPerBlock, 
             numBlocks, 
             adjMatrix, 
             numNodes, 
             devEdgeList,
             edges.size(),
             nodeOffset,
             localNumNodes,
             rank);*/
  
  // cleanup edge list
  CUDA_CALL(cudaFree(devEdgeList));

  // DEBUG: copy graph back
  memSize = sizeof(int) * graph.numLocalNodes() * graph.numGlobalNodes();
  int *hostAdjMatrix = new int[memSize];
  CUDA_CALL(cudaMemcpy(hostAdjMatrix, graph.deviceAdjMatrix(), memSize, cudaMemcpyDeviceToHost));

  // inspect the graph
  for (int i = 0; i < edges.size(); ++i) {
    std::cout << edges.edges()[i] << "\t";
  }
  std::cout << std::endl;
  for (int i = 0; i < edges.size(); ++i) {
    std::cout << edges.edges()[i + edges.size()] << "\t";
  }
  
  std::cout << std::endl;
  std::cout << "printing graph" << std::endl;
  int num = memSize / sizeof(int);
  for (int i = 0; i < num; ++i) {
    std::cout << hostAdjMatrix[i] << " ";
    
    if (!((i+1) % numNodes)) { std::cout << std::endl; }
  }
}
