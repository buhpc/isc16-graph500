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

  // calculate num blocks & num threads per block based on edges.size()
  int threadsPerBlock = 1024;
  int numBlocks = ceil(float(edges.size()) / threadsPerBlock);
  
  // start timer
  PROFILER_START_EVENT("Construct Graph");

  // launch kernel
  buildGraph(threadsPerBlock, 
  	     numBlocks, 
             graph.deviceAdjMatrix(), 
             graph.numGlobalNodes(), 
             devEdgeList,
             edges.size(),
             graph.nodeOffset(),
             graph.numLocalNodes(),
             graph.rank());

  // stop timer
  PROFILER_STOP_EVENT("Construct Graph");

  // cleanup edge list
  CUDA_CALL(cudaFree(devEdgeList));
}
