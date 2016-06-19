/**
 * filename: graph.cpp
 * contents: this files contains the declaration of the graph class that is used
 * to store information about the graph
 */

#include "graph.h"

/**
 * init member variables
 */
Graph::Graph(int numGlobalNodes, int numGlobalEdges, int rank, int np) :
  numGlobalNodes_(numGlobalNodes), numGlobalEdges_(numGlobalEdges),
  nodeOffset_(0), numLocalNodes_(0), 
  deviceAdjMatrix_(nullptr), visitedNodes_(nullptr),
  rank_(rank), np_(np) {}

/** 
 * calculates numLocalNodes & nodeOffset
 */
void Graph::chunkGraph() {
  numLocalNodes_ = numGlobalNodes_ / np_;
  int leftover = numGlobalNodes_ % np_;

  if (rank_ < leftover) {
    numLocalNodes_ += 1;
    nodeOffset_ = rank_ * numLocalNodes_;
  }
  else {
    nodeOffset_ = rank_ * numLocalNodes_ + leftover;
  }
}

/**
 * allocate the graph on device and set to 0
 */
void Graph::allocateDeviceAdjMatrix() {
  size_t memSize = sizeof(bool) * numLocalNodes_ * numGlobalNodes_;
  CUDA_CALL(cudaMalloc((void**)&deviceAdjMatrix_, memSize));
  CUDA_CALL(cudaMemset(deviceAdjMatrix_, 0, memSize));
}

