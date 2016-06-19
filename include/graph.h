/**
 * filename: graph.h
 * contents: this files contains the declaration of the graph class that is used to store
 * information about the graph
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <cuda_runtime_api.h>

#include "util.h"

/**
 * stores graph information and data
 */
class Graph {

 public:
  
  // store all member of the graph
  Graph(int numGlobalNodes, int numGlobalEdges, int rank, int np);

  /**
   * getters for all members
   */
  int numGlobalNodes() const { return numGlobalNodes_; }
  int numGlobalEdges() const { return numGlobalEdges_; }
  int nodeOffset() const { return nodeOffset_; }
  int numLocalNodes() const { return numLocalNodes_; }
  int rank() const { return rank_; }
  int np() const { return np_; }
  int* deviceAdjMatrix() const { return deviceAdjMatrix_; }
  int* visitedNodes() const { return visitedNodes_; }
  
  /**
   * setters
   */ 
  void setNumGlobalNodes(int numNodes) { numGlobalNodes_ = numNodes; }

  /**
   * calculate numLocalNodes & nodeOffset
   */
  void chunkGraph();

  /**
   * create device adjacency matrix
   */
  void allocateDeviceAdjMatrix();

  /**
   * create visited array
   */
  void allocateVisitedArray();

  /**
   * copy constructor and copy assigment operators
   */
  Graph& operator=(Graph &&graph) = delete;
  Graph& operator=(const Graph &graph) = delete;
  Graph(Graph &&graph) = delete;
  Graph(const Graph &graph) = delete;

 private:

  /**
   *global info
   */
  int numGlobalNodes_; // number of nodes in the whole graph (2**scale)
  int numGlobalEdges_; // number of edges in the whole graph (edgefactor*2**scale)
  int rank_; // rank of this proc
  int np_; // number of procs

  /**
   * local info
   */
  int nodeOffset_; // offset of the first node this rank is responsible for
  int numLocalNodes_; // number of nodes this rank is responsible for
  int *deviceAdjMatrix_; // ptr to adjacency matrix in device memory
  int *visitedNodes_; // array of length globalNumNodes (on device)
  
};

#endif // GRAPH_H
