/**
 * filename: graph.h
 * contents: this files contains the declaration of the graph class that is used to store
 * information about the graph
 */

#ifndef GRAPH_H
#define GRAPH_H

/**
 * stores graph information and data
 */
class Graph {

 public:
  
  // store all member of the graph
  Graph(int numGlobalNodes, int numGlobalEdges, int nodeOffset,
        int numLocalNodes, int *deviceAdjMatrix, int *visitedNodes, 
        int rank, int np) :
        numGlobalNodes_(numGlobalNodes), numGlobalEdges_(numGlobalEdges), 
        nodeOffset_(nodeOffset), numLocalNodes_(numLocalNodes),
        deviceAdjMatrix_(deviceAdjMatrix), visitedNodes_(visitedNodes),
        rank_(rank), np_(np){}
        
  /**
   * getters for all members
   */
  int numGlobalNodes() const { return numGlobalNodes_; }
  int numGlobalEdges() const { return numGlobalEdges_; }
  int nodeOffset() const { return nodeOffset_; }
  int numLocalNodes() const { return numLocalNodes_; }
  int getRank() const { return rank_; }
  int getNp() const { return np_; }
  int* deviceAdjMatrix() const { return deviceAdjMatrix_; }
  int* visitedNodes() const { return visitedNodes_; }
  
  /**
   * setters for offset & numLocalNodes
   */ 
  void setNodeOffset(int offset) { nodeOffset_ = offset; }
  void setNumLocalNodes(int numNodes) { numLocalNodes_ = numNodes; }

  /**
   * calculates localNumNodes based on rank, np, and 
   */
  
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
