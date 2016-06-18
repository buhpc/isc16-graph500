/**
 * filename: edgeList.h
 * contents: this file contains the declaration of the routine used to init a
 * random edge list for testing. Kronecker graph generator will be used in 
 * competition
 */

#ifndef EDGE_LIST_H
#define EDGE_LIST_H

#include <cstdlib>
#include <ctime>

// #include "util.h"

using namespace std;

/**
 * struct to hold start vertex connecting end vertex
 */
struct edge {
  long long from;
  long long to;
};

/**
 * generates a struct edge using a start vertex and end vertex
 */
edge generatedEdge(long long from, long long to);

/**
 * class to store list of edges
 */
class EdgeList {
 
 public:

  EdgeList() : size_(0), edges_(nullptr) {}

  /**
   * allocate memory for list of size numEdges
   */
  EdgeList(int numEdges) : size_(numEdges) { edges_ = new edge[numEdges*2](); }

  /**
   * generate random edge list in allocated buffer
   */
  void create(int numNodes, int scale, int seed = 0);
  edge generateRandomEdge(int scale, double A, double B, double C);

  /**
   * clear edge list
   */
  ~EdgeList() { if(edges_ != nullptr) delete[] edges_; }

  /**
   * getters
   */
  int size() const { return this->size_; }

  edge* edges() const { return this->edges_; }

  /**
   * copy constructor and copy assigment operators
   */
  EdgeList& operator=(EdgeList &&edgeList) = delete;

  EdgeList& operator=(const EdgeList &edgeLIst) = delete;

  EdgeList(EdgeList &&edgeList) = delete;

  EdgeList(const EdgeList &edgeList) = delete;
  
 private:
  
  edge* edges_;
  
  int size_;
};

#endif // EDGE_LIST_H
