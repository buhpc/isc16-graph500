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

#include "util.h"

using namespace std;

/**
 * class to store list of edges
 */
class EdgeList {
 
 public:

  EdgeList() : size_(0), edges_(nullptr) {}

  /**
   * generate random edge list of length = numEdges
   */
  EdgeList(int numEdges);

  /**
   * clear edge list
   */
  ~EdgeList() { if(edges_ != nullptr) delete[] edges_; }

  /**
   * getters
   */
  int size() const { return this->size_; }

  int64* edges() const { return this->edges_; }

  /**
   * copy constructor and copy assigment operators
   */
  EdgeList& operator=(EdgeList &&edgeList) = delete;

  EdgeList& operator=(const EdgeList &edgeLIst) = delete;

  EdgeList(EdgeList &&edgeList) = delete;

  EdgeList(const EdgeList &edgeList) = delete;
  
 private:
  
  int64 *edges_;
  
  int size_;
};

#endif // EDGE_LIST_H
