/**
 * filename: edgeList.cpp
 * contents: this file contains the definition of the funciton used ot 
 * generate a random edge list for testing
 */

#include "edgeList.h"

/**
 * generate a list of edges of length == numEdges
 */
EdgeList::EdgeList(int numEdges) : size_(numEdges) {
  // seed prng
  srand(time(NULL));

  // allocate enought space for edge list
  edges_ = new int64[numEdges * 2]();

  // set vertices for each edge randomly
  for (int i = 0; i < numEdges*2; ++i) {
    edges_[i] = rand();
  }
}
