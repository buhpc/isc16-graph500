/**
 * filename: edgeList.cpp
 * contents: this file contains the definition of the funciton used ot 
 * generate a random edge list for testing
 */

#include "edgeList.h"

/**
 * generate a list of edges of length == numEdges
 */
void EdgeList::create(int numNodes, int seed) {
  // seed prng
  if (!seed) { 
    srand(time(NULL)); 
  }
  else { 
    srand(seed);
  }
  
  // set vertices for each edge randomly
  for (int i = 0; i < size_; ++i) {
    edges_[i] = rand() & numNodes;
  }
}
