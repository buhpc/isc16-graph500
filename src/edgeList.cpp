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
    edges_[i] = rand() % numNodes;
  }
}

/**
 * generates a random edge to return
 */
int EdgeList::generateEdge(int seed) {
  int startL = 0;
  int startR = seed - 1;

  int endL = 0;
  int endR = seed - 1;

  while (startR - startL != 1 && endR - endL != 1) {

  }

  // Filler number for now.
  return 0;
}
