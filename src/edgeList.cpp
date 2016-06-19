/**
 * filename: edgeList.cpp
 * contents: this file contains the definition of the funciton used ot 
 * generate a random edge list for testing
 */

#include <math.h>
 
#include "edgeList.h"

/**
 * generates a struct edge using a start vertex and end vertex
 */
edge generatedEdge(long long from, long long to) {
  edge generatedEdge;
  generatedEdge.from = from;
  generatedEdge.to = to;
  return generatedEdge;
}

/**
 * generate s list of edges of length == numEdges
 */
void EdgeList::create(int numNodes, int scale, int seed) {
  // seed prng
  if (!seed) { 
    srand(time(NULL)); 
  }
  else { 
    srand(seed);
  }

  // set initiator probabilities
  double a = 0.57;
  double b = 0.19;
  double c = 0.19;

  // set vertices for each edge randomly
  for (int i = 0; i < size_; ++i) {
    edges_[i] = generateRandomEdge(scale, a, b, c);
  }
}

/**
 * returns a random edge struct based on
 *    scale and initiator probabilities
 */
edge EdgeList::generateRandomEdge(int scale, double A, double B, double C) {
  long long from = 0;
  long long to = 0;

  // loop over each order of bit
  double ab = A + B;
  double cNorm = C / (1 - ab);
  double aNorm = A / ab;

  for(int ib = 0; ib < scale; ib++) {
    // compare with probabilities and set bits of indices
    int coeff = 1 << uint(ib);

    long long from2 = 0;
    long long to2 = 0;

    // permute vertex labels
    double rand1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    if (rand1 > ab) {
      from2 = 1;
    }

    double rand2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    if (rand2 > (cNorm * double(from2)) + (aNorm * fmod(from2 + 1.0, 2.0))) {
      to2 = 1;
    }

    from = from + coeff * from2;
    to = to + coeff * to2;
  }

  return generatedEdge(from, to);
}
