/**
 * filename: edgeList.cpp
 * contents: this file contains the definition of the function used to 
 * generate a random edge list for testing
 */

#include <math.h>
 
#include "edgeList.h"

/**
 * generate a list of edges of length == numEdges
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
    generateKroneckerEdge(scale, a, b, c, i);
  }
}

/**
 * returns a random edge struct based on
 *    scale and initiator probabilities
 */
void EdgeList::generateKroneckerEdge(int scale, double A, double B, double C, int i) {
  long long start = 0;
  long long end = 0;

  // loop over each order of bit
  double ab = A + B;
  double cNorm = C / (1 - ab);
  double aNorm = A / ab;

  for(int ib = 0; ib < scale; ib++) {
    // compare with probabilities and set bits of indices
    int coeff = 1 << uint(ib);

    long long start2 = 0;
    long long end2 = 0;

    // permute vertex labels
    double rand1 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    if (rand1 > ab) {
      start2 = 1;
    }

    double rand2 = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    if (rand2 > (cNorm * double(start2)) + (aNorm * fmod(start2 + 1.0, 2.0))) {
      end2 = 1;
    }

    start = start + coeff * start2;
    end = end + coeff * end2;
  }

  // Set a start vertex
  edges_[i] = start;
  // Connect the start vertex to the end vertex
  edges_[i + size_] = end;
}
