/**
 * filename: edgeList.cpp
 * contents: this file contains the definition of the funciton used ot 
 * generate a random edge list for testing
 */

#include <math.h>
 
#include "edgeList.h"

edge generatedEdge(long long from, long long to) {
  edge generatedEdge;
  generatedEdge.from = from;
  generatedEdge.to = to;
  return generatedEdge;
}

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

  float a = 0.57;
  float b = 0.19;
  float c = 0.19;

  // set vertices for each edge randomly
  for (int i = 0; i < size_; ++i) {
    edges_[i] = generateRandomEdge(scale, a, b, c);
  }
}

/**
 * generates a random edge to return
 */
edge EdgeList::generateRandomEdge(int scale, float A, float B, float C) {
  long long from = 0;
  long long to = 0;

  float ab = A + B;
  float cNorm = C / (1 - ab);
  float aNorm = A / ab;

  for(int ib = 0; ib < scale; ib++) {
    int coeff = 1 << uint(ib);

    long long from2 = 0;
    long long to2 = 0;

    float rand1 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    if (rand1 > ab) {
      from2 = 1;
    }

    float rand2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    if (rand2 > (cNorm * float(from2)) + (aNorm * fmodf(float(from2 + 1.0), 2.0))) {
      to2 = 1;
    }

    from = from + coeff + from2;
    to = to + coeff * to2;
  }

  return generatedEdge(from, to);
}
