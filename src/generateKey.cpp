/**
 * filename: generateKey.h
 * contents: this file contains the definition of the routine used ot
 * sample random search keys
 */

#include "generateKey.h"

long long generateKey(const Graph& graph, const EdgeList &edges) {
  bool cont = true;
  int numNodes = graph.numGlobalNodes();
  long long key = 0;

  while (cont) {
    // generaet random key
    key = rand() % numNodes;

    // confirm this node has at least one edge
    for (int i = 0; i < edges.size(); ++i) {
      int vertA = edges.edges()[i];
      int vertB = edges.edges()[i + edges.size()];
      if ((vertA != vertB) && (vertA == key || vertB == key)) {
        cont = false;
        break;
      }
    }
  }

  return key;
}
