/**
 * filename: breadthFirstSearch.h
 * contents: this file contains the declaration of the function that performs
 * the bfs on the distributed graph
 */

#ifndef BREADTH_FIRST_SEARCH_H
#define BREADTH_FIRST_SEARCH_H

#include "graph.h"
#include "util.h"
#include "bfsStep.h"
#include "profiler.h"

#include <cstdlib>
#include <cmath>
using std::ceil;

/**
 * performs breadth first search startin at the input key
 *
 * args:
 *   - key: search key to start the bfs at (node ID)
 *   - graph: graph object to search over
 *   - hostParent: array of len == graph.numGlobalNodes. Stores 
 * results of bfs
 */
void breadthFirstSearch(long long key, const Graph &graph, long long *hostParent);

#endif // BREADTH_FIRST_SEARCH
