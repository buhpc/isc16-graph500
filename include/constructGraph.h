/**
 * filename: constructGraph.h
 * contents: this file contains the kernel 1 routine to construct the graph
 * from only the list of edges and the number of edges
 */

#ifndef CONSTRUCT_GRAPH_H
#define CONSTRUCT_GRAPH_H

#include <cuda_runtime_api.h>
#include <utility>
#include <cmath>

#include "edgeList.h"
#include "buildAdjMatrix.h"

using namespace std;

/**
 * constructs the graph on GPU from the CPU edge list
 * 
 * return:
 *   - pair where pair.first = num nodes in this ranks graph
 * and pair.second = offset within the global graph
 */
pair<int,int> constructGraph(EdgeList &edges, int rank, int np);

#endif // CONSTRUCT_GRAPH_H
