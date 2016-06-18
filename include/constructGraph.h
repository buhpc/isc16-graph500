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
#include "graph.h"

using namespace std;

/**
 * constructs the graph on GPU from the CPU edge list
 */
void constructGraph(EdgeList &edges, Graph &graph, int rank, int np);

#endif // CONSTRUCT_GRAPH_H
