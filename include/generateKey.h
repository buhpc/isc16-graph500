/**
 * filename: generateKey.h
 * contents: this file containts the declaration of the routine used to sample
 * random search keys
 */

#ifndef GENERATE_KEY_H
#define GENERATE_KEY_H

#include "graph.h"
#include "edgeList.h"

/**
 * generates a random search key in the graph
 */
long long generateKey(const Graph &graph, const EdgeList &edges);

#endif // GENERATE_KEY_H
