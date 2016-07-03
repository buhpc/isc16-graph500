#ifndef VALIDATION_H
#define VALIDATION_H

#include "edgeList.h"

typedef long long int VertexId; //< As per Graph500 spec, needs to be over 48 bits. 64 here. Can hold [0 .. 2^63 - 1]. Negative reserved for vertices not in tree.
typedef int RetType; //< Needs to be large enough to hold all error flags. Typedef mostly for quick changes
//const VertexId VERTICIES = 100; // Opens us up to an edge condition where VertexId sized to hold [0 .. VERTICES - 1] but VERTICES pushes over 
const RetType VALIDATION_SUCCESS = 0;
const RetType ERR_HAS_CYCLE = 1;
const RetType ERR_INVALID_LEVEL = 2;
const RetType ERR_INVALID_GRAPH = 4;
const RetType ERR_DOESNT_SPAN = 8;
const RetType ERR_UNREAL_EDGE = 16;

/**
 * Tests an array of parent information for the following validation
 * criteria:
 *
 * 1) The BFS tree is a tree and does not contain cycles
 * 2) Each tree edge connects vertices whose BFS levels differ by exactly one
 * 3) Every edge in the input list has vertices with levels that differ by
 *    at most one or that both are not in the BFS tree
 * 4) The BFS tree spans an entire connected component's vertices
 * 5) A node and its parent are joined by an edge of the original graph
 *
 * @param parent_array An int array VERTICIES long, where each element's value
 *                     is the index of it's parent, therefore a vertex tree.
 * @param root_id The index of the root vertex in the parent_array.
 *
 * @returns VALIDATION_SUCCESS(0) on success, or throws an error.
 *
 * @throws  A string representation of a mix of error flags. ERR_HAS_CYCLE(1) if a cycle
 *          is detected, ERR_INVALID_LEVEL(2) If a tree edge connects two
 *          vertecies that don't differ by one level. ERR_INVALID_GRAPH(4) if 
 *          any nodes in the , ERR_DOESNT_SPAN(8) if the tree doesn't cover 
 *          the entire connected component of the graph, and
 *          ERR_UNREAL_EDGE(16) if a tree edge doesn't correspond to a graph
 *          edge.
 */
RetType validate(VertexId* parent_array, VertexId root_id, EdgeList& edgelist, long long int VERTICIES);
long long int getNumEdges(VertexId* parent_array, long long int VERTICIES);

#endif //VALIDATION_H
