/**
 * filename: buildAdjMatrix.h
 * contents: this files contains the declaration of the cuda kernel
 * used to build the adjacency matrix
 */

#ifndef BUILD_ADJ_MATRIX_H
#define BUILD_ADJ_MATRIX_H

/**
 * wrapper for cuda kernel
 */
void buildGraph(int threadsPerBlock, 
                int numBlocks, 
                int *adjMatrix,
                int numNodes,
                long long *edgeList,
                int numEdges,
                int offset, 
                int graphSize);

/**
 * constructs an adjacency matrix in device memory from the list
 * of edges
 *
 * args: 
 *   - adjMatrix: pointer to adjacency matrix object
 *   - numNodes: the number of nodes int the global graph
 *   - edgeList: pointer to the list of edges to build the graph from
 *   - numEdges: number of edges in the edge list
 *   - offset: the offset of the first first for this rank
 *   - graphSize: the number of vertices this graph has
 */
__global__ void buildAdjMatrix(int *adjMatrix,
                               int numNodes,
                               long long *edgeList,
                               int numEdges,
                               int offset, 
                               int graphSize);

#endif // BUID_ADJ_MATRIX_H
