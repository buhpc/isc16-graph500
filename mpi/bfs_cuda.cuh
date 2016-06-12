#ifndef BFS_CUDA_H
#define BFS_CUDA_H

/**
 * Data structure for a node of the graph.
 */
struct Node {
	int start;
	int no_of_edges;
};

/**
 * Assertion to check for errors
 */
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

/**
 * Function that performs the breadth first search.
 */
void bfs(Node* graph_nodes, int* graph_edge, int vertex, bool* visited);

#endif