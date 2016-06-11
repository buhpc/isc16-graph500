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
 * Implementing Breadth first search on CUDA using algorithm given in HiPC'07
 * paper "Accelerating Large Graph Algorithms on the GPU using CUDA"
 * 
 * The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish.
 *
 * Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
 * All rights reserved.
 */
__global__ void Kernel( Node* g_graph, int *g_edge, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited ) {
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if (tid < VERTICES && g_graph_mask[tid]) {
		g_graph_mask[tid] = false;
		// cuPrintf("Visiting: %d, %d\n", tid, g_graph[tid].start);
		for(int i=g_graph[tid].start; i < (g_graph[tid].no_of_edges+g_graph[tid].start); i++) {
			int id = g_edge[i];
			if (!g_graph_visited[id]) {
				// cuPrintf("Branches: %d,", id);
				g_updating_graph_mask[id] = true;
			}
		}
		// cuPrintf("\n");
	}
}

__global__ void Kernel2( bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over ) {
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if (tid < VERTICES && g_updating_graph_mask[tid]) {
		g_graph_mask[tid] = true;
		g_graph_visited[tid] = true;
		*g_over = true;
		// cuPrintf("Visiting: branches in kernel2, %d,", tid);
		g_updating_graph_mask[tid] = false;
	}
}

/**
 * Function that performs the breadth first search.
 */
void bfs(Node* graph_nodes, int* graph_edge, int vertex, bool* visited);

#endif