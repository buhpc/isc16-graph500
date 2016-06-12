/**
 * Implementing Breadth first search on CUDA using algorithm given in HiPC'07
 * paper "Accelerating Large Graph Algorithms on the GPU using CUDA"
 *
 * Copyright (c) 2008 
 * International Institute of Information Technology - Hyderabad. 
 * All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for educational purpose is hereby granted without fee, 
 * provided that the above copyright notice and this permission notice 
 * appear in all copies of this software and that you do not sell the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, 
 * IMPLIED OR OTHERWISE.
 *
 * Created by Pawan Harish.
 *
 * Modified by Boston Green Team.
 */

 __global__ void Kernel(Node* g_graph, int *g_edge, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited) {
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