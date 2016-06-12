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