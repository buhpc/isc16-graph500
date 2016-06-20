/**
 * filename: bfsStep.cpp
 * contents: this file contain the definition of the bfs kernel and the 
 * cpp wrapper
 */

#include "bfsStep.h"

#include <stdio.h>

/**
 * wrapper for cuda kernel
 */
void bfsStep(int threadsPerBlock,
             int numBlocks,
             bool *adjMatrix,
             int numNodes,
             int offset,
             int numLocalNodes,
             bool *visited,
             bool *done,
             long long *parentInfo, 
             int rank,
             int key) {
// launch kernel
bfsStepKernel<<<numBlocks, threadsPerBlock>>>(adjMatrix,
                                              numNodes,
                                              offset,
                                              numLocalNodes,
                                              visited,
                                              done,
                                              parentInfo,
                                              rank,
                                              key);
}

/**
 * perform a single bfs step on the graph
 */
__global__ void bfsStepKernel(bool *adjMatrix,
                              int numNodes,
                              int offset,
                              int numLocalNodes,
                              bool *visited,
                              bool *done,
                              long long *parentInfo,
                              int rank,
                              int key) {
  // local vertex id for this graph
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int globalIndex = index + offset;
  
  // if within bounds for this graph chunk
  if (index < numLocalNodes) {
    
    // if my node is visited but not completed
    if (visited[globalIndex] && !done[index]) {
      
      // visited all my children
      for (int nodeId = 0; nodeId < numNodes; ++nodeId) {
        
        // if there is an edge && child hasn't been visited
        if (adjMatrix[index*numNodes + nodeId] && !visited[nodeId]) {
          // visit this node
          visited[nodeId] = true;
          
          // mark me as it's parent
          parentInfo[nodeId] = globalIndex;
        }
      }

      // mark this node as done
      done[index] = true;

      // indicate work was done
      visited[numNodes] = true;
    }
  }
}

