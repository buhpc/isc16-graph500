/**
 * filename: breadthFirstSearch.cpp
 * contents: this file contains the definition of the bfs function
 */

#include "breadthFirstSearch.h"

#include <stdio.h>

/**
 * performs bfs on the input graph from the input key
 */
void breadthFirstSearch(long long key, const Graph &graph, long long *hostParent) {
  // set hostParent to -1's
  memset((void *) hostParent, -1, sizeof(long long) * graph.numGlobalNodes());

  // allocate host and device buffer for visited flags
  int numHostVisited = graph.numGlobalNodes() + 1;
  bool *hostVisited = new bool[numHostVisited]();

  // mark starting node index as visited
  hostVisited[key] = true;

  // allocate parent info on device
  long long *devParent;
  size_t parentSize = sizeof(long long) * graph.numGlobalNodes();
  CUDA_CALL(cudaMalloc((void **)&devParent, parentSize));
  CUDA_CALL(cudaMemset(devParent, -1, parentSize));

  // allocate visited info on device
  bool *devVisited;
  size_t visitedSize = sizeof(bool) * numHostVisited;
  CUDA_CALL(cudaMalloc((void **)&devVisited, visitedSize));
  CUDA_CALL(cudaMemcpy(devVisited,
            hostVisited,
            visitedSize, 
            cudaMemcpyHostToDevice));
  
  // step through bfs until complete
  do {
    //bfsStep();

    // copy visited back from device
    CUDA_CALL(cudaMemcpy(hostVisited, 
              devVisited, 
              visitedSize, 
              cudaMemcpyDeviceToHost));

    // update global visited information
    MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE, 
                              hostVisited,
                              numHostVisited, 
                              MPI::BOOL,
                              MPI::LOR);
  
  } while(hostVisited[numHostVisited - 1]);

  // copy parent info back from device
  CUDA_CALL(cudaMemcpy(hostParent,
                       devParent, 
                       parentSize,
                       cudaMemcpyDeviceToHost));

  // reduce max to get parent info on master
  if (graph.rank() == 0) {
    MPI::COMM_WORLD.Reduce(MPI::IN_PLACE, 
                           hostParent, 
                           graph.numGlobalNodes(),
                           MPI::LONG_LONG,
                           MPI::MAX,
                           0);
  }
  else {
    MPI::COMM_WORLD.Reduce(hostParent,
                           hostParent,
                           graph.numGlobalNodes(),
                           MPI::LONG_LONG,
                           MPI::MAX,
                           0);
  }
}
