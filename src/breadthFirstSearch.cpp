/**
 * filename: breadthFirstSearch.cpp
 * contents: this file contains the definition of the bfs function
 */

#include "breadthFirstSearch.h"

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
  
  // indicates if this vertex has already been visited and processed
  bool *devDone;
  CUDA_CALL(cudaMalloc((void**)&devDone, sizeof(bool)*graph.numLocalNodes()));
  CUDA_CALL(cudaMemset(devDone, 0, sizeof(bool)*graph.numLocalNodes()));

  // calculate numThreads & numBlocks: 1 thread for each vertex
  int threadsPerBlock = 1024;
  int numBlocks = ceil(float(graph.numLocalNodes()) / threadsPerBlock);

  // start timer
  PROFILER_START_EVENT("BFS");
  // step through bfs until complete
  do {
    // unset work flag
    CUDA_CALL(cudaMemset(devVisited + graph.numGlobalNodes(), 0, sizeof(bool)));

    // executed step
    bfsStep(threadsPerBlock, 
            numBlocks,
            graph.deviceAdjMatrix(),
            graph.numGlobalNodes(),
            graph.nodeOffset(),
            graph.numLocalNodes(),
            devVisited,
            devDone,
            devParent,
            graph.rank(),
            key);

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
  
//  if (graph.rank() == 1) {
//    hostParent[graph.numGlobalNodes() - 1] = graph.numGlobalNodes() - 2;
//  }
//  // print all parent info
//  for (int i = 0; i < graph.numGlobalNodes(); ++i) {
//    long long parent = hostParent[i];
//    printf("(rank, parent[%d]) = (%d, %d)\n", i, graph.rank(), parent);
//  }
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
  
  PROFILER_STOP_EVENT("BFS");
//  printf("key = %d", key);
//  for (int i = 0; i < graph.numGlobalNodes(); ++i) {
//    long long parent = hostParent[i];
//    if (graph.rank() == 0 && parent != -1)
//      printf("after:(rank, parent[%d]) = (%d, %d)\n", i, graph.rank(), parent);
//  }
} 
