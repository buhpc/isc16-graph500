/** 
 * filename: main.cpp
 * contents: this file contains the main routine for graph500
*/

#include "util.h"
#include "init.h"
#include "edgeList.h"
#include "constructGraph.h"
#include "graph.h"
#include "generateKey.h"
#include "breadthFirstSearch.h"

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <string>
#include <cstdlib>
#include <utility>

static const string usage = "[USAGE] ./main <config.ini> <scale> <edgefactor>";

int main(int argc, char **argv) {
  MPI::Init(argc, argv);
  MPI::COMM_WORLD.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);
  
  try {
    // get rank and number of processes
    int rank = MPI::COMM_WORLD.Get_rank();
    int np = MPI::COMM_WORLD.Get_size();

    if (argc != 4) { LOG_ERROR(usage); }

    // parse config, sets each slave device
    init(argv[1], rank, np);

    // get scale and edge factor
    int scale = atoi(argv[2]);
    int edgeFactor = atoi(argv[3]);
    
    // compute num nodes & edges
    int numNodes = pow(2, scale);
    int numEdges = edgeFactor * numNodes;

    // check for cuda aware MPI
    // if (rank == 0) { CheckForCudaAwareMPI(false); }

    // allocate buffer 
    EdgeList edges(numEdges);

    // randomly generate on host
    if(rank == 0) { edges.create(numNodes, scale); }

    // Broadcast data to all nodes
    MPI::COMM_WORLD.Bcast(edges.edges(), edges.size()*2, MPI::LONG_LONG, 0);
    
    // construct a graph object for this proc
    Graph graph(0, numEdges, rank, np);

    // kernel 1
    if (rank == 0) { cout << "Constructing graph..." << endl; }
    constructGraph(edges, graph);
    if (rank == 0) { cout << "Done." << endl; }

    /** TODO
     * 1. sample keys [x]
     * 2. BFS [x]
     * 3. integrate knronecker generator
     * 4. integrate validation
     * 5. add timing [x]
     * 6. remove rank from construction kernel & other debugging info
     * 7. tune numThreads & block for our architecture
     * 9. scripts to analyze results from logs
     */
    
    // copy graph back for sampling
    if (rank == 0) { cout << endl << "Running 64 BFSs..." << endl; }
    long long key = 0;
    int numIters = min(64, graph.numGlobalNodes());      
    long long *hostParent = new long long[graph.numGlobalNodes()];
    for (int iter = 0; iter < numIters; ++iter) {
      if (rank == 0) {
        // sample random search key in [0,graph.numGlobalNodes
        key = generateKey(graph, edges);
      }
      // broadcast key to all ranks
      MPI::COMM_WORLD.Bcast(&key, 1, MPI::LONG_LONG, 0);
      
      // begin bread first search
      breadthFirstSearch(key, graph, hostParent); 
    }
    if (rank == 0) { cout << "Done." << endl; }

    // cleanup
    delete[] hostParent;
  }
  catch(MPI::Exception e) {
    cout << "MPI ERROR(" << e.Get_error_code() \
         << "):" << e.Get_error_string()       \
         << endl;
  }
  catch(Error err) {
    cout << err.GetError() << endl;
  }
  MPI::Finalize();
}  
