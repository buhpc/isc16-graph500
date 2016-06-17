/** 
 * filename: main.cpp
 * contents: this file contains the main routine for graph500
*/

#include "util.h"
#include "init.h"
#include "edgeList.h"

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <string>
#include <cstdlib>

static const string usage = "[USAGE] ./main <config.ini> <scale> <edgefactor>";

int main(int argc, char **argv) {
  MPI::Init(argc, argv);
  MPI::COMM_WORLD.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);
  
  try {
    // get rank and number of processes
    int rank = MPI::COMM_WORLD.Get_rank();
    int np = MPI::COMM_WORLD.Get_size();

    if (argc != 4) { LOG_ERROR(usage); }

    // call init(): parse config, broadcast device IDs sets each slave sets device
    init(argv[1], rank, np);

    // get scale and edge factor
    int scale = atoi(argv[2]);
    int edgeFactor = atoi(argv[3]);
    
    // compute num nodes & edges
    int numNodes = pow(2, scale);
    int numEdges = edgeFactor * numNodes;

    // check for cuda aware MPI
    if (rank == 0) { CheckForCudaAwareMPI(false); }
    
    // generate random edge list on master
    EdgeList edges(numEdges);

    // Broadcast data to all nodes
      
  }
  catch(MPI::Exception e) {
    std::cout << "MPI ERROR(" << e.Get_error_code()   \
              << "):" << e.Get_error_string()         \
              << std::endl;
  }
  catch(Error err) {
    std::cout << err.GetError() << std::endl;
  }
  MPI::Finalize();
}  
