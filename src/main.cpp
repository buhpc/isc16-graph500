/** 
 * filename: main.cpp
 * contents: this file contains the main routine for graph500
*/

#include "util.h"
#include "init.h"

#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
  MPI::Init(argc, argv);
  MPI::COMM_WORLD.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);
  
  try {
    if (argc != 2) { LOG_ERROR("Must enter config file for first argument"); }

    // get rank and number of processes
    int rank = MPI::COMM_WORLD.Get_rank();
    int np = MPI::COMM_WORLD.Get_size();

    // call init(): parse config, broadcast device IDs sets each slave sets device
    init(argv[1], rank, np);

    // check for cuda aware MPI
    if (rank == 0) { CheckForCudaAwareMPI(); }
    

      
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
