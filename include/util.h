/* filename: util.h
 * contents: this file contains the definition of the error class that is used
 * when throwing exception, as well as basic wrappers for cuda error checking, 
 * error logging, and checking for cuda support in MPI
 *
 * author: Trevor Gale
 * date: 6.16.16
**/

#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <string>
#include <cuda_runtime_api.h>
#include <mpi.h>

using string = std::string;

/* class Error
 * stores error string to indicate what caused error
**/
class Error
{
public:
	// constructor
	// sets error string
 Error(const string &string) : error(string) {}

	// returns error string
	string GetError() { return this->error; }

private:
	// error string
	string error;
};

// generic exception wrapper
#define LOG_ERROR(msg) {int rank = MPI::COMM_WORLD.Get_rank();    \
    Error err(string(__FILE__) + "(" + std::to_string(__LINE__) + \
              "): " + "rank(" + std::to_string(rank) + ") "+ msg);      \
    throw err;}

#define CUDA_CALL(func) {GpuAssert(func);}

// check for cudaSuccess
inline void GpuAssert(cudaError_t code) {
  if (code != cudaSuccess) {
    LOG_ERROR(cudaGetErrorString(code));
  }
}

// check for cuda aware mpi. throws error if no support
inline void CheckForCudaAwareMPI() {
#ifdef MPIX_CUDA_AWARE_SUPPORT
  if ( 1 != MPIX_QUERY_cuda_support())
    LOG_ERROR("No CUDA support in this MPI installation");
#else
  LOG_ERROR("No CUDA support in this MPI installation");
#endif
}

// only executes on master
#define MASTER(func, rank) {if (rank == 0) {func;}}

#endif // UTIL_H
