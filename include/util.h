/**
 * filename: util.h
 * contents: this file contains the definition of the error class that is used
 * when throwing exception, as well as basic wrappers for cuda error checking, 
 * error logging, and checking for cuda support in MPI
**/

#ifndef UTIL_H
#define UTIL_H

#include <cstdlib>
#include <string>
#include <cuda_runtime_api.h>
#include <mpi.h>
#include <cstdint>

using namespace std;

/** 
 * stores error string to indicate what caused error
 */
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
#define LOG_ERROR(msg) {int rank = MPI::COMM_WORLD.Get_rank();     \
    Error err(string(__FILE__) + "(" + to_string(__LINE__) +  \
              "): " + "rank(" + to_string(rank) + ") "+ msg); \
    throw err;}

// printing helper
#define LOG_INFO(msg) {cout << __FILE__ << "(" << to_string(__LINE__) \
                            << "): " << msg << endl;}

// wrapper for cuda calls
#define CUDA_CALL(func) {GpuAssert(func, __FILE__, __LINE__);}

// check for cudaSuccess
inline void GpuAssert(cudaError_t code, const string file, int line) {
  if (code != cudaSuccess) {
    std::cout << file << "(" << to_string(line) \
              << "): " << cudaGetErrorString(code) << endl;;
  }
}

// check for cuda aware mpi. throws error if no support
inline void CheckForCudaAwareMPI(bool abort) {
#ifdef MPIX_CUDA_AWARE_SUPPORT
  if ( 1 != MPIX_Query_cuda_support())
    if (abort) {
      LOG_ERROR("No CUDA support in this MPI installation");
    }
    else {
      LOG_INFO("No CUDA support in this MPI installation");
    }
#else
  if (abort) {
    LOG_ERROR("No CUDA support in this MPI installation");
  }
  else {
    LOG_INFO("No CUDA support in this MPI installation");
  }
#endif
}

#endif // UTIL_H