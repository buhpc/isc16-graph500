/**
 * filename: init.cpp
 * contents: this file contains the routines to initialize all slave processes
 * with the correct CUDA device ID 
 */

#include "init.h"
#include "util.h"

// parse command line and setup  
void init(const string &configFile, int rank, int np) {
  ifstream fp;
  fp.open(configFile);

  vector<int> deviceIds;

  // get device ids
  string line;
  while(getline(fp, line)) {
    deviceIds.push_back(stoi(line));
  }

  // bind to device
  CUDA_CALL(cudaSetDevice(deviceIds[rank]));
}
