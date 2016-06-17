/**
 * filename: init.h
 * contents: this file contains the routines to initialize all slave processes
 * with the correct CUDA device ID 
 */

#ifndef INIT_H
#define INIT_H

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>

using namespace std;

/**
 * parse config file and set cuda device
 */
void init(const std::string &configFile, int rank, int np);

#endif // INIT_H
