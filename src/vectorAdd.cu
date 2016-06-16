#include "vectorAdd.h"

// VectorAdd: c = a + b
__global__ void VectorAdd(float* a, float* b, float* c, int n)
{
  // get index to add
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}
