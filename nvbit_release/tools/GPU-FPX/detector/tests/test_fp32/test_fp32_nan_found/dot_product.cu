
#include <stdio.h>

// __device__ void mul(float a, float b, float *res)
// {
//   *res = a * b;
//   // NaN
//   *res = (*res)-(*res) / (*res)-(*res);
// }

__global__ void dot_prod(float *x, float *y, int size) {
  float d;
  for (int i = 0; i < size; ++i) {
    float tmp;
    // mul(x[i], y[i], &tmp);
    tmp = x[i] * y[i];
    printf("first tmp is %f\n", tmp);
    // NaN
    tmp = (tmp - tmp) / (tmp - tmp);
    printf("second tmp is %f\n", tmp);
    d += tmp;
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    printf("dot: %f\n", d);
  }
}
