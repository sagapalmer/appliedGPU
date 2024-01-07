#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cmath>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    out[idx] = in1[idx] + in2[idx];
  }
}

//@@ Insert code to implement timer start
double getTime() {
  struct timeval t1;
  gettimeofday(&t1, NULL);
  return ((double)t1.tv_sec + (double)t1.tv_usec * 1.e-6);
}

//@@ Insert code to implement timer stop
double getElapsedTime(double startTime) {
  return getTime() - startTime;
}


int main(int argc, char **argv) {
  
  int inputLength;

  //CPU
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  
  //GPU
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  inputLength = atoi(argv[1]); //convert char to int

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
  resultRef = (DataType *)malloc(inputLength * sizeof(DataType));

  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  double startTimeCPU = getTime();
  for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = static_cast<DataType>(rand()) / RAND_MAX;
        hostInput2[i] = static_cast<DataType>(rand()) / RAND_MAX;
        resultRef[i] = hostInput1[i] + hostInput2[i];
  }
  double elapsedTimeCPU = getElapsedTime(startTimeCPU);
  
  double startTotal = getTime();
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  double startDataCopyFromHostToDevice = getTime();
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  double endDataCopyFromHostToDevice = getElapsedTime(startDataCopyFromHostToDevice);
  
  //printf("Data copy from host to device: %.6f seconds\n", endDataCopyFromHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  int TPB = 128;
  int gridSize = (inputLength + TPB - 1) / TPB;


  //@@ Launch the GPU Kernel here
  double startTimeKernel = getTime();
  vecAdd<<<gridSize, TPB>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  double elapsedTimeKernel = getElapsedTime(startTimeKernel);
  
  //printf("The CUDA kernel: %.6f seconds\n", elapsedTimeKernel);


  //@@ Copy the GPU memory back to the CPU here
  double startDataCopyFromDeviceToHost = getTime();
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  double endDataCopyFromDeviceToHost = getElapsedTime(startDataCopyFromDeviceToHost);

  //printf("Data copy from device to host: %.6f seconds\n", endDataCopyFromDeviceToHost);
  
  double elapsedTimeTotal = getElapsedTime(startTotal);
  printf("Total time: %.6f seconds\n", elapsedTimeTotal);

  //@@ Insert code below to compare the output with the reference
  //printf("Elapsed Time for CPU: %.6f seconds\n", elapsedTimeCPU);
  for (int i = 0; i < inputLength; ++i) {
    if (std::abs(hostOutput[i] - resultRef[i]) > 1e-5) {
        fprintf(stderr, "hostOutput and resultRef does not match on index %d: %f (GPU) != %f (CPU)\n", i, hostOutput[i], resultRef[i]);
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  return 0;
}
