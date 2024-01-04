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
  int S_seg;

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
  S_seg = atoi(argv[2]);

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


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));

  cudaStream_t streams[S_seg];
  for (int i = 0; i < S_seg; i++) {
    cudaStreamCreate(&streams[i]);
  }




  //@@ Insert code to below to Copy memory to the GPU here
  // double startDataCopyFromHostToDevice = getTime();
  // //cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  // //cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

  // cudaMemcpyAsync(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  // cudaMemcpyAsync(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  // double endDataCopyFromHostToDevice = getElapsedTime(startDataCopyFromHostToDevice);

  // printf("Data copy from host to device: %.6f seconds\n", endDataCopyFromHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here
  int streamSize = inputLength / S_seg;
  int streamBytes = streamSize * sizeof(DataType);

  int TPB = 128;
  int gridSize = (streamSize + TPB - 1) / TPB;

  //@@ Launch the GPU Kernel here


  double startTimeGPU = getTime();
  for (int i = 0; i < S_seg; i++) {
    int offset = i * streamSize;

    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);

    vecAdd<<<gridSize, TPB, 0, streams[i]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], streamSize);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
  }


  for (int i = 0; i < S_seg; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  double elapsedTimeGPU = getElapsedTime(startTimeGPU);

  for (int i = 0; i < S_seg; i++) {
    cudaStreamDestroy(streams[i]);
  }

  printf("Total time: %.6f seconds\n", elapsedTimeGPU);

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
