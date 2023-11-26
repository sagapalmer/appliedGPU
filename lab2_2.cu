
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cmath>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){

  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  DataType sum = 0.0;

  if (row < numARows && col < numBColumns) {
    for (int k = 0; k < numAColumns; k++) {
      sum += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
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
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);

  numCRows = numARows;
  numCColumns = numBColumns;

  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int col = 0; col < numAColumns; col++) {
    for (int row = 0; row < numARows; row++) {
        hostA[row * numAColumns + col] = static_cast<DataType>(rand()) / RAND_MAX;
    }
  }

  for (int col = 0; col < numBColumns; col++) {
    for (int row = 0; row < numBRows; row++) {
        hostB[row * numBColumns + col] = static_cast<DataType>(rand()) / RAND_MAX;
    }
  }

  double startCPU = getTime();
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      double sum = 0.0;
      for (int k = 0; k < numAColumns; k++) {
        sum += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
      resultRef[i * numCColumns + j] = sum;
    }
  }
  double elapsedTimeCPU = getElapsedTime(startCPU);


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  double startDataCopyFromHostToDevice = getTime();
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  double endDataCopyFromHostToDevice = getElapsedTime(startDataCopyFromHostToDevice);

  printf("Data copy from host to device: %.6f seconds\n", endDataCopyFromHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 threadsPerBlock(16, 16); 
  dim3 blocksPerGrid((numCColumns + threadsPerBlock.x - 1) / threadsPerBlock.x, (numCRows + threadsPerBlock.y - 1) / threadsPerBlock.y);


  //@@ Launch the GPU Kernel here
  double startKernel = getTime();
  gemm<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();
  double elapsedTimeKernel = getElapsedTime(startKernel);

  printf("The CUDA kernel: %.6f seconds\n", elapsedTimeKernel);


  //@@ Copy the GPU memory back to the CPU here
  double startDataCopyFromDeviceToHost = getTime();
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
  double endDataCopyFromDeviceToHost = getElapsedTime(startDataCopyFromDeviceToHost);

  printf("data copy from device to host: %.6f seconds\n", endDataCopyFromDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  printf("Elapsed Time for CPU: %.6f seconds\n", elapsedTimeCPU);
  
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      //if (hostC[i * numCColumns + j] - resultRef[i * numCColumns + j] != 0) {
      double diff = hostC[i * numCColumns + j] - resultRef[i * numCColumns + j];
      if (fabs(diff) > 1e-5) {
        fprintf(stderr, "Result verification failed at element (%d, %d)\n", i, j);
        printf("Diff: %.6f \n", diff);
      }
    }
  }


  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
