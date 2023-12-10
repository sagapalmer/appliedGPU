
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define TPB 1024


__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
  extern __shared__ unsigned int shared_bins[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int j = tid; j < num_bins; j += blockDim.x) {
      shared_bins[j] = 0;
  }
  __syncthreads();

  if (i < num_elements) {
      atomicAdd(&shared_bins[input[i]], 1);
  }
  __syncthreads();

  // global memory
  for (unsigned int j = tid; j < num_bins; j += blockDim.x) {
      atomicAdd(&bins[j], shared_bins[j]);
  }
}


__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  //@@ Insert code below to clean up bins that saturate at 127
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_bins) {
    if (bins[i] > 127) {
      bins[i] = 127;
    }
  }
}


int main(int argc, char **argv) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2) {
    fprintf(stderr, "Usage: [input length]\n");
    return 1;
  }

  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *) malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *) malloc(NUM_BINS * sizeof(unsigned int));


  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand(time(NULL));
  for (int i = 0; i < inputLength; ++i) {
    hostInput[i] = rand() % (NUM_BINS - 1);
  }

  //@@ Insert code below to create reference result in CPU
  resultRef = (unsigned int *) malloc(NUM_BINS * sizeof(unsigned int));
  for (unsigned int i = 0; i < NUM_BINS; ++i) {
    resultRef[i] = 0;
  }
  for (unsigned int i = 0; i < inputLength; ++i) {
      if (resultRef[hostInput[i]] < 127) {
          resultRef[hostInput[i]]++;
      }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **) &deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **) &deviceBins, NUM_BINS * sizeof(unsigned int));


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));


  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  dim3 blockDim(TPB);
  dim3 gridDim((inputLength + blockDim.x - 1) / blockDim.x);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<gridDim, blockDim, NUM_BINS * sizeof(unsigned int)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "histogram kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cuda sync for histogram kernel failed. error code %d\n", cudaStatus);
    return 1;
  }

  //@@ Initialize the second grid and block dimensions here
  dim3 blockDim2(TPB);
  dim3 gridDim2((NUM_BINS + blockDim2.x - 1) / blockDim2.x);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<gridDim2, blockDim2>>>(deviceBins, NUM_BINS);

  cudaError_t cudaStatus2 = cudaGetLastError();
  if (cudaStatus2 != cudaSuccess) {
    fprintf(stderr, "convert kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    return 1;
  }

  cudaStatus2 = cudaDeviceSynchronize();
  if (cudaStatus2 != cudaSuccess) {
    fprintf(stderr, "cuda sync for convert kernel failed. error code %d\n", cudaStatus);
    return 1;
  }

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  bool correct = true;
  for (unsigned int i = 0; i < NUM_BINS; i++) {
    if (hostBins[i] != resultRef[i]) {
      fprintf(stderr, "BIN %u: GPU %u != CPU %u\n", i, hostBins[i], resultRef[i]);
      correct = false;
      break;
    }
  }

  if (correct) {
    printf("GPU results match the reference results!\n");
  } else {
    printf("GPU results do not match reference!\n");
  }

  printf("Reference Histogram Results:\n");

  printf("Bin,Count\n");
  for (unsigned int i = 0; i < NUM_BINS; i++) {
      printf("%u,%u\n", i, resultRef[i]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}
