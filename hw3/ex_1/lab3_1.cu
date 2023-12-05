
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

  //@@ Insert code below to compute histogram of input using shared memory and atomics
  extern __shared__ unsigned int s_bins[];
  int s_idx = threadIdx.x;


  //initialize shared memory
  if (s_idx < num_bins) { 
    s_bins[s_idx] = 0; 
  }

  //wait for threads to zero out shared memory
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_elements) {
    int bin = input[idx];
    atomicAdd(&s_bins[bin], 1U);
  }

  //wait for threads to finish
  __syncthreads();

  //write to global memory
  if(s_idx < num_bins) { 
    atomicAdd(&bins[s_idx], s_bins[s_idx]); 
  }

}


__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

  //@@ Insert code below to clean up bins that saturate at 127
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_bins) {
    if (bins[idx] > 127U) {
      bins[idx] = 127U;
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
  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

  //cudaMallocManaged(&input, inputLength * sizeof(unsigned int));
  //cudaMallocManaged(&bins, NUM_BINS * sizeof(unsigned int));

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> randomValue(0, NUM_BINS-1);
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = randomValue(gen);
  }

  /*
  for (int i; i < inputLength; i++) {
    hostInput[i] = rand() % NUM_BINS;
  }
  */

 
  //@@ Insert code below to create reference result in CPU
  memset(resultRef, 0, NUM_BINS * sizeof(unsigned int));
  //memset(hostBins, 0, NUM_BINS * sizeof(unsigned int));
  for (int i = 0; i < inputLength; i++) {
    int bin = hostInput[i];
    if (resultRef[bin] < 127) {
      resultRef[bin] += 1;
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  //cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));


  //@@ Initialize the grid and block dimensions here
  int TPB = 256;
  int BLOCKS = (inputLength  + TPB - 1) / TPB;

  // set the size of shared memory
  int smemSize = (TPB)*sizeof(unsigned int);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<BLOCKS, TPB, smemSize>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();


  //@@ Initialize the second grid and block dimensions here
  int TPB_2 = 256;
  int BLOCKS_2 = (inputLength  + TPB_2 - 1) / TPB_2;


  //@@ Launch the second GPU Kernel here
  convert_kernel<<<BLOCKS_2, TPB_2>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < NUM_BINS; ++i) {
    if (fabs(hostBins[i] - resultRef[i]) > 1e-5) {
        fprintf(stderr, "hostBins and resultRef does not match on index %d: %u (GPU) != %u (CPU)\n", i, hostBins[i], resultRef[i]);
    }
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

