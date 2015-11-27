// write your code into this file
#define BLOCK_SIZE 8

/*
  int ** cells         input cells grid
  int *  cellsOut      ouput cells grid - result of one iteration
  int    n             grid x/y/z dimension
 */
__global__ void solveIteration(int *cells, int *cellsOut, int n) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  __shared__ float cellsBlock[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

  // searching the neighbourhood for alive cells
  int alive = 0; // number of alive neighbours
  for (int ii = max(i-1, 0); ii <= min(i+1, n-1); ii++)
    for (int jj = max(j-1, 0); jj <= min(j+1, n-1); jj++)
      for (int kk = max(k-1, 0); kk <= min(k+1, n-1); kk++)
	alive += cells[ii*n*n + jj*n + kk]; // global memory access
  alive -= cells[i*n*n + j*n + k]; // global memory access

  if (alive < 4 || alive > 5) {
    cellsOut[i*n*n + j*n + k] = 0;
  } else if (alive == 5) {
    cellsOut[i*n*n + j*n + k] = 1;
  } else {
    cellsOut[i*n*n + j*n + k] = cells[i*n*n + j*n + k];
  }
  
}

// debug
void printGrid(int *cells, int n) {
  printf("\n");
  //for (int i = 0; i < n; i++) {
  int i = 0;
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
	printf("%d ", cells[i*n*n + j*n + k]);
      }
      printf("\n");
    }
    printf("\n\n");
    //}
}

/* 
   int** dCells       input/output parameter (as far as I understand)
   int   n            grid x/y/z dimension
   int   iters        how many iteration are meant to be simulated
 */
void solveGPU(int **dCells, int n, int iters){
  // alocate array for computing next iteration
  int *cellsNextIter = NULL;
  // memory allocated on the graphics card (can't be accessed in this function!!!)
  if (cudaMalloc((void**)&cellsNextIter, n*n*n*sizeof(cellsNextIter[0])) != cudaSuccess) {
    printf("Device memory allocation error\n");
  }

  // debug
  //printf("first slide of input");
  //int *cellsToPrint = (int*)malloc(n*n*n*sizeof(int));
  //cudaMemcpy(cellsToPrint, cellsNextIter, n*n*n*sizeof(int), cudaMemcpyDeviceToHost);
  //cudaMemcpy(cellsToPrint, *dCells, n*n*n*sizeof(int), cudaMemcpyDeviceToHost);
  //printGrid(cellsToPrint, n);
  
  for (int i = 0; i < iters; i++) {
    // grid and block dimensions setup
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE, n / BLOCK_SIZE);
    
    // kernel invocation
    solveIteration<<<dimGrid, dimBlock>>>(*dCells, cellsNextIter, n);

    // kernel invocation error checking
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    // swap grids
    int *tmp = *dCells;
    *dCells = cellsNextIter; // setting newly computed iteration to as the result
    cellsNextIter = tmp; // unnecessary

  }

  // debug
  //printf("first slide of output");
  //int *cellsToPrint = (int*)malloc(n*n*n*sizeof(int));
  //cudaMemcpy(cellsToPrint, *dCells, n*n*n*sizeof(int), cudaMemcpyDeviceToHost);
  //printGrid(cellsToPrint, n);

  // TODO free allocated memory
  cudaFree(*dCells); // the memory that I allocated should end up in dCells
	
}

