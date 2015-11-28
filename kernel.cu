// write your code into this file
#define BLOCK_SIZE 8

/*
  int ** cells         input cells grid
  int *  cellsOut      ouput cells grid - result of one iteration
  int    n             grid x/y/z dimension
 */
__global__ void solveIteration(int *cells, int *cellsOut, int n) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x; // global x coord
  int j = blockIdx.y * blockDim.y + threadIdx.y; // global y coord
  int k = blockIdx.z * blockDim.z + threadIdx.z; // global z coord

  int tx = threadIdx.x;  // block-local x coord
  int ty = threadIdx.y;  // block-local y coord
  int tz = threadIdx.z;  // block-local z coord
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  // alocating memory with 1 cell border
  __shared__ int cellsBlock[BLOCK_SIZE + 2][BLOCK_SIZE + 2][BLOCK_SIZE + 2];

  // TODO copy stuff from global memory to shared memory
  cellsBlock[tx + 1][ty + 1][tz + 1] = cells[i*n*n + j*n + k];
  // border - this is starting to look pretty fucked up
  if (tx == 0)
    cellsBlock[0][ty][tz] = cells[(i-1)*n*n + j*n + k];
  if (ty == 0)
    cellsBlock[tx][0][tz] = cells[i*n*n + (j-1)*n + k];
  if (tz == 0)
    cellsBlock[tx][ty][0] = cells[i*n*n + j*n + (k - 1)];
  if (tx == BLOCK_SIZE - 1)
    cellsBlock[BLOCK_SIZE + 1][ty][tz] = cells[(i+1)*n*n + j*n + k];
  if (ty == BLOCK_SIZE - 1)
    cellsBlock[tx][BLOCK_SIZE + 1][tz] = cells[i*n*n + (j+1)*n + k];
  if (tz == BLOCK_SIZE - 1)
    cellsBlock[tx][ty][BLOCK_SIZE + 1] = cells[i*n*n + j*n + (k+1)];
  // corners
  if ((tx == 0) && (ty == 0))
    cellsBlock[0][0][tz] = cells[(i-1)*n*n + (j-1)*n + (k+1)];
  
  if ((tx == BLOCK_SIZE - 1) && (ty == 0))
    cellsBlock[BLOCK_SIZE + 1][0][tz] = cells[i*n*n + j*n + (k+1)];
  
  if ((tx == BLOCK_SIZE - 1 ) && (ty == BLOCK_SIZE - 1))
    cellsBlock[0][0][tz] = cells[i*n*n + j*n + (k+1)];
  
  if ((tx == 0) && (ty == BLOCK_SIZE - 1))
    cellsBlock[0][0][tz] = cells[i*n*n + j*n + (k+1)];
  
  __syncthreads();
  
  // TODO use stuff from shared memory when computing alive neighbours
  int alive = 0;
  for (int ii = max(tx - 1, 0); ii <= min(tx + 1, BLOCK_SIZE - 1); ii++) {
    for (int jj = max(ty - 1, 0); jj <= min(ty + 1, BLOCK_SIZE - 1); jj++) {
      for (int kk = max(tz - 1, 0); kk <= min(tz + 1, BLOCK_SIZE - 1); kk++) {
	alive += cellsBlock[ii][jj][kk];
      }
    }
  }
  alive -= cellsBlock[tx][ty][tz];

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
    //dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE, n / BLOCK_SIZE);
    int blocksNum = (int)ceil(n / (float)(BLOCK_SIZE - 2));
    dim3 dimGrid(blocksNum, blocksNum, blocksNum);
    
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

