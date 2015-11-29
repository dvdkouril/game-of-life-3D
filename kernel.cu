// write your code into this file
#define BLOCK_SIZE 8
//#define BLOCK_SIZE 8 // debug

/*
  This needs to be uniform for all memory accesses!
  WELL FUCK...
 */
__device__ int linearize(int x, int y, int z, int n) {
  //return x*n*n + y*n + z;
  // or
  return z*n*n + y*n + x;
}

/*
  int ** cells         input cells grid
  int *  cellsOut      ouput cells grid - result of one iteration
  int    n             grid x/y/z dimension
 */
__global__ void solveIteration(int *cells, int *cellsOut, int n) {
  
  int i = blockIdx.x * (blockDim.x-2) + threadIdx.x - 1; // global x coord (IN MEMORY, NOT BLOCK!!!)
  int j = blockIdx.y * (blockDim.y-2) + threadIdx.y - 1; // global y coord
  int k = blockIdx.z * (blockDim.z-2) + threadIdx.z - 1; // global z coord

  int tx = threadIdx.x;  // block-local x coord
  int ty = threadIdx.y;  // block-local y coord
  int tz = threadIdx.z;  // block-local z coord

  /*if ((i < 0) || (j < 0) || (k < 0)) { // debug
    printf("(%d,%d,%d)\n", i, j, k);
    }*/
  
  //int cellId = i*n*n + j*n + k;
  int cellId = linearize(i, j, k, n);

  // debug
  /*if ((i == 11) && (j == 11) && (k == 0)) {
    printf("block[%d, %d, %d], thread[%d, %d, %d] => (%d, %d, %d)\n",
	   blockIdx.x, blockIdx.y, blockIdx.z,
	   threadIdx.x, threadIdx.y, threadIdx.z,
	   i, j, k);
	   }*/
  
  // alocating memory with 1 cell border
  //__shared__ int cellsBlock[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int cellsBlock[BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE];

  //printf("(%d, %d, %d)\n", i, j, k);
  // TODO copy stuff from global memory to shared memory
  if ((i <= n - 1) && (j <= n - 1) && (k <= n - 1)) {
    if ((i < 0) || (j < 0) || (k < 0) ||
	(i > BLOCK_SIZE - 1) || (j > BLOCK_SIZE - 1) || (k > BLOCK_SIZE - 1)) {
      cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE)] = 0;
    } else {
      cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE)] = cells[cellId];
    }
  }
  
  __syncthreads();
  
  // TODO use stuff from shared memory when computing alive neighbours
  if ((i >= 0) && (j >= 0) && (k >= 0)&& (i <= n - 1) && (j <= n - 1) && (k <= n - 1)) {
    if ((tx > 0) && (tx < BLOCK_SIZE - 1) &&
	(ty > 0) && (ty < BLOCK_SIZE - 1) &&
	(tz > 0) && (tz < BLOCK_SIZE - 1)) {
      int alive = 0;
      for (int ii = max(tx - 1, 0); ii <= min(tx + 1, BLOCK_SIZE - 1); ii++) {
	for (int jj = max(ty - 1, 0); jj <= min(ty + 1, BLOCK_SIZE - 1); jj++) {
	  for (int kk = max(tz - 1, 0); kk <= min(tz + 1, BLOCK_SIZE - 1); kk++) {
	    alive += cellsBlock[linearize(ii, jj, kk, BLOCK_SIZE)];
	  }
	}
      }
      alive -= cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE)];

      if (alive < 4 || alive > 5) {
	//cellsOut[cellId] = alive;
	cellsOut[cellId] = 0;
      } else if (alive == 5) {
	cellsOut[cellId] = 1;
	//cellsOut[cellId] = alive;
      } else {
	cellsOut[cellId] = cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE)];
	//cellsOut[cellId] = alive;
	//cellsOut[i*n*n + j*n + k] = cells[i*n*n + j*n + k];
      }

    }
  }
  
  
}

// debug
void printGrid(int *cells, int n) {
  printf("\n");
  for (int i = 0; i < n; i++) {
  //int i = 0;
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
	printf("%d ", cells[i*n*n + j*n + k]);
      }
      printf("\n");
    }
    printf("\n\n");
  }
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

  /*printf("first slide of input");
  int *cellsToPrint = (int*)malloc(n*n*n*sizeof(int));
  cudaMemcpy(cellsToPrint, cellsNextIter, n*n*n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(cellsToPrint, *dCells, n*n*n*sizeof(int), cudaMemcpyDeviceToHost);
  printGrid(cellsToPrint, n);*/
  
  for (int i = 0; i < iters; i++) {
    // grid and block dimensions setup
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    //dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE, n / BLOCK_SIZE);
    int blocksNum = (int)ceil(n / (float)(BLOCK_SIZE - 2));
    dim3 dimGrid(blocksNum, blocksNum, blocksNum);

    // debug
    printf("grid size: %dx%dx%d\n", blocksNum, blocksNum, blocksNum);
    printf("block size: %dx%dx%d\n", BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    
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
  /*printf("first slide of output");
  //int *cellsToPrint = (int*)malloc(n*n*n*sizeof(int));
  cudaMemcpy(cellsToPrint, *dCells, n*n*n*sizeof(int), cudaMemcpyDeviceToHost);
  printGrid(cellsToPrint, n);*/
  
  // TODO free allocated memory
  //cudaFree(*dCells); // the memory that I allocated should end up in dCells
	
}

