// write your code into this file
//#define BLOCK_SIZE 8
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8
//#define BLOCK_SIZE 8 // debug

/*
  This needs to be uniform for all memory accesses!
 */
__device__ int linearize(int x, int y, int z, int n) {
  return z*n*n + y*n + x;
}

__device__ int linearize(int x, int y, int z, int n_x, int n_y, int n_z) {
  return z*n_x*n_y + y*n_x + x;
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
  
  int cellId = linearize(i, j, k, n);
  
  // alocating memory with 1 cell border
  __shared__ int cellsBlock[BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z];

  // TODO copy stuff from global memory to shared memory
  //if ((i <= n - 1) && (j <= n - 1) && (k <= n - 1)) {
    if ((i < 0) || (j < 0) || (k < 0) ||
	(i > n - 1) || (j > n - 1) || (k > n - 1)) {
      //cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE)] = 0;
      cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)] = 0;
    } else {
      //cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE)] = cells[cellId];
      cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)] = cells[cellId];
    }
    //}
  
  __syncthreads();
  
  // TODO use stuff from shared memory when computing alive neighbours
  if ((i >= 0) && (j >= 0) && (k >= 0)&& (i <= n - 1) && (j <= n - 1) && (k <= n - 1)) {
    if ((tx > 0) && (tx < BLOCK_SIZE_X - 1) &&
	(ty > 0) && (ty < BLOCK_SIZE_Y - 1) &&
	(tz > 0) && (tz < BLOCK_SIZE_Z - 1)) {
      int alive = 0;
      for (int ii = max(tx - 1, 0); ii <= min(tx + 1, BLOCK_SIZE_X - 1); ii++) {
	for (int jj = max(ty - 1, 0); jj <= min(ty + 1, BLOCK_SIZE_Y - 1); jj++) {
	  for (int kk = max(tz - 1, 0); kk <= min(tz + 1, BLOCK_SIZE_Z - 1); kk++) {
	    alive += cellsBlock[linearize(ii, jj, kk, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)];
	  }
	}
      }
      alive -= cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)];

      // debug
      /*if ((i == 0) && (j == 0) && (k == 7)) {
	printf("(%d, %d, %d), alive = %d\n", i, j, k, alive);
	}*/
	
      if (alive < 4 || alive > 5) {
	cellsOut[cellId] = 0;
      } else if (alive == 5) {
	cellsOut[cellId] = 1;
      } else {
	cellsOut[cellId] = cellsBlock[linearize(tx, ty, tz, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)];
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
  
  for (int i = 0; i < iters; i++) {
    // grid and block dimensions setup
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    //dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE, n / BLOCK_SIZE);
    int blocksNumX = (int)ceil(n / (float)(BLOCK_SIZE_X - 2));
    int blocksNumY = (int)ceil(n / (float)(BLOCK_SIZE_Y - 2));
    int blocksNumZ = (int)ceil(n / (float)(BLOCK_SIZE_Z - 2));
    dim3 dimGrid(blocksNumX, blocksNumY, blocksNumZ);

    // debug
    //printf("grid size: %dx%dx%d\n", blocksNum, blocksNum, blocksNum);
    //printf("block size: %dx%dx%d\n", BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    
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
  	
}

