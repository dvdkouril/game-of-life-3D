// write your code into this file
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 4
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
  This should be actually named reduceHorizontally but I don't have enough time to change it ATM
 */
__global__ void reduceVertically(int *input, int *vertReduced, int n) {
  int i = blockIdx.x * (blockDim.x) + threadIdx.x; // global x coord
  int j = blockIdx.y * (blockDim.y) + threadIdx.y; // global y coord
  int k = blockIdx.z * (blockDim.z) + threadIdx.z; // global z coord

  int cellId = linearize(i, j, k, n);
  int cellRightId = linearize(i + 1, j, k, n);
  int cellLeftId = linearize(i - 1, j, k, n);

  int cellLeft = (i - 1 < 0) ? 0 : input[cellLeftId];
  int cellRight = (i + 1 > n - 1) ? 0 : input[cellRightId];
  //if ((i - 1 >= 0) && (i + 1 <= n - 1)) {
    int sum = input[cellId] + cellLeft + cellRight;
    if (input[cellId] == 1) sum *= -1;
    vertReduced[cellId] = sum;
    //}
}

__global__ void reduceHorizontally(int *vertReduced, int *horReduced, int n) {
  int i = blockIdx.x * (blockDim.x) + threadIdx.x; // global x coord
  int j = blockIdx.y * (blockDim.y) + threadIdx.y; // global y coord
  int k = blockIdx.z * (blockDim.z) + threadIdx.z; // global z coord

  int cellId = linearize(i, j, k, n);
  int cellUpId = linearize(i, j - 1, k, n);
  int cellDownId = linearize(i, j + 1, k, n);

  int cellUp = (j - 1 < 0) ? 0 : vertReduced[cellUpId];
  int cellDown = (j + 1 > n - 1) ? 0 : vertReduced[cellDownId];
  int cell = vertReduced[cellId];
  //if ((j - 1 >= 0) && (j + 1) <= n - 1) {
  int sum = abs(cell) + abs(cellUp) + abs(cellDown);
  if (cell < 0) {
    sum *= -1;
  }
    //if (vertReduced[cellId] < 0) // if current cell is negative
      //sum -= 1; // don't add current cell
    horReduced[cellId] = sum;
    //}
}

__global__ void reduceZically(int *horReduced, int *neighbours, int n) {
  int i = blockIdx.x * (blockDim.x) + threadIdx.x; // global x coord
  int j = blockIdx.y * (blockDim.y) + threadIdx.y; // global y coord
  int k = blockIdx.z * (blockDim.z) + threadIdx.z; // global z coord

  int cellId = linearize(i, j, k, n);
  int cellForthId = linearize(i, j, k + 1, n);
  int cellBackId = linearize(i, j, k - 1, n);

  int cellBack = (k - 1 < 0) ? 0 : horReduced[cellBackId];
  int cellForth = (k + 1 > n - 1) ? 0 : horReduced[cellForthId];
  int cell = horReduced[cellId];
  int sum = abs(cell) + abs(cellBack) + abs(cellForth);
  if (cell < 0) { // if current cell is negative
    sum *= -1;
    sum += 1; // don't add current cell, this is still negative (sign that there was 1)
  }
  neighbours[cellId] = sum;
    //}
}


/*
  int *  neighbours         grid with neighbours computed for each cell
  int *  cellsOut           ouput cells grid - result of one iteration
  int    n                  grid x/y/z dimension
 */
__global__ void solveIteration(int *neighbours, int *cellsOut, int n) {
  
  int i = blockIdx.x * (blockDim.x) + threadIdx.x; // global x coord (IN MEMORY, NOT BLOCK!!!)
  int j = blockIdx.y * (blockDim.y) + threadIdx.y; // global y coord
  int k = blockIdx.z * (blockDim.z) + threadIdx.z; // global z coord
  
  int cellId = linearize(i, j, k, n);
  int alive = neighbours[cellId];
  int aliveAbs = abs(alive);

  if (aliveAbs < 4 || aliveAbs > 5) {
    cellsOut[cellId] = 0;
  } else if (aliveAbs == 5) {
    cellsOut[cellId] = 1;
  } else {
    if (alive < 0)
      cellsOut[cellId] = 1;
    else
      cellsOut[cellId] = 0;
    //cellsOut[cellId] = [linearize(tx, ty, tz, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z)];
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
  int *cellsTemp = NULL;
  // memory allocated on the graphics card (can't be accessed in this function!!!)
  if (cudaMalloc((void**)&cellsTemp, n*n*n*sizeof(cellsTemp[0])) != cudaSuccess) {
    printf("Device memory allocation error\n");
  }
  
  for (int i = 0; i < iters; i++) {
    // grid and block dimensions setup
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    int blocksNumX = (int)ceil(n / (BLOCK_SIZE_X));
    int blocksNumY = (int)ceil(n / (BLOCK_SIZE_Y));
    int blocksNumZ = (int)ceil(n / (BLOCK_SIZE_Z));
    dim3 dimGrid(blocksNumX, blocksNumY, blocksNumZ);

    // TODO call kernel that reduces grid horizontally
    reduceVertically<<<dimGrid, dimBlock>>>(*dCells, cellsTemp, n);
    // TODO call kernel that reduces reduced grid vertically
    reduceHorizontally<<<dimGrid, dimBlock>>>(cellsTemp, *dCells, n);

    reduceZically<<<dimGrid, dimBlock>>>(*dCells, cellsTemp, n);
    
    // kernel invocation
    // cellsTemp = grid with neighbours computed for each cell
    // dCells = grid for storing result
    solveIteration<<<dimGrid, dimBlock>>>(cellsTemp, *dCells, n);

    // kernel invocation error checking
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    // TODOOOOOOOOOOOOO I don't think I need to do this - think about it later!!!!
    // swap grids
    //int *tmp = *dCells;
    //*dCells = cellsNextIter; // setting newly computed iteration to as the result
    //cellsNextIter = tmp; // unnecessary

  }
  	
}

