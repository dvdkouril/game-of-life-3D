// write your code into this file
#define BLOCK_SIZE 16

/*
  int ** cells         input cells grid
  int *  cellsOut      ouput cells grid - result of one iteration
  int    n             grid x/y/z dimension
 */
__global__ void solveIteration(int **cells, int *cellsOut, int n) {
  
  //int i = threadIdx.x;
  //int j = threadIdx.y;
  //int k = threadIdx.z;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  int alive = 0; // number of alive neighbours
  for (int ii = max(i-1, 0); ii <= min(i+1, n-1); ii++)
    for (int jj = max(j-1, 0); jj <= min(j+1, n-1); jj++)
      for (int kk = max(k-1, 0); kk <= min(k+1, n-1); kk++)
	alive += (*cells)[ii*n*n + jj*n + kk];
  alive -= (*cells)[i*n*n + j*n + k];

  if (alive < 4 || alive > 5)
    cellsOut[i*n*n + j*n + k] = 0;
  else if (alive == 5)
    cellsOut[i*n*n + j*n +k] = 1;
  else
    cellsOut[i*n*n + j*n + k] = (*cells)[i*n*n + j*n + k];
  
}

/* 
   int** dCells       input/output parameter (as far as I understand)
   int   n            grid x/y/z dimension
   int   iters        how many iteration are meant to be simulated
 */
void solveGPU(int **dCells, int n, int iters){
  // TODO alocate array for computing next iteration
  int *cellsNextIter = NULL;
  cudaMalloc((void**)&cellsNextIter, n*n*n*sizeof(cellsNextIter[0])); // TODO error checking
  
  for (int i = 0; i < iters; i++) {
    // grid and block dimensions setup
    dim3 dimBlock(n, n, n);
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n, n);
    // kernel invocation
    solveIteration<<<1, dimBlock>>>(dCells, cellsNextIter, n);

    // TODO swap grids
    int *tmp = *dCells;
    *dCells = cellsNextIter;
    cellsNextIter = tmp; // unnecessary
  }

  // TODO free allocated memory
	
}

