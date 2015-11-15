// write your code into this file
#define BLOCK_SIZE 16


__global__ void solveIteration(int **cells, int *cellsOut, int n) {
  
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;

  int alive = 0;
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
    cellsOut[i*n*n + j*n + k] = (*in)[i*n*n + j*n + k];
  
}

/* 
   int** dCells       input/output parameter (as far as I understand)
   int   n            grid x/y/z dimension
   int   iters        how many iteration are meant to be simulated
 */
void solveGPU(int **dCells, int n, int iters){
  // TODO alocate array for computing next iteration
  
  
  for (int i = 0; i < n; n++) {
    // grid and block dimensions setup
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid();
    // kernel invocation
    solveIteration<<<n, dimBlock>>>(dCells, n);

    // TODO swap grids
  }
	
}

