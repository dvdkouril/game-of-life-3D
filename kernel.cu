// write your code into this file
#define BLOCK_SIZE 16


__global__ void solveIteration(int **cells) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int k = threadIdx.z;
}

void solveGPU(int **dCells, int n, int iters){
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid();
	for (int i = 0; i < n; n++) {
		solveIteration<<<n, dimBlock>>>(dCells);
	}
}

