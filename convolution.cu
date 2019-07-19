#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 4
#define TPB 4
#define INPUT_SIZE 12
#define MAX_MASK_WIDTH 5
__constant__ float M[MAX_MASK_WIDTH];

__global__ void convolution_shared_memory(float *N, float *P){
	
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	__shared__ float N_ds[TILE_SIZE];

	N_ds[threadIdx.x]=N[i];

	__syncthreads();

	int this_title_start_point = blockIdx.x*blockDim.x;
	int next_tile_start_point = (blockIdx.x+1)*blockDim.x;
	int N_start_point = i-(MAX_MASK_WIDTH/2);
	float Pvalue = 0;


	for(int j =0; j < MAX_MASK_WIDTH; j++){

		int N_index = N_start_point+j;

		if(N_index >=0 && N_index < INPUT_SIZE){
			if((N_index>= this_title_start_point) && (N_index<next_tile_start_point)){
				Pvalue+=N_ds[threadIdx.x+j-(MAX_MASK_WIDTH/2)]*M[j];
			}
			else{
				Pvalue+=N[N_index]*M[j];
			}
		}
	}

	P[i]=Pvalue;	
}

int main(){

	//device input and output
	float *d_N = 0;
	float *d_P = 0;

	cudaMalloc(&d_N,INPUT_SIZE*sizeof(float));
	cudaMalloc(&d_P,INPUT_SIZE*sizeof(float));


	//host input and output
	float *h_N = (float*)malloc(INPUT_SIZE*sizeof(float));
	float *h_P = (float*)malloc(INPUT_SIZE*sizeof(float));
	float *h_M = (float*)malloc(MAX_MASK_WIDTH*sizeof(float));

	//initialize input on host
	for(int i=0;i<INPUT_SIZE;++i){
		h_N[i]=(float)i;
	}

	//transfer input to device
	cudaMemcpy(d_N,h_N,INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_P,h_P,INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice);

	//initialize mask on host
	for(int j=0;j<MAX_MASK_WIDTH;++j){
		h_M[j]=(float)j;
	}

	//transfer mask to constant memory
	cudaMemcpyToSymbol(M,h_M,MAX_MASK_WIDTH*sizeof(float));


	//call convolution kernel
	convolution_shared_memory<<<(INPUT_SIZE+TPB-1)/TPB,TPB >>>(d_N,d_P);

	//retrieve result from device
	cudaMemcpy(h_P,d_P,INPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=0; i<INPUT_SIZE;++i){
		printf("%f\n", h_P[i]);
	}


	cudaFree(d_N);
	cudaFree(d_P);
	cudaFree(M);

	free(h_N);
	free(h_P);
	free(h_M);

	printf("Hello world \n");

}