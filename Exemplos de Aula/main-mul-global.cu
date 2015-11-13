#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#define BLOCK_SIZE 32

#define CHECK_ERROR(call) do {                                                    \
   if( cudaSuccess != call) {                                                     \
      fprintf(stderr,"CUDA ERROR:%s in file: %s in line: ", cudaGetErrorString(call),  __FILE__, __LINE__); \
         exit(0);                                                                                 \
   } } while (0)


// Set a matrix element
__device__ __forceinline__  int getPosition(int row, int col)
{
	return row * (gridDim.x * blockDim.x) + col;
}


/*
 * Multiplicação usando memória global
 */

__global__ void multMatrixG(float *C, float *A, float *B)
{
    float Cvalue = 0.0f;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //int width =  (gridDim.x * blockDim.x);
    //int height = (gridDim.y * blockDim.y);


    for (int e = 0; e < (gridDim.x * blockDim.x); ++e){
      Cvalue += A[getPosition(row, e)] * B[getPosition(e, col)];
    }


    C[getPosition(row, col)] = Cvalue;// (float) height;
}



void printMatrix(float *m, float w, float h){
   int i, j;

   printf("\n");

   for (j = 0; j < h; j++){
      for (i = 0; i < w; i++){
         int k = j * w + i;
         printf("%.2f ", m[k]);
      }
      printf("\n");
   }

}

int main (int argc, char **argv){

   float *h_A, *h_B, *h_C;
   int iC, jC;

   int width      = atoi(argv[1]);
   int height     = width;
   int GPU        = 0;

   float   *d_C = NULL,
           *d_A = NULL,
           *d_B = NULL;

   cudaEvent_t start; 
   cudaEvent_t stop;  
   
   float GPUTime = 0.0f,
         MEMTime = 0.0f,
         aux     = 0.0f;

   srand (time(NULL));

   printf("\nMultiplicando matriz - GPU\n");
   printf("Tamanho da matriz: %d x %d \n", width, height);

    h_A = (float*) malloc (width * height * sizeof(float));
    h_B = (float*) malloc (width * height * sizeof(float));
    h_C = (float*) malloc (width * height * sizeof(float));


   for (jC = 0; jC < height; jC++){
      for (iC = 0; iC < width; iC++){
         int kC = jC * width + iC;
         h_A[kC] = (float) (rand() % 65536 + 1) / 65536.0f;

         if (jC == iC)
           h_B[kC] = 1.0f;
         else
            h_B[kC] = 0.0f;

      }
   }

	CHECK_ERROR(cudaSetDevice(GPU));

	//Reset na GPU selecionada
	CHECK_ERROR(cudaDeviceReset());

   CHECK_ERROR(cudaEventCreate(&start));
   CHECK_ERROR(cudaEventCreate(&stop));
   

   //Aloca memória GPU
   CHECK_ERROR(cudaMalloc((void**) &d_A, width * height * sizeof(float)));
   CHECK_ERROR(cudaMalloc((void**) &d_B, width * height * sizeof(float)));
   CHECK_ERROR(cudaMalloc((void**) &d_C, width * height * sizeof(float)));

   //Copiando CPU --> GPU
   CHECK_ERROR(cudaEventRecord(start, 0));
   
   CHECK_ERROR(cudaMemcpy(d_A, h_A, width * height * sizeof(float),  cudaMemcpyHostToDevice));
   CHECK_ERROR(cudaMemcpy(d_B, h_B, width * height * sizeof(float),  cudaMemcpyHostToDevice));

   CHECK_ERROR(cudaEventRecord(stop, 0));
   CHECK_ERROR(cudaEventSynchronize(stop));
   CHECK_ERROR(cudaEventElapsedTime(&aux, start, stop));

   MEMTime = aux;
      

   
      //int numBlocks = 1;
   //int threadsPerBlock = WIDTH*HEIGHT / numBlocks;

	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);


   CHECK_ERROR(cudaEventRecord(start, 0));
	multMatrixG <<<numBlocks, threadsPerBlock >>> (d_C, d_A, d_B);
	CHECK_ERROR(cudaDeviceSynchronize());
   CHECK_ERROR(cudaEventRecord(stop, 0));
   CHECK_ERROR(cudaEventSynchronize(stop));
   CHECK_ERROR(cudaEventElapsedTime(&GPUTime, start, stop));

      
      
   CHECK_ERROR(cudaEventRecord(start, 0));
	CHECK_ERROR(cudaMemcpy(h_C, d_C,  width*height * sizeof(float),  cudaMemcpyDeviceToHost));
   CHECK_ERROR(cudaEventRecord(stop, 0));
   CHECK_ERROR(cudaEventSynchronize(stop));
   CHECK_ERROR(cudaEventElapsedTime(&aux, start, stop));

   MEMTime += aux;
	
	fprintf(stderr, "\n %f %f %f", MEMTime, GPUTime, (MEMTime + GPUTime));
	fprintf(stdout, "\n MEM:\t %f \nGPU:\t %f \n MEM+GPU:\t %f", MEMTime, GPUTime, (MEMTime + GPUTime));
	

   CHECK_ERROR(cudaEventDestroy(start));
   CHECK_ERROR(cudaEventDestroy(stop));
   CHECK_ERROR(cudaFree(d_A));  //Liberando memorias GPU e CPU
   CHECK_ERROR(cudaFree(d_B));  //Liberando memorias GPU e CPU
   CHECK_ERROR(cudaFree(d_C));  //Liberando memorias GPU e CPU

	
   float err = 0.0f;
   for (jC = 0; jC < height; jC++){
        for (iC = 0; iC < width; iC++){
           int kC = jC * width + iC;
           if (fabs(h_A[kC]-h_C[kC]) > 0.000000001f)
        	   err++;
        }
   }
   fprintf(stdout, "\nError: %f\n", (err / (float)(width*height)));
   //validando

   //printMatrix(h_A, width, height);
   //printMatrix(h_C, width, height);

   free(h_A);
   free(h_B);
   free(h_C);

   fprintf(stdout, "FIM\n");

   return EXIT_SUCCESS;
}
