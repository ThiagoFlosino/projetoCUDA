#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <ctime>

//Comando de compilação: nvcc hello.cu -o hello
#define CHECK_ERROR(call) do {                                         \
   if( cudaSuccess != call) {                                          \
      std::cerr << std::endl << "CUDA ERRO: " <<                       \
         cudaGetErrorString(call) <<  " in file: " << __FILE__         \
         << " in line: " << __LINE__ << std::endl;                     \ 
         exit(0);                                                      \
   } } while (0)


using namespace std;

__constant__ double c_value;
__device__ double myVariable = 3.14f;

__device__ double inverte(double a){
   double b = 1.0f / a;
   return b;
}
__global__  void MyKernel(double *output){
   const unsigned int i = (blockDim.x * blockIdx.x + threadIdx.x);
   myVariable = inverte(myVariable);
   output[i] = output[i] + c_value + myVariable;
 
}

int main (int argc, char **argv){
   double *d_buffer = NULL;   //Vetor na GPU
   double *h_buffer = NULL;   //Vetor na GPU
   double value = 0.42f;
   unsigned int size = 16;//atoi(argv[1]);;
   unsigned int threads = 1024;//atoi(argv[2]);;
   unsigned int blocks = 1;
  
   cout << "Acesso a memoria" << endl;
   //Reset no device
   CHECK_ERROR(cudaDeviceReset());
   
   h_buffer = new double [size];
   for (int i = 0; i < size; i++)
      h_buffer[i] = 42.0f;
   
   CHECK_ERROR(cudaMalloc((void**) &d_buffer, size * sizeof(double)));
   CHECK_ERROR(cudaMemcpy(d_buffer, h_buffer,  sizeof(double) * size, cudaMemcpyHostToDevice));

   CHECK_ERROR(cudaMemcpyToSymbol (c_value,  &value,   sizeof(double)));
   MyKernel<<<blocks, threads>>> (d_buffer);
   CHECK_ERROR(cudaDeviceSynchronize());
   
   CHECK_ERROR(cudaMemcpy(h_buffer, d_buffer,  sizeof(double) * size, cudaMemcpyDeviceToHost));
   
   for (int i = 0; i < size; i++)
      cout << h_buffer[i] << endl;
   
   CHECK_ERROR(cudaFree(d_buffer));
   delete[] h_buffer;
   
   cout << "FIM" << endl << endl; 
  
   return EXIT_SUCCESS;
}


