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




__global__  void MyKernel(double *output, const double value){
   const unsigned int i = (blockDim.x * blockIdx.x + threadIdx.x);
   output[i] = value;
 
}

int main (int argc, char **argv){
   double *d_buffer = NULL;   //Vetor na GPU
   unsigned int size = 0;//atoi(argv[1]);;
   unsigned int threads = 1024;//atoi(argv[2]);;
   unsigned int blocks = 0;
  
   float elapsedTime;
   cudaEvent_t start; 
   cudaEvent_t stop;
   cout << "Acesso a memoria" << endl;
   //Reset no device
   CHECK_ERROR(cudaDeviceReset());
   
   //Criando os eventos: start e stop
   CHECK_ERROR(cudaEventCreate(&start));
   CHECK_ERROR(cudaEventCreate(&stop));
   
   
  
   
      
   for (unsigned int i = 1024; i < 4194304; i*=2){ //2^10 a 2^22
       //Alocando memória na GPU:
      size = i;
      blocks = size / threads;
      
      CHECK_ERROR(cudaMalloc((void**) &d_buffer, size * sizeof(double)));
      
      CHECK_ERROR(cudaEventRecord(start, 0));
      MyKernel<<<blocks, threads>>> (d_buffer, 42.0f);
      CHECK_ERROR(cudaDeviceSynchronize());
      
      //Registrando o fim de um evento
      CHECK_ERROR(cudaEventRecord(stop, 0));
      CHECK_ERROR(cudaEventSynchronize(stop));
      CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
      
      //Liberando memória GPU 
      CHECK_ERROR(cudaFree(d_buffer));
      
      cout << "Tempo: " << elapsedTime << " (ms)" << endl;
      cerr << i << " " << elapsedTime << endl;
   
   }
     
  
     //Destruindo os eventos: start e stop
   CHECK_ERROR(cudaEventDestroy(start));
   CHECK_ERROR(cudaEventDestroy(stop));
   
   //Liberando memória da CPU
   cout << "FIM" << endl << endl; 
  
   return EXIT_SUCCESS;
}


