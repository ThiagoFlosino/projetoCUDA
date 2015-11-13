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

void cpu (int *a, int *b, int *c, int size){
   for (int  i = 0; i < size; i++){
         c[i] = a[i] + b[i];
   }
      
}


__global__ void gpu(int *a, int *b, int *c){
   const int i = blockDim.x * blockIdx.x + threadIdx.x;
   c[i] = a[i] + b[i];
}




int main (int argc, char **argv){
   int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, size = 8;
   
   size_t free, total;
   float elapsedTime;
   cudaEvent_t start; 
   cudaEvent_t stop;
   
   cout << "GPU CODE" << endl;
   //Reset no device
   CHECK_ERROR(cudaDeviceReset());

   //Criando os eventos: start e stop
   CHECK_ERROR(cudaEventCreate(&start));
   CHECK_ERROR(cudaEventCreate(&stop));


   //Alocando memória na GPU:
   CHECK_ERROR(cudaMalloc((void**) &d_a, size * sizeof(int)));
   CHECK_ERROR(cudaMalloc((void**) &d_b, size * sizeof(int)));
   CHECK_ERROR(cudaMalloc((void**) &d_c, size * sizeof(int)));
   
   //Alocando memória na CPU
   h_a = new int[size];
   h_b = new int[size];
   h_c = new int[size];
   
   for (int i = 0; i < size; i++){
      h_a[i] = (i+1);
      h_b[i] = (i+1) * 10;
   }
   
   //Registrando o inicio de um evento
   CHECK_ERROR(cudaEventRecord(start, 0));
   
   //Copia CPU --> GPU
   CHECK_ERROR(cudaMemcpy(d_a, h_a, size * sizeof(int),  cudaMemcpyHostToDevice));
   CHECK_ERROR(cudaMemcpy(d_b, h_b, size * sizeof(int),  cudaMemcpyHostToDevice));

   gpu<<<1, size>>> (d_a, d_b, d_c);
   CHECK_ERROR(cudaDeviceSynchronize());
   
   //Copia dado da GPU para CPU (cudaMemcpyHostToDevice)
   CHECK_ERROR(cudaMemcpy(h_c, d_c, size * sizeof(int),  cudaMemcpyDeviceToHost));
   
   //Registrando o fim de um evento
   CHECK_ERROR(cudaEventRecord(stop, 0));
   CHECK_ERROR(cudaEventSynchronize(stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  
   
   
   for (int i = 0; i < size; i++){
         cout << i << ": " << h_c[i] << endl;
   }
   cout << "Tempo gasto: " << elapsedTime << " ms" << endl;

   

   //Liberando memória GPU 
   CHECK_ERROR(cudaFree(d_a));
   CHECK_ERROR(cudaFree(d_b));
   CHECK_ERROR(cudaFree(d_c));
   
   
   //Destruindo os eventos: start e stop
   CHECK_ERROR(cudaEventDestroy(start));
   CHECK_ERROR(cudaEventDestroy(stop));
   
   
   delete[] h_a;
   delete[] h_b;
   delete[] h_c;
   //Liberando memória da CPU
   cout << "FIM" << endl << endl;   
   return EXIT_SUCCESS;
}


