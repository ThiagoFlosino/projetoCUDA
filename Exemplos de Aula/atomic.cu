#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <ctime>
#include <fstream>

#define TAM 512 * 1024 * 1024  // 128MB
//Comando de compilação: nvcc hello.cu -o hello

#define CHECK_ERROR(call) do {                                         \
   if( cudaSuccess != call) {                                          \
      std::cerr << std::endl << "CUDA ERRO: " <<                       \
         cudaGetErrorString(call) <<  " in file: " << __FILE__         \
         << " in line: " << __LINE__ << std::endl;                     \ 
         exit(0);                                                      \
   } } while (0)


using namespace std;

void CPU_Hist(unsigned int *h_hist, unsigned char *h_buffer, int tam){

      for (int i = 0; i < tam; i++)
         h_hist[(int)h_buffer[i]]++;

}

__global__ void  GPU_Hist(unsigned int *d_hist, unsigned char *d_buffer){
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   atomicAdd(&d_hist[d_buffer[i]], 1);

}

int main (int argc, char **argv){
   unsigned char *h_buffer = new unsigned char [TAM];
   unsigned char *d_buffer = NULL;
   
   unsigned int *hist = new unsigned int [256];
   unsigned int *h_hist = new unsigned int [256];   
   unsigned int *d_hist = NULL;
   
 
   cout << endl;
   cout << "Histograma" << endl;

   cout << "Inicializando as variaveis..." << endl;
   bzero(h_hist, 256 * sizeof(unsigned char));
   srand (time(NULL));

   for (int i = 0; i < TAM; i++)
      h_buffer[i] = rand()%256;


   cout << "Calculando histograma na CPU..." << endl;
   CPU_Hist(hist, h_buffer, TAM);

   cout << "Calculando histograma na GPU..." << endl;
   int threads = 1024, blocos = TAM / threads;

   CHECK_ERROR(cudaMalloc((void**) &d_buffer, TAM * sizeof(unsigned char)));
   CHECK_ERROR(cudaMalloc((void**) &d_hist, 256 *  sizeof(unsigned int)));
   CHECK_ERROR(cudaMemset(d_hist, 0, 256 * sizeof(unsigned int)));
   
   //copiando dados CPU -> GPU
   CHECK_ERROR(cudaMemcpy(d_buffer, h_buffer, TAM * sizeof(unsigned char),  cudaMemcpyHostToDevice));
   GPU_Hist<<< blocos, threads >>> (d_hist, d_buffer);
   CHECK_ERROR(cudaDeviceSynchronize());
   

   //copiando dados GPU -> CPU
   CHECK_ERROR(cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int),  cudaMemcpyDeviceToHost));

   
   cout << endl;
   int count  = 0;
   for (int i = 0; i < 256; i++){
      if (h_hist[i] != hist[i]) count++;
   
   }
   cout << "Erros: " << count << " / 256 " << endl;
   cout << endl;
   
   delete[] h_buffer;
   return EXIT_SUCCESS;
}


