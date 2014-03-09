//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <cuda.h>
//#include <fstream>
//#include <iostream>
//#include <stdio.h>
//#include <ctime>
//
//using namespace std;
//
//__global__ void matrixMultiply(float * A, float * B, float * C,
//	int numARows, int numAColumns,
//	int numBRows, int numBColumns,
//	int numCRows, int numCColumns) {
//		int tx = threadIdx.x + blockIdx.x*blockDim.x;
//		int ty = threadIdx.y + blockIdx.y*blockDim.y;
//		
//		float p = 0;
//
//		if (tx < numBColumns && ty < numARows){
//			for (int k = 0; k < numAColumns; k++){
//				p += A[ty*numAColumns + k]*B[k*numBColumns + tx];
//			}
//			C[ty*numBColumns + tx] = p;
//		}
//}
//
//void calc(char *file)
//{
//	float * hostA;
//	float * hostB;
//	float * hostC;
//	float * deviceA;
//	float * deviceB;
//	float * deviceC;
//	int numARows;
//	int numAColumns;
//	int numBRows;
//	int numBColumns;
//	int numCRows;
//	int numCColumns;
//
//	float * result;
//
//	/// Charger le fichier d'entree
//	char * in0 = new char();
//	strcpy(in0, file);
//	strcat(in0, "/input0.raw");
//	ifstream fin0(in0);
//	fin0 >> numARows >> numAColumns;
//	hostA = (float*)malloc(numARows*numAColumns*sizeof(float));
//	for (int i = 0; i < numARows*numAColumns; i++){
//		fin0 >> hostA[i];
//	}
//	fin0.close();
//	
//	char * in1 = new char();
//	strcpy(in1, file);
//	strcat(in1, "/input1.raw");
//	ifstream fin1(in1);
//	fin1 >> numBRows >> numBColumns;
//	hostB = (float*)malloc(numBRows*numBColumns*sizeof(float));
//	for (int i = 0; i < numBRows*numBColumns; i++)
//		fin1 >> hostB[i];
//	fin1.close();
//
//	/// Initialiser numCRows et numCColumns
//	numCRows = numARows;
//	numCColumns = numBColumns;
//	/// Allouer hostC
//	hostC = (float*)malloc(numCRows*numCColumns*sizeof(float));
//
//	/// Afficher les informations sur la matrice
//	/// Allouer la memoire sur GPU
//	cudaMalloc((float**)&deviceA, numARows*numAColumns*sizeof(float));
//	cudaMalloc((float**)&deviceB, numBRows*numBColumns*sizeof(float));
//	cudaMalloc((float**)&deviceC, numCRows*numCColumns*sizeof(float));
//
//	/// Copier la memoire sur le GPU
//	cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);
//
//	/// Initialise la grille et les dimensions de chaque bloc
//	int gridX = ceil((double)numBColumns/16.);
//	int gridY = ceil((double)numARows/16.);
//	dim3 dimGrid(gridX, gridY, 1);
//	dim3 dimBlock(16 , 16, 1);
//
//	/// Execute le kernel
//	matrixMultiply<<<dimGrid , dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
//
//	cudaThreadSynchronize();
//
//	/// Charge le resultat en memoire CPU
//	cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);
//
//	//TEST
//	char * out = new char();
//	strcpy(out, file);
//	strcat(out, "/output.raw");
//	ifstream fout(out) ;
//	fout >> numCRows >> numCColumns;
//	result = (float*)malloc(numCRows*numCColumns*sizeof(float));
//	for (int i = 0; i < numCRows*numCColumns; i++)
//		fout >> result[i];
//	fout.close();
//
//	for (int i = 0; i < numCRows*numCColumns; i++){
//		printf("%f \n", fabs(result[i]-hostC[i]));
//	}
//	/// Libere la memoire
//	free(hostA);
//	free(hostB);
//	free(hostC);
//	cudaFree(deviceA);
//	cudaFree(deviceB);
//	cudaFree(deviceC);
//	free(result);
//
//	printf("\n%d %d\n%d %d\n%d %d\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
//	printf("%d %d\n", gridX, gridY);
//}
//
//int main()
//{
//	clock_t tbegin, tend;
//	tbegin = clock();
//
//	calc("ex2_data2/0");
//	printf("0\n");
//
//	calc("ex2_data2/1");
//	printf("1\n");
//
//	calc("ex2_data2/2");
//	printf("2\n");
//
//	calc("ex2_data2/3");
//	printf("3\n");
//
//	calc("ex2_data2/4");
//	printf("4\n");
//
//	calc("ex2_data2/5");
//	printf("5\n");
//	
//	calc("ex2_data2/6");
//	printf("6\n");
//	
//	calc("ex2_data2/7");
//	printf("7\n");
//	
//	calc("ex2_data2/8");
//	printf("8\n");
//	
//	calc("ex2_data2/9");
//	printf("9\n");
//	tend = clock();
//
//	printf("%f\n", (float)(tend-tbegin)/CLOCKS_PER_SEC);
//	system("pause");
//	return 0;
//}
