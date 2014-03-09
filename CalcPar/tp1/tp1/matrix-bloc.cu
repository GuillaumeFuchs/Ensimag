#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cmath>

#define TILE_WIDTH 16

using namespace std;

// Calcul C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns,	int numCRows, int numCColumns) {
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx; 

	float Pvalue = 0.;

	int iteration = max(min(numAColumns, numBColumns)/TILE_WIDTH, 1);

	for (int m = 0; m < iteration; ++m){
		if (m * TILE_WIDTH + tx < numAColumns && Row < numARows)
			ds_A[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
		if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
			ds_B[ty][tx] = B[(m*TILE_WIDTH+ty) * numBColumns + Col];

		__syncthreads();

		if (Col < numCColumns && Row < numCRows){
			for (int k = 0; k < TILE_WIDTH; ++k)
				Pvalue += ds_A[ty][k] * ds_B[k][tx];
			__syncthreads();
		}
	}
	if (Col < numCColumns && Row < numCRows)
		C[Row*numCColumns+Col] = Pvalue;
}

void calc(char *file)
{
	float * hostA;
	float * hostB;
	float * hostC;
	float * deviceA;
	float * deviceB;
	float * deviceC;
	int numARows;
	int numAColumns;
	int numBRows;
	int numBColumns;
	int numCRows;
	int numCColumns;

	float * result;

	/// Charger le fichier d'entree
	char * in0 = new char();
	strcpy(in0, file);
	strcat(in0, "/input0.raw");
	ifstream fin0(in0);
	fin0 >> numARows >> numAColumns;
	hostA = (float*)malloc(numARows*numAColumns*sizeof(float));
	for (int i = 0; i < numARows*numAColumns; i++){
		fin0 >> hostA[i];
	}
	fin0.close();

	char * in1 = new char();
	strcpy(in1, file);
	strcat(in1, "/input1.raw");
	ifstream fin1(in1);
	fin1 >> numBRows >> numBColumns;
	hostB = (float*)malloc(numBRows*numBColumns*sizeof(float));
	for (int i = 0; i < numBRows*numBColumns; i++)
		fin1 >> hostB[i];
	fin1.close();

	/// Initialiser numCRows et numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;
	/// Allouer hostC
	hostC = (float*)malloc(numCRows*numCColumns*sizeof(float));

	/// Afficher les informations sur la matrice
	/// Allouer la memoire sur GPU
	cudaMalloc((float**)&deviceA, numARows*numAColumns*sizeof(float));
	cudaMalloc((float**)&deviceB, numBRows*numBColumns*sizeof(float));
	cudaMalloc((float**)&deviceC, numCRows*numCColumns*sizeof(float));

	/// Copier la memoire sur le GPU
	cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);

	/// Initialise la grille et les dimensions de chaque bloc
	int gridX = ceil((double)numBColumns/16.);
	int gridY = ceil((double)numARows/16.);
	dim3 dimGrid(gridX, gridY, 1);
	dim3 dimBlock(16, 16, 1);

	/// Execute le kernel
	matrixMultiply<<<dimGrid , dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

	cudaThreadSynchronize();

	/// Charge le resultat en memoire CPU
	cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);

	//TEST
	char * out = new char();
	strcpy(out, file);
	strcat(out, "/output.raw");
	ifstream fout(out) ;
	fout >> numCRows >> numCColumns;
	result = (float*)malloc(numCRows*numCColumns*sizeof(float));
	for (int i = 0; i < numCRows*numCColumns; i++)
		fout >> result[i];
	fout.close();

	for (int i = 0; i < numCRows*numCColumns; i++){
		printf("%f \n", fabs(result[i]-hostC[i]));
	}
	/// Libere la memoire
	free(hostA);
	free(hostB);
	free(hostC);
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	free(result);

	printf("\n%d %d\n%d %d\n%d %d\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	printf("%d %d\n", gridY, gridX);
}

int main()
{
	clock_t tbegin, tend;

	/*
	tbegin = clock();
	calc("mp2_data/0");
	tend = clock();
	printf("%f\n", (float)(tend-tbegin)/CLOCKS_PER_SEC);
	system("pause");

	tbegin = clock();
	calc("mp2_data/1");
	tend = clock();
	printf("%f\n", (float)(tend-tbegin)/CLOCKS_PER_SEC);
	system("pause");
	*/

	tbegin = clock();
	calc("mp2_data/2");
	tend = clock();
	printf("%f\n", (float)(tend-tbegin)/CLOCKS_PER_SEC);
	system("pause");

	return 0;
}
