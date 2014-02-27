#include  <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

using namespace std;

/*
int main(int argc, char ** argv) {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               // Ne detecte pas CUDA
                return -1;
            } else if (deviceCount == 1) {
               // Ne supporte pas CUDA
	       cout << "Il y a un seul deviceCount" << endl;
            } else {
              // Afficher le nombre de deviceCount
	      cout << "Le nombre de deviceCount est de :" << deviceCount << "\n" << endl;
            }
        }

        // Afficher le nom de la device
	cout << "Le nom de la device est :" << deviceProp.name << "\n" << endl;
        // Donner le numero de version majeur et mineur
	cout << "Les numéros de version majeur et mineur sont :" << deviceProp.major << "   " << deviceProp.minor << "\n" << endl;
        // Donner la taille de la memoire globale
	cout << "La taille de la mémoire globale est de :" << deviceProp.totalGlobalMem << "\n" << endl;
        // Donner la taille de la memoire constante
	cout << "La taille de la mémoire constante est de :" << deviceProp.totalConstMem << "\n" << endl;
        // Donner la taille de la memoire partagee par bloc
	cout << "La taille de la mémoire partagée est de :" << deviceProp.sharedMemPerBlock << "\n" << endl;
        // Donner le nombre de thread max dans chacune des directions
	cout << "La nombre de threads max par direction est :" << deviceProp.maxThreadsDim[0] << "  " << deviceProp.maxThreadsDim[1] << "   " << deviceProp.maxThreadsDim[2] << "\n" << endl;

	cout << "Nombre max thread par bloc " << deviceProp.maxThreadsPerBlock << "\n" << endl;

        // Donner le taille maximum de la grille pour chaque direction
	cout << "Nombre de block par grille :" << deviceProp.maxGridSize[0] << "  " << deviceProp.maxGridSize[1] << "   " << deviceProp.maxGridSize[2] << "\n" << endl;
        // Donner la taille du warp
	cout << "La taille du warp est de :" << deviceProp.warpSize << "\n" << endl;
    }

	system("pause");
    return 0;
}
*/