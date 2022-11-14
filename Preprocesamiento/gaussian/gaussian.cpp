#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
typedef int filtro;
// Autores: Paul Morales, Jefferson Romero, Leyton Valencia
// Descripción del filtro: Este filtro se encarga de difuminar las imágenes aplicando un efecto que contraste 
// el ruido que produce pequeñas variaciones en la imágen, logrando asi eliminar el ruido excesivo de las imágenes
// y corrigiendo los bordes de la misma.

vector<vector<filtro>> Filtro_Gauss(vector<vector<filtro>> data,int SizeKernel){ //Parámetros de la función Matriz de imágen, Tamaño del kernel
    // Parametro 1 data: Matriz de imágen- Representa los pixeles que posee la imágen dicom
    // Parametro 2 sizeKernel- Representa el tamaño del kernel el cuál sea mas grande se eliminara más el ruido de la imágen.

    // Variable inicializada con el paso del parámetro del tamaño del kernel
    double GKernel[SizeKernel][SizeKernel];
    // El valor del sigma determina si es un filtro en 1-D o 2-D
    double sigma = 2.0;
    // Dentro de la variable s, se guarda la optimización del sigma.
    double r, s = 2.0 * sigma * sigma;
    // Se obtiene y guarda el tamaño de las filas y columnas de la Matriz de imágen
    int filas = (sizeof(data)/sizeof(data[0]));
    int columnas = (sizeof(data[0])/sizeof(data[0][0]));
    // dst es la matriz resultante de aplicar el filtro y se iguala en dimensiones a la Matriz original.
    vector<vector<filtro>> dst = data;
    //vector<vector<filtro>> dst(data.size());
    //copy(data.begin(), data.end(), dst.begin());
    float sum = 0.0;
    double sumary = 0.0;
    #pragma omp parallel num_threads(32){
        #pragma opm parallel for num_threads(16) collapse (16)
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            r = sqrt(x * x + y * y);
            GKernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s);
            sumary += GKernel[x + 2][y + 2];   
            }
    }
    // Normalización del Kernel
    #pragma opm parallel for num_threads(16) collapse (16)
    for (int i = 0; i < 31; ++i){
        for (int j = 0; j < 31; ++j){
            GKernel[i][j] /= sumary;
        }
    }
    #pragma omp barrier
    // APLICACIÓN DEL FILTRO GAUSSIANO
    #pragma opm parallel for num_threads(32) collapse (16)
    for(int y = 0; y < filas; y++){
        for(int x =0 ; x < columnas; x++){
            // Inicializamos las filas y columnas de la matriz resultante con 0.
            dst[y][x] = 0.0;
        }
    }
}
    // APLICAMOS LA OPERACIÓN DE CONVULUCIÓN
    for(int y = 1; y < filas - 1; y++){
        for(int x = 1; x < columnas - 1; x++){
            sum = 0.0;
            for(int k = -1; k <= 1;k++){
                for(int j = -1; j <=1; j++){
                    //En la variable sum guardamos los datos de los pixeles resultantes de aplicar el filtro
                    sum = sum + GKernel[j+1][k+1]*data[y-j][x-k];
                }
            }
            dst[y][x] = sum;
        }
        // Retornamos la matriz resultante dst
        return dst;
        // En caso de que se requiera guardar un archivo csv con los pixeles resultantes, usamos la función dicomOBj.saveData
        // Parametros: vector resultante,path,nombre del archivo, delimitador, true o false si se requiere cabeceras en el csv.
        //dicomObj.saveData(dst,"/home/debian/Documentos/Proyecto_Paralelo_FiltroG_Secuencial/IMAGENES/IMG4/DCM50.2_C++.csv",",",false);
    }
} 

