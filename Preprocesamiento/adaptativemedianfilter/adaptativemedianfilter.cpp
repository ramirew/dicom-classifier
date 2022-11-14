//ADAPTIVE MEDIAN FILTER
//AUTORES:Mariño Daniel, Sarmiento Steven, Raza Sebastian
//FECHA: Julio 2022


// El filtro adaptativo de la mediana está diseñado para eliminar ruido presente en imágenes. En especial, los inconvenientes que presenta el filtro estandar de la mediana. Su mayor ventaja es el tamaño variable del kernel que rodea a la imagen corrupta para obtener un mejor resultado.

#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <memory.h>

using namespace std;
typedef int element;

//Método de filtrado del pixel recibido del método adaptivemedianfilter.
int  adaptiveProcess(vector<vector<element>> im, int row, int col, int kernelSize, int maxSize)
{
    //Se determina los valores máximo, mínimo y mediana del kernel.
    vector <int> pixels;
    int a,b;
    #pragma opm parallel num_threads(32) {
      #pragma omp for collapse(2) shared(im,kernelSize,row,col) private(a,b) schedule(static)
      for(int a = -kernelSize / 2; a <= kernelSize / 2; a++){
        for(int b = -kernelSize / 2; b <= kernelSize / 2; b++){
            pixels.push_back(im[row + a][col + b]);
        }
      }
    }
    sort(pixels.begin(), pixels.end());
    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    element zxy = im[row][col];
    //Se determina si el punto central del kernel es o no ruido. Si no es ruido se mantiene el punto, caso contrario toma el valor de la mediana.
    if(med > min && med < max){
        if(zxy > min && zxy < max){
            return zxy;
        }else{
            return med;
        }
    }
    else{
        //Si el valor de la mediana obtenido es un punto de ruido, se aumenta el tamaño del kernel hasta que deje de serlo o llegue a su valor máximo de tamaño.
        kernelSize += 2;
        if(kernelSize <= maxSize)
            return adaptiveProcess(im, row, col, kernelSize, maxSize);
        else
            return med;
    }
}

//Método principal. Requiere tres argumentos: Vector de imagen, tamaño minimo inicial del kernel, tamaño máximo que tomará el kernel.
//Tanto el tamaño mínimo (minSize), como el tamaño máximo (maxSize) deben ser números impares.
//Se recomienda minSize=3, maxSize=7 o minSize=3, maxSize=5. Si hay abundante ruido se recomienda un kernel de mayor tamaño.
//Método que recorre el kernel a través de la imagen.

vector<vector<element>> adaptivemedianfilter(vector<vector<element>> src, int minSize, int maxSize){
    vector<vector<element>> dst = src;
    int rows = dst.size();
    int cols = dst[0].size();
    int i,j;
    #pragma opm parallel num_threads(32) {
      #pragma opm for collapse(2) shared(rows,cols,maxSize,minSize) private(i,j) schedule(static)
      for(j = maxSize / 2; j < rows - maxSize / 2; j++){
        for(i = maxSize / 2; i < cols * 1 - maxSize / 2; i++){
            dst[i][j] = adaptiveProcess(dst, j, i, minSize, maxSize);
        }
      }
    } 
    return dst;
}
