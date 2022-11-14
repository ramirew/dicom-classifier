// Binarizacion Otsu  filtro
// Autor: Jonathan Barrionuevo, Sanchez Carolina, Loachamin Andres, Campoverde Daniela
// Fecha: Julio 28 2022
// Universidad Politecnica Salesiana

/* El filtro de Otsu busca clasificar los pixeles presentes en una imagen clasificiando la escala de grises mediante un histograma
 * esto nos permite mediante segmentar la imagen y binarizar el histograma de grises en solo blancos o negros, esto gracias al uso
 * de la varianza interclases y la cantidad de blancos o grises presentes en la imagen
*/

#include <iostream>
#include <bits/stdc++.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include "dicom/DicomReader.h"

#include<omp.h>

using namespace std;
typedef int element;

int Binarizacion_Otsu(vector<vector<element>> im){
    vector <int> hist, omega, myu , prob,sigma;
    vector<vector <int>> im2;
    int x,y, y_size1 ,x_size1 , y_size2 ,x_size2;
    int threshold = 0;
    float max_sigma = 0.0;

    cout << "Iniciando Proceso de Binarizacion." << endl;

    /*Generando el Histograma*/

    #pragma opm parallel num_threads(32){ /*Indicando el numero de hilos con el que vamos a trabajar*/
    int n,h1[4096];	 // indicando la dimension por la profundidad de pixels a 12 bits
    #pragma omp for schedule(dynamic,20)
            for ( int i = 0; i < 4096; i++) hist[i] = 0,h1[i]=0;
        #pragma omp for collapse(2) private(x,y,h1) schedule(static,5000)
            for (y = 0; y < y_size1; y++)
                for (x = 0; x < x_size1; x++) {
                  h1[im[y][x]]++;
                }
      #pragma omp critical
    {
        for(n=0; n<10; ++n) {
            hist[n] += h1[n];
        }
    }

            /* Calculando la probabilidad de densidad */
        #pragma omp for schedule(static)
          for ( int i = 0; i < 4096; i ++ ) {
            prob[i] = (double)hist[i] / (x_size1 * y_size1);
          }

          /* Generacion de omega & myu */
          #pragma omp single
          {
          omega[0] = prob[0];
          myu[0] = 0.0;       /* 0.0 veces la probabilidad[0] se iguala a 0 por cada pixel que recorreos */
          for (int i = 1; i < 4096; i++) {
            omega[i] = omega[i-1] + prob[i];
            myu[i] = myu[i-1] + i*prob[i];
          }}

          /* maximizacion de sigma
           * el sigma se mantiene para la varianza interclase y nos ayuda a determinar
           * el valor optimo del threshold */

        #pragma omp parallel for schedule(dynamic,20) reduction(max:max_sigma)
          for (int i = 0; i < 4096-1; i++) {
          if (omega[i] != 0.0 && omega[i] != 1.0)
            sigma[i] = pow(myu[4096-1]*omega[i] - myu[i], 2) / (omega[i]*(1.0 - omega[i]));
          else
            sigma[i] = 0.0;
          if (sigma[i] > max_sigma) {
            max_sigma = sigma[i];
            threshold = i;
          }
        }


          /* Guardar  la imagen binarizada mediante el csv*/
          x_size2 = x_size1;
          y_size2 = y_size1;
            #pragma omp for collapse(2) private(x,y) schedule(static,5000)
                for (y = 0; y < y_size2; y++)
                  for (x = 0; x < x_size2; x++)
                    if (im[y][x] > threshold)
                im2[y][x] = 4095;
                    else
                im2[y][x] = 0;

        DicomReader::saveData(im2,"/home/jbarrionuevop/Desktop/dicom/Resultados/matriz_original.csv",",",0);

} // no tocar fin de la funcion
