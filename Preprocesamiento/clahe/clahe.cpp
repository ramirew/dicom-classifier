#include <iostream>
#include "dicom/DicomReader.h"
#include "performance/benchmark.h"
#include <sstream>
#include <array>
#include <chrono>
#include <algorithm>
#include <omp.h>

using namespace std;


//---------------------------------------------------------------------------------


//Funcion para lectura de imagen
//Guardar Matriz de imagen Dicom de 12 bits 0 - 4096
//retornar la matriz (vector de vectores)
vector<vector<int>> newMatriz(DicomReader dicomObj){
     vector<vector<int>> data = dicomObj.getIntImageMatrix(12); //guarda la matriz de la img dicom
     return data;
}
//Crear una copia de matriz data
//retornar la matriz (vector de vectores)
vector<vector<int>> newCLAHE_GO(vector<vector<int>> data){
     vector<vector<int>> CLAHE_GO(data.size());
     copy(data.begin(), data.end(), CLAHE_GO.begin());
     return CLAHE_GO;
 }
 //Fucion clahe tipo (vector de vectores) recibe como parametro la matriz de imagen original y la matriz
// de imagen clonada
 vector<vector<int>> claheGo(vector<vector<int>> CLAHE_GO,vector<vector<int>> data,DicomReader dicomObj){
     int _step = 12; //paso para cada celda resultante (12 pixels x 12 pixels)
     int block = _step;//block -> longitud maxima que tomara cada celda de cada bloque movil en la imagen
     int width = (sizeof(data)/sizeof(data[0])); //Guardar el valor de ancho de la imagen
     int height= (sizeof(data[0])/sizeof(data[0][0])); // Guardar el valor de alto de imagen
     int width_block = width/block; //separar cada bloque movil a lo ancho con un maximo de 12 elementos
     int height_block = height/block;//separar cada bloque movil a lo alto con un maximo de 12 elementos
     //variables (declaro matriz) Guardar cada histograma generado de la imagen
     int tmp2[12*12][4096] ={{0}}; // matriz temporal para almacenar cada histograma
     float C2[12*12][4096] = {{0.0}};// matriz que guardara el valor temporal de cada histograma

     //Tamaño total elementos de cada bloque movil
     int total = width_block * height_block;

     //recorrer cada bloque movil matriz (vector de vectores)
     //filas
     for (int i=0;i<block;i++)
     {
         //columnas
         for (int j=0;j<block;j++)
         {
             //Declarar variables que tomaran en cuenta el tamaño de cada bloque movil
             //Eje X
             int start_x = i*width_block; // Se debe multiplicar cada valor ancho maximo por el iterador filas
             int end_x = start_x + width_block; // Se debe sumar "star_x" y el ancho del bloque, que sera el final del bloque movil
             //Eje Y
             int start_y = j*height_block;// Se debe multiplicar cada valor alto maximo por el iterador columnas
             int end_y = start_y + height_block;// Se debe sumar "star_y" y el alto del bloque, que sera el final del bloque movil
             int num = i+block*j; //Variable para guardar valor de fila en el recorrido del histograma

             // Recorrer los bloques pequeños y calcular el histograma
             // Recorrer cada bloque movil en filas y columnas con la dimencion establecida
             //#pragma omp parallel for num_threads(2) collapse (2)
      #pragma omp parallel num_threads(2)
        {
             #pragma opm parallel for num_threads(1) collapse (1)
             for( int ii = start_x ; ii < end_x ; ii++)
             {

                 for(int jj = start_y ; jj < end_y ; jj++)
                 {
                     // declarar variable que recorrerar cada bloque
                     int index =data[jj][ii];
                     // Guardar valor de histrograma en la variable temporal
                        #pragma omp atomic
                        tmp2[num][index]++;
                 }
             }

             //Declarar variables internas
             //Promedio del bloque movil
             int average = width_block * height_block / 4095; //Guardar el promedio de ancho x alto / 4095 correspondiente a 12 bits
             int LIMIT = 40 * average;// Dar un limite y multiplicarlo por el promedio
             int steal = 0; //Variable que guarde un valor temporal del histograma - el Limite
             int k = 0;
             #pragma opm parallel for
             for(k = 0 ; k < 4096 ; k++)
             {
                 if(tmp2[num][k] > LIMIT){
                     #pragma omp atomic
                     steal += tmp2[num][k] - LIMIT;
                     #pragma omp atomic write
                     tmp2[num][k] = LIMIT;
                 }
             }
             #pragma omp barrier
             int bonus = steal/4096;
             //Repartir los valores temporales en promedio´
             //#pragma omp parallel for reduction(+:tmp2[num][k])
             #pragma opm parallel for
             for(int k = 0 ; k < 4096 ; k++)
             {
                 #pragma omp atomic
                 tmp2[num][k] += bonus;
             }
             //Clacular histograma de distribucion acumulada
             #pragma opm parallel for
             for(int k = 0 ; k < 4096 ; k++)
             {
                 if( k == 0)
                     #pragma omp atomic write
                     C2[num][k] = 1.0f * tmp2[num][k] / total;
                 else
                     #pragma omp atomic write
                     C2[num][k] = C2[num][k-1] + 1.0f * tmp2[num][k] / total;
             }
            }//final hilos

         }

     }
     //Calcular el valor del píxel transformado
     //Según la posición del píxel, elija un método de cálculo diferente
    #pragma omp master
     for(int  i = 0 ; i < width; i++)
     {
        #pragma omp parallel for collapse(1)
         for(int j = 0 ; j < height; j++)
         {
             //Leer 4 esquinas
             #pragma execution_frequency(very_high)
             if(i <= width_block/2 && j <= height_block/2)
             {
                 int num = 0;
                 CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 4095);
             }else if(i <= width_block/2 && j >= ((block-1)*height_block + height_block/2)){
                 int num = block*(block-1);
                 CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 4095);
             }else if(i >= ((block-1)*width_block+width_block/2) && j <= height_block/2){
                 int num = block-1;
                 CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 4095);
             }else if(i >= ((block-1)*width_block+width_block/2) && j >= ((block-1)*height_block + height_block/2)){
                 int num = block*block-1;
                 CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 4095);
             }
             //Leer 4 bordes exepto esquinas
             #pragma execution_frequency(very_high)
             else if( i <= width_block/2 )
             {
                 //Interpolación lineal
                 int num_i = 0;
                 int num_j = (j - height_block/2)/height_block;
                 int num1 = num_j*block + num_i;
                 int num2 = num1 + block;
                 float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                 float q = 1-p;
                 CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 4095);

             }
             #pragma execution_frequency(very_high)
             else if( i >= ((block-1)*width_block+width_block/2)){
                 //Interpolación lineal
                 int num_i = block-1;
                 int num_j = (j - height_block/2)/height_block;
                 int num1 = num_j*block + num_i;
                 int num2 = num1 + block;
                 float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                 float q = 1-p;
                 CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 4095);
             }
             #pragma execution_frequency(very_high)
             else if( j <= height_block/2 ){
                 //Interpolación lineal
                 int num_i = (i - width_block/2)/width_block;
                 int num_j = 0;
                 int num1 = num_j*block + num_i;
                 int num2 = num1 + 1;
                 float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                 float q = 1-p;
                 CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 4095);
             }
             #pragma execution_frequency(very_high)
             else if( j >= ((block-1)*height_block + height_block/2) ){
                 //Interpolación lineal
                 int num_i = (i - width_block/2)/width_block;
                 int num_j = block-1;
                 int num1 = num_j*block + num_i;
                 int num2 = num1 + 1;
                 float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                 float q = 1-p;
                 CLAHE_GO[j][i] = (int)((q*C2[num1][CLAHE_GO[j][i]]+ p*C2[num2][CLAHE_GO[j][i]])* 4095);
             }
             #pragma execution_frequency(very_low)
             else{
                 // interpolación bilineal
                 int num_i = (i - width_block/2)/width_block;
                 int num_j = (j - height_block/2)/height_block;
                 int num1 = num_j*block + num_i;
                 int num2 = num1 + 1;
                 int num3 = num1 + block;
                 int num4 = num2 + block;
                 float u = (i - (num_i*width_block+width_block/2))/(1.0f*width_block);
                 float v = (j - (num_j*height_block+height_block/2))/(1.0f*height_block);
                 CLAHE_GO[j][i] = (int)((u*v*C2[num4][CLAHE_GO[j][i]] +
                     (1-v)*(1-u)*C2[num1][CLAHE_GO[j][i]] +
                     u*(1-v)*C2[num2][CLAHE_GO[j][i]] +
                     v*(1-u)*C2[num3][CLAHE_GO[j][i]]) * 4095);
             }
             //suavizamiento de imagen
             CLAHE_GO[j][i] = CLAHE_GO[j][i] + (CLAHE_GO[j][i] << 8) + (CLAHE_GO[j][i] << 16);
         }
     }
     //Obtener datos de matriz en CSV
     dicomObj.saveData(CLAHE_GO,"/home/boris/Documentos/DicomClaheH2paralelizado/csvC/csv5/MasaMicroC1.csv",",",false);
     return CLAHE_GO;

 }
