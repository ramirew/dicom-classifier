//REGION GROWING
//Autores: Jonathan Adriano, Kevin Jimenez
//FECHA: 26 Julio 2022

#include <iostream>
#include <vector>
#include "regiongrowing.h"
#include <bits/stdc++.h>
#include <iostream>
#include <segmentacion/regiongrowing.cpp>
#include <memory.h>
#include "dicom/DicomReader.h"
#include "performance/benchmark.h"
#include <omp.h>

using namespace std;
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <memory.h>

using namespace std;
typedef int element;

//reg_mean = I(x,y); La media de la region segmentada
//reg_size = 1;  Numero de pixels en la region
//x,y : La posicion de los puntos semilla

int regiongrowin (vector<vector<element>> im, int x, int y, int reg_size, int reg_mean)
{
vector <int> pixels;
int a, b;

#pragma opm parallel num threads(32) [
#pragma opm for collapse(2) shared (Im, reg_size, x, y) private (a, b) schedule(static)
//Agregar nuevos píxeles vecinos
for (int a = -reg_size / 2; a <= reg_size / 2; a++) {
    for (int b = -reg_size / 2; b <= reg_size / 2; b++){
        pixels.push_back(im[x + a][y + b]) ;
}
}

//Neighbor locations
//neigb=[-1 0; 1 0; 0 -1;0 1];

sort(pixels.begin(), pixels.end());
auto min = pixels[0];
auto max = pixels[reg_size * reg_size -1];
auto med = pixels[reg_size * reg_size / 1];
element zxy = im[x][y];

//Agregamos vecinos si está dentro y aún no forma parte del área segmentada
if(med > min && med < max) {
        if (zxy > min && zxy < max) {
            return zxy;
         }else{
  return med;
          }
}

else{
reg_size +=2;

if (reg_size <= reg_mean)
    return regiongrowing(im, x, y, reg_size, reg_mean); //Se devuela las coordenadas x e y del píx
else
    return med;
}

}

int main()
{

    DicomReader dicomobj("C://Users//Jonathan//Desktop//imag.dcm");
    vector<vector<element>> vect = dicomobj.getIntImageMatrix(32);
    systemMetrics systemMetrics("regiongrowing");
    systemMetrics.resetCounters();
    vect = regiongrowing(vect);
    systemMetrics.calculate();
    systemMetrics.printMetrics();
        cout<<"RAM: " <<getRamUsage()<<"kb"<<endl;
    DicomReader::saveData(vect,"C://Users//Jonathan//Desktop//matrizresultante.txt",",",0);
    return 0;

}




