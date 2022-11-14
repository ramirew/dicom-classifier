#include <iostream>
#include <vector>
#include <math.h>
#include "dicom/DicomReader.h"
#include "Canny/canny.h"
#include "performance/benchmark.h"
#include "omp.h"

using namespace std;

int main()
{
    //Lectura de la imagen con la libreria dicom
    DicomReader dicomObj("/home/usuario/Descargas/imagendicom/dataset/MasaMicro4.dcm");
    //Transforma la imagen dcm  en una matriz de tipo int
    vector<vector<int>> vect = dicomObj.getIntImageMatrix(12);
    //Metricas referente a la libreria dicomclsasifier
    systemMetrics systemMetrics("Canny");
    systemMetrics.resetCounters();
    //metodo canny tiene como parametros de entrada(matriz de imagen(vect),valorminimo(30),valormaximo(50))
    vect = canny(vect,30,50);
    systemMetrics.calculate();
    //Imprime las metricas en consola
    systemMetrics.printMetrics();
    //Imprimire en consola el total de ram utilizado en el programa
    cout <<"RAM: " <<getRamUsage()<<" kb"<<endl;
    //Guardra la matriz de imagen en un txt
    dicomObj.saveData(vect,"/home/usuario/canny_paralelo1.csv",",",false);
    return 0;

}
