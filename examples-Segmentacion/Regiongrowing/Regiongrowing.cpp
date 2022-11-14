#include <iostream>
#include <segmentacion/regiongrowing.h>
#include <vector>
#include "regiongrowing.h"
#include <bits/stdc++.h>
#include <memory.h>
#include "dicom/DicomReader.h"
#include "performance/benchmark.h"
#include <omp.h>
using namespace std;

int main()
{
    int reg_size;//Numeros de pixeles en la region
    int reg_mean;//La media de la región segmentada
    //puntos semilla (x,y)
    int x=198;
    int y=359;

    //Leemos nuestra imagen dicom
    DicomReader dicomobj("C://Users//Jonathan//Desktop//imag.dcm");
    //Convertimos en vector de vectores
    vector<vector<element>> vect = dicomobj.getIntImageMatrix(12);
    systemMetrics systemMetrics("regiongrowing");
    systemMetrics.resetCounters();
    //llamamos a los parametros de nuestra tecnica
    vector<vector<element>> regiongrowing(vector<vector<element>> src,int x,int y, int reg_size, int reg_mean);
    //Llamamos las funciones de la tecnica
    vect = ::regiongrowing(vect);
    systemMetrics.calculate();
    systemMetrics.printMetrics();
    //Imprimimos las metricas
        cout<<"RAM: " <<getRamUsage()<<"kb"<<endl;
    DicomReader::saveData(vect,"C://Users//Jonathan//Desktop//matrizresultante.txt",",",0);
    return 0;

}

