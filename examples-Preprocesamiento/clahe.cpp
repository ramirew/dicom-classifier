#include <iostream>
#include "Clahe/clahe.h"
#include "dicom/DicomReader.h"
#include "performance/benchmark.h"
#include <omp.h>

using namespace std;

int main()
{
    double tbegin, time;

    //Leer la imagen Dicom del directorio dataset
    DicomReader dicomObj("/home/boris/Documentos/dicom-classifier/dataset/MasaMicro5.dcm");
    systemMetrics metricas("clahe");
    metricas.resetCounters();
    vector<vector<int>> data = dicomObj.getIntImageMatrix(12);
    //claheGo(newCLAHE_GO(newMatriz(dicomObj)),newMatriz(dicomObj),dicomObj);
    //Llamar a las Funciones de cabecera clahe.h
    tbegin = omp_get_wtime();
    claheGo(newCLAHE_GO(newMatriz(dicomObj)),newMatriz(dicomObj),dicomObj); //llamar a la librería clahe.h
    metricas.calculate();
    time = omp_get_wtime() - tbegin;
    int RAM = getRamUsage();
    metricas.printMetrics();
    cout << "RAM = " << RAM << endl;
    cout << "Tiempo paralelo = " << time << endl;
    cout << "Proceso finalizado, CSV GUARDADO" << endl;

    return 0;
}
