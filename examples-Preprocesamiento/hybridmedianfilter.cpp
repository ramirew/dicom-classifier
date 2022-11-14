#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <vector>
#include <math.h>

#include <omp.h>
#include "performance/benchmark.h"
#include "hybridmedianfilter/hybridmedianfilter.h"
#include "dicom/DicomReader.h"

using namespace std;

int main()
{
   //enviando numero de hilos en los subprocesos posteriores
    int nHilos=8;
    omp_set_num_threads(nHilos);
    double tbegin, time;
    //leyendo imagen de origen Dicom
    DicomReader dicomObj("/home/jair/Pictures/datasetDicom/MasaMicro5.dcm");
    //la convierte en vector de vectores int
    vector<vector<int>> data = dicomObj.getIntImageMatrix(12);
    systemMetrics metricas("filtro HMF");
    metricas.resetCounters();
    //ejecucion del filtro/
    tbegin = omp_get_wtime();
         //ejecucion del filtro//
         data=hybridmedianfilterN(data,7);
    //tomar tiempo final de ejecucion y calcularlo
    time = omp_get_wtime() - tbegin;
    metricas.calculate();
    int RAM = getRamUsage();
    metricas.printMetrics();
    cout << "RAM = " << RAM << endl;
    cout << "Hilos:" << nHilos << " ; Tiempo:" << time <<endl;
    //saveData(vector de vectores, path de imagen en char, separador entre comillas, si se quiere cabecera en booleano)
    dicomObj.saveData(data,"/home/jair/Pictures/datasetDicomR/MasaMicro5hmf7.csv",",",false);

    cout << "Proceso finalizado, CSV GUARDADO" << endl;

   

    return 0;
}
