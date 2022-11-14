#include <iostream>
#include <vector>
#include "adaptativemedianfilter/adaptativemedianfilter.h"
#include <bits/stdc++.h>
#include <iostream>
#include <memory.h>
#include "dicom/DicomReader.h"
#include "performance/benchmark.h"
#include <omp.h>

using namespace std;

int main()
{

    DicomReader dicomObj("/home/dmarino/Documentos/build-proyecto_qt-Desktop_Qt_6_2_4_GCC_64bit-Debug/MasaMicro1.dcm");
    vector<vector<element>> vect = dicomObj.getIntImageMatrix(12);
    systemMetrics systemMetrics("adaptivemedianfilter");
    systemMetrics.resetCounters();
    vect = adaptivemedianfilter(vect,3,7);
    systemMetrics.calculate();
    systemMetrics.printMetrics();
    cout <<"RAM: " <<getRamUsage()<<" kb"<<endl;

    DicomReader::saveData(vect,"/home/dmarino/Documentos/dicom-classifier/matriz_resultante.txt",",",0);

    return 0;
}
