#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <bits/stdc++.h>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include "dicom/DicomReader.h"
#include "Otsu/Otsu_Bin.h"
#include "performance/benchmark.h"
#include<omp.h>

using namespace std;
typedef int element;

int main()
{

    DicomReader dicomObj("/home/jbarrionuevop/Desktop/dicom/images/MasaMicro1.dcm");
    vector<vector<element>> vect = dicomObj.getIntImageMatrix(12);
    systemMetrics systemMetrics("Otsu");
    systemMetrics.resetCounters();
    int rows = vect.size();
    int cols = vect[0].size();
    systemMetrics.calculate();
    systemMetrics.printMetrics();
    cout <<"RAM: " <<getRamUsage()<<" kb"<<endl;
    cv :: Mat imagen = cv:: Mat(rows, cols, CV_16UC1, &vect);

    cout << "columnas de Sla Imagen: "<<cols<< endl;
    cout << "filas de la Imagen : "<<rows<< endl;

    return 0;
}
