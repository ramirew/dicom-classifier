#include <iostream>
#include <vector>
#include "gaussian/gaussian.h"
#include "dicom/DicomReader.h"
#include <bits/stdc++.h>
#include <memory.h>

using namespace std;

// AUtores:  Paul Morales, Jefferson Romero, Leyton Valencia

int main()
{
    //Lectura Imágen Dicom
    DicomReader dicomObj("/home/debian/Documentos/dicom-classifier/dataset/MasaMicro5.dcm");
    // Conversión de imagén a matriz de imágen en 12 bits.
    vector<vector<filtro>> vect = dicomObj.getIntImageMatrix(12);
    //Llamada a la función del Filtro Gaussiano con sus respectivos parámetros.
    // Parámetro 1: Vector de imágen
    // Parámetro 2: Tamaño del Kernel (Entre mayor sea el tamaño se tendrá un efecto mayor del filtro).
    vect = Filtro_Gauss(vect,31);
    return 0;
}
