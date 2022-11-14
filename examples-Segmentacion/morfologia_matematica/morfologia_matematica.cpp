#include <iostream>
#include <vector>
#include <stdio.h>
#include <omp.h>
#include "performance/benchmark.h"
#include "morfologia_matematica/morfologia_matematica.h"
#include "dicom/DicomReader.h"

using namespace std;

// Autores: Alvaro Lomas, Jhony Pillajo, Joel Romo
// Fecha: Julio 2022

int main()
{
    //DEFINIR NÚMERO DE HILOS A UTILIZAR, SI NO SE DEFINE OCUPA EL NÚMERO TOTAL DE PROCESADORES (6 POR DEFECTO)
    omp_set_num_threads(32);

    //DECLARAR E INICIALIZAR VARIABLE DE TIEMPO
    double tbegin = omp_get_wtime();

    //LECTURA DE IMAGEN DE TIPO DICOM
    DicomReader dicomObj("/home/debian11/Documentos/morfologia-matematica/build-test_morfologia_matematica-Desktop_Qt_6_2_4_GCC_64bit-Debug/dataset/MasaMicro1.dcm");
    
    //CONVERSIÓN DE IMAGEN A MATRIZ DE IMAGEN DE 12 BITS.
    vector<vector<int>> data = dicomObj.getIntImageMatrix(12);

    //DEFINIR PARÁMETROS DEL ELEMENTO ESTRUCTURANTE A CREAR, CON UNA DIMENSIÓN IMPAR Y CUALQUIERA DE LOS SIGUIENTES TIPOS:
    //1 -> KERNEL CUADRADO
    //2 -> KERNEL HORIZONTAL
    //3 -> KERNEL VERTICAL
    //4 -> KERNEL CRUZADO
    int dimension = 3; //DIMENSIÓN DE 3x3
    int tipo = 4; //KERNEL CRUZADO
    
    //LLAMADA A LA FUNCIÓN PARA CREAR EL ELEMENTO ESTRUCTURANTE
    // PARÁMETRO 1: Dimensión del elemento estructurante que debe ser impar.
    // PARÁMETRO 2: Tipo de elemento estructurante según los proporcionados por el método. 
    vector<vector<element>> elementoEstructurante = determinarKernel(dimension,tipo);

    //CREACION DE VECTORES PARA CADA OPERACIÓN MORFOLÓGICA
    vector<vector<element>> vectDilatacion, vectErosion, vectApertura, vectCierre;

    //LLAMADA A LAS FUNCIONES CORRESPONDIENTES A LAS OPERACIONES MORFOLÓGICAS
    //ENVÍA COMO PARÁMETROS LA IMAGEN DICOM Y EL ELEMENTO ESTRUCTURANTE PREVIAMENTE CREADO
    // PARÁMETRO 1: Vector de imagen
    // PARÁMETRO 2: Vector de elementos
    vectApertura = apertura(data, elementoEstructurante);
    vectCierre = cierre(data, elementoEstructurante);
    vectDilatacion = dilatacion(data, elementoEstructurante);
    vectErosion = erosion(data, elementoEstructurante);

    //GUARDAR RESULTADO EN UN ARCHIVO DE TEXTO
    // En caso de que se requiera guardar un archivo csv/txt con los píxeles resultantes, usamos la función DicomReader::saveData
    // PARÁMETRO 1: Matriz de imagen con los píxeles resultantes.
    // PARÁMETRO 2: Ruta dónde se almacenará la matriz de píxeles resultantes.
    // PARÁMETRO 3: Delimitador entre cada uno de los píxeles.
    // PARÁMETRO 4: 1 o 0 (Verdadero o False) según se desee o no añadir cabeceras al archivo csv/txt.
    DicomReader::saveData(vectApertura, "/home/debian11/Documentos/morfologia-matematica/build-test_morfologia_matematica-Desktop_Qt_6_2_4_GCC_64bit-Debug/Resultados/MM1_A.txt", ",", 0);
    DicomReader::saveData(vectCierre, "/home/debian11/Documentos/morfologia-matematica/build-test_morfologia_matematica-Desktop_Qt_6_2_4_GCC_64bit-Debug/Resultados/MM1_C.txt", ",", 0);
    DicomReader::saveData(vectDilatacion, "/home/debian11/Documentos/morfologia-matematica/build-test_morfologia_matematica-Desktop_Qt_6_2_4_GCC_64bit-Debug/Resultados/MM1_D.txt", ",", 0);
    DicomReader::saveData(vectErosion, "/home/debian11/Documentos/morfologia-matematica/build-test_morfologia_matematica-Desktop_Qt_6_2_4_GCC_64bit-Debug/Resultados/MM1_E.txt", ",", 0);

    //TOMAR TIEMPO FINAL Y CALCULARLO
    double time = omp_get_wtime() - tbegin;
    cout << "Execution Time De La Técnica: " << time <<"; Max Threads:" << omp_get_max_threads() <<endl;

    return 0;
}
