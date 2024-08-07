#include "ReadDataset.h"
#include "dataset.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <vector>

using namespace std;

// Definición de la función read
DatasetPointer ReadDataset::read(std::string filename, int nLabels) {
    // Abre el archivo en modo binario
    ifstream myFile(filename, ios::binary);

    // Verifica si el archivo se abrió correctamente
    if (!myFile.is_open()) {
        throw invalid_argument("filename");
    }

    size_t nExamples, nDim;


    DatasetPointer result = DatasetPointer(new dataset_base(nExamples,nDim,nLabels));

    // Itera sobre cada ejemplo
    for(size_t i = 0; i < nExamples; i++) {
        // Lee los valores de las características para cada ejemplo
        for (size_t j = 0; j < nDim; j++) {
            myFile.read(reinterpret_cast<char*>(&result->pos(i,j)), sizeof(float)); // Asumiendo que los datos son de tipo float
        }
        // Lee la etiqueta correspondiente para cada ejemplo
        myFile.read(reinterpret_cast<char*>(&result->label(i)), sizeof(int)); // Asumiendo que las etiquetas son de tipo int
    }

    // Retorna el puntero inteligente al dataset cargado
    return result;
}
