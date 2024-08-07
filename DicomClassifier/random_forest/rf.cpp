/*-------------------------------------------------------------------------------
                                                                                                                      This file is part of ranger.

                                                                                                                      Copyright (c) [2014-2018] [Marvin N. Wright]

                                                                                                                      This software may be modified and distributed under the terms of the MIT license.

                                                                                                                      Please note that the C++ core of ranger is distributed under MIT license and the
                                                                                                                      R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/

#include "rf.h"
#include <stdexcept>

#include "random_forest/globals.h"
#include "random_forest/ForestClassification.h"
#include "random_forest/utility.h"
#include "random_forest/Data.h"

                                                                                                                      using namespace std;

// Constructor de la clase RF, inicializa los valores por defecto y crea el objeto forest
RF::RF(int trees) : totalTrees(trees), mtry(0), mode(MEM_DOUBLE), outprefix("random_forest"),
    default_seed(0), predict_file(""), split_weights_file(""), depvarname("LABEL"),
    status_var_name(""), replacement(false), save_memory(false), predall(false),
    samplefraction(0), holdout(false), reg_usedepth(false) {

            // Crear el objeto forest como un puntero único a ForestClassification
            forest = std::unique_ptr<ranger::Forest>(new ranger::ForestClassification());

}

// Destructor de la clase RF
RF::~RF() {}

// Función para establecer los datos de entrenamiento
void RF::setTrainData(vector<vector<double>> data, vector<int> target) {
    filename = generateDataFormat(data, target); // Necesita el nombre del archivo de entrenamiento
}

// Función para realizar predicciones
void RF::predict(vector<vector<double>> data, vector<int> target, bool showOutput) {
    predict_file = generateDataFormat(data, target); // Necesita el nombre del archivo de predicción



            init(showOutput);

}

// Función para inicializar el modelo
void RF::init(bool showOutput) {
    if (showOutput) {
        forest->initCpp(depvarname, mode, filename, mtry, outprefix, totalTrees, &std::cout,
                        default_seed, DEFAULT_NUM_THREADS, predict_file, DEFAULT_IMPORTANCE_MODE, DEFAULT_MIN_NODE_SIZE_CLASSIFICATION,
                        split_weights_file, split_vars, status_var_name, replacement, cat_vars, save_memory,
                        DEFAULT_SPLITRULE, weights_file, predall, samplefraction, DEFAULT_ALPHA,
                        DEFAULT_MINPROP, holdout, DEFAULT_PREDICTIONTYPE, DEFAULT_NUM_RANDOM_SPLITS,
                        DEFAULT_MAXDEPTH, reg_factor, reg_usedepth);
    } else {
        forest->initCpp(depvarname, mode, filename, mtry, outprefix, totalTrees, nullptr,
                        default_seed, DEFAULT_NUM_THREADS, predict_file, DEFAULT_IMPORTANCE_MODE, DEFAULT_MIN_NODE_SIZE_CLASSIFICATION,
                        split_weights_file, split_vars, status_var_name, replacement, cat_vars, save_memory,
                        DEFAULT_SPLITRULE, weights_file, predall, samplefraction, DEFAULT_ALPHA,
                        DEFAULT_MINPROP, holdout, DEFAULT_PREDICTIONTYPE, DEFAULT_NUM_RANDOM_SPLITS,
                        DEFAULT_MAXDEPTH, reg_factor, reg_usedepth);
    }
}

// Función para establecer el nombre del archivo de entrenamiento
void RF::setFile(string filename) {
    this->filename = filename;
}

// Función para establecer el nombre del archivo de predicción
void RF::setPredictFile(string filename) {
    this->predict_file = filename;
}

// Función para establecer el nombre de la variable dependiente
void RF::setDepVarName(string name) {
    this->depvarname = name;
}

// Función para generar el formato de los datos y guardarlos en un archivo temporal
string RF::generateDataFormat(vector<vector<double>> data, vector<int> target) {
    string filename = "temp.data"; // Nombre del archivo temporal
    string delimiter = " "; // Delimitador de campos



        if (data.size() != target.size()) {
        string message = "Data dimension and target size must be equal";
        throw runtime_error(message);
    }

    std::ofstream output_file(filename);

    // Escribir el encabezado
    for (size_t i = 0; i < data[0].size(); ++i) {
        output_file << "X" << i << delimiter;
    }
    output_file << "LABEL" << endl;

    // Escribir los datos
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            output_file << data[i][j] << delimiter;
        }
        output_file << target[i] << "\n";
    }

    return filename;

}
