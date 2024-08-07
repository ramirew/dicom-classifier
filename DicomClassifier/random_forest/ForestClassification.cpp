/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software puede ser modificado y distribuido bajo los términos de la licencia MIT.

 Tenga en cuenta que el núcleo en C++ de ranger se distribuye bajo la licencia MIT y el
 paquete R "ranger" bajo la licencia GPL3.
 #-------------------------------------------------------------------------------*/

#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <random>
#include <stdexcept>
#include <cmath>
#include <string>
#include <memory> // Incluir el encabezado <memory> para usar punteros inteligentes

#include "random_forest/utility.h"
#include "ForestClassification.h"
#include "random_forest/TreeClassification.h"
#include "random_forest/Data.h"

namespace ranger {

// Función para cargar el bosque desde archivos
void ForestClassification::loadForest(size_t num_trees,
                                      std::vector<std::vector<std::vector<size_t>> >& forest_child_nodeIDs,
                                      std::vector<std::vector<size_t>>& forest_split_varIDs, std::vector<std::vector<double>>& forest_split_values,
                                      std::vector<double>& class_values, std::vector<bool>& is_ordered_variable) {

    this->num_trees = num_trees;
    this->class_values = class_values;
    data->setIsOrderedVariable(is_ordered_variable);

    // Crear árboles
    trees.reserve(num_trees);
    for (size_t i = 0; i < num_trees; ++i) {
        // Cambiado std::make_unique por std::unique_ptr con new para C++11
        trees.push_back(std::unique_ptr<TreeClassification>(
            new TreeClassification(forest_child_nodeIDs[i], forest_split_varIDs[i], forest_split_values[i],
                                   &this->class_values, &response_classIDs)));
    }

    // Crear rangos de hilos
    equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
}

// Función de inicialización interna
void ForestClassification::initInternal() {

    // Si mtry no está configurado, usar la raíz cuadrada del número de variables independientes.
    if (mtry == 0) {
        unsigned long temp = sqrt((double) num_independent_variables);
        mtry = std::max((unsigned long) 1, temp);
    }

    // Establecer tamaño mínimo del nodo
    if (min_node_size == 0) {
        min_node_size = DEFAULT_MIN_NODE_SIZE_CLASSIFICATION;
    }

    // Crear class_values y response_classIDs
    if (!prediction_mode) {
        for (size_t i = 0; i < num_samples; ++i) {
            double value = data->get_y(i, 0);

            // Si classID ya está en class_values, usar ID. De lo contrario, crear uno nuevo.
            uint classID = find(class_values.begin(), class_values.end(), value) - class_values.begin();
            if (classID == class_values.size()) {
                class_values.push_back(value);
            }
            response_classIDs.push_back(classID);
        }

        if (splitrule == HELLINGER && class_values.size() != 2) {
            throw std::runtime_error("Hellinger splitrule only implemented for binary classification.");
        }
    }

    // Crear sampleIDs_per_class si es necesario
    if (sample_fraction.size() > 1) {
        sampleIDs_per_class.resize(sample_fraction.size());
        for (auto& v : sampleIDs_per_class) {
            v.reserve(num_samples);
        }
        for (size_t i = 0; i < num_samples; ++i) {
            size_t classID = response_classIDs[i];
            sampleIDs_per_class[classID].push_back(i);
        }
    }

    // Establecer todos los pesos de clase a 1
    class_weights = std::vector<double>(class_values.size(), 1.0);

    // Ordenar datos si está en modo de ahorro de memoria
    if (!memory_saving_splitting) {
        data->sort();
    }
}

// Función interna para hacer crecer el bosque
void ForestClassification::growInternal() {
    trees.reserve(num_trees);
    for (size_t i = 0; i < num_trees; ++i) {
        // Cambiado std::make_unique por std::unique_ptr con new para C++11
        trees.push_back(std::unique_ptr<TreeClassification>(
            new TreeClassification(&class_values, &response_classIDs, &sampleIDs_per_class, &class_weights)));
    }
}

// Función para asignar memoria para predicciones
void ForestClassification::allocatePredictMemory() {
    size_t num_prediction_samples = data->getNumRows();
    if (predict_all || prediction_type == TERMINALNODES) {
        predictions = std::vector<std::vector<std::vector<double>>>(1,
                                                                    std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_trees)));
    } else {
        predictions = std::vector<std::vector<std::vector<double>>>(1,
                                                                    std::vector<std::vector<double>>(1, std::vector<double>(num_prediction_samples)));
    }
}

// Función interna para realizar predicciones
void ForestClassification::predictInternal(size_t sample_idx) {
    if (predict_all || prediction_type == TERMINALNODES) {
        // Obtener todas las predicciones de los árboles
        for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
            if (prediction_type == TERMINALNODES) {
                predictions[0][sample_idx][tree_idx] = getTreePredictionTerminalNodeID(tree_idx, sample_idx);
            } else {
                predictions[0][sample_idx][tree_idx] = getTreePrediction(tree_idx, sample_idx);
            }
        }
    } else {
        // Contar clases sobre los árboles y guardar la clase con la cuenta máxima
        std::unordered_map<double, size_t> class_count;
        for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
            ++class_count[getTreePrediction(tree_idx, sample_idx)];
        }
        predictions[0][0][sample_idx] = mostFrequentValue(class_count, random_number_generator);
    }
}

// Función interna para calcular el error de predicción
void ForestClassification::computePredictionErrorInternal() {

    // Contadores de clases para las muestras
    std::vector<std::unordered_map<double, size_t>> class_counts;
    class_counts.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        class_counts.push_back(std::unordered_map<double, size_t>());
    }

    // Para cada árbol, iterar sobre las muestras OOB y contar las clases
    for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        for (size_t sample_idx = 0; sample_idx < trees[tree_idx]->getNumSamplesOob(); ++sample_idx) {
            size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
            ++class_counts[sampleID][getTreePrediction(tree_idx, sample_idx)];
        }
    }

    // Calcular el voto mayoritario para cada muestra
    predictions = std::vector<std::vector<std::vector<double>>>(1,
                                                                std::vector<std::vector<double>>(1, std::vector<double>(num_samples)));
    for (size_t i = 0; i < num_samples; ++i) {
        if (!class_counts[i].empty()) {
            predictions[0][0][i] = mostFrequentValue(class_counts[i], random_number_generator);
        } else {
            predictions[0][0][i] = NAN;
        }
    }

    // Comparar predicciones con los datos reales
    size_t num_missclassifications = 0;
    size_t num_predictions = 0;
    for (size_t i = 0; i < predictions[0][0].size(); ++i) {
        double predicted_value = predictions[0][0][i];
        if (!std::isnan(predicted_value)) {
            ++num_predictions;
            double real_value = data->get_y(i, 0);
            if (predicted_value != real_value) {
                ++num_missclassifications;
            }
            ++classification_table[std::make_pair(real_value, predicted_value)];
        }
    }
    overall_prediction_error = (double) num_missclassifications / (double) num_predictions;
}

// #nocov start
// Función para escribir la salida interna
void ForestClassification::writeOutputInternal() {
    if (verbose_out) {
        *verbose_out << "Tree type:                         " << "Classification" << std::endl;
    }
}

// Función para escribir el archivo de confusión
void ForestClassification::writeConfusionFile() {

    // Abrir archivo de confusión para escribir
    std::string filename = output_prefix + ".confusion";
    std::ofstream outfile;
    outfile.open(filename, std::ios::out);
    if (!outfile.good()) {
        throw std::runtime_error("Could not write to confusion file: " + filename + ".");
    }

    // Escribir confusión en el archivo
    outfile << "Overall OOB prediction error (Fraction missclassified): " << overall_prediction_error << std::endl;
    outfile << std::endl;
    outfile << "Class specific prediction errors:" << std::endl;
    outfile << "           ";
    for (auto& class_value : class_values) {
        outfile << "     " << class_value;
    }
    outfile << std::endl;
    for (auto& predicted_value : class_values) {
        outfile << "predicted " << predicted_value << "     ";
        for (auto& real_value : class_values) {
            size_t value = classification_table[std::make_pair(real_value, predicted_value)];
            outfile << value;
            if (value < 10) {
                outfile << "     ";
            } else if (value < 100) {
                outfile << "    ";
            } else if (value < 1000) {
                outfile << "   ";
            } else if (value < 10000) {
                outfile << "  ";
            } else if (value < 100000) {
                outfile << " ";
            }
        }
        outfile << std::endl;
    }

    outfile.close();
    if (verbose_out)
        *verbose_out << "Saved confusion matrix to file " << filename << "." << std::endl;
}

// Función para escribir el archivo de predicción
void ForestClassification::writePredictionFile() {

    // Abrir archivo de predicción para escribir
    std::string filename = output_prefix + ".prediction";
    std::ofstream outfile;
    outfile.open(filename, std::ios::out);
    if (!outfile.good()) {
        throw std::runtime_error("Could not write to prediction file: " + filename + ".");
    }

    // Escribir predicciones
    outfile << "Predictions: " << std::endl;
    if (predict_all) {
        for (size_t k = 0; k < num_trees; ++k) {
            outfile << "Tree " << k << ":" << std::endl;
            for (size_t i = 0; i < predictions.size(); ++i) {
                for (size_t j = 0; j < predictions[i].size(); ++j) {
                    outfile << predictions[i][j][k] << std::endl;
                }
            }
            outfile << std::endl;
        }
    } else {
        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[i].size(); ++j) {
                for (size_t k = 0; k < predictions[i][j].size(); ++k) {
                    outfile << predictions[i][j][k] << std::endl;
                }
            }
        }
    }

    if (verbose_out)
        *verbose_out << "Saved predictions to file " << filename << "." << std::endl;
}

// Función para guardar el archivo interno
void ForestClassification::saveToFileInternal(std::ofstream& outfile) {

    // Escribir num_variables
    outfile.write(reinterpret_cast<const char*>(&num_independent_variables), sizeof(num_independent_variables));

    // Escribir treetype
    TreeType treetype = TREE_CLASSIFICATION;
    outfile.write(reinterpret_cast<const char*>(&treetype), sizeof(treetype));

    // Escribir class_values
    saveVector1D(class_values, outfile);
}

// Función para cargar el archivo interno
void ForestClassification::loadFromFileInternal(std::ifstream& infile) {

    // Leer el número de variables
    size_t num_variables_saved;
    infile.read(reinterpret_cast<char*>(&num_variables_saved), sizeof(num_variables_saved));

    // Leer treetype
    TreeType treetype;
    infile.read(reinterpret_cast<char*>(&treetype), sizeof(treetype));
    if (treetype != TREE_CLASSIFICATION) {
        throw std::runtime_error("Wrong treetype. Loaded file is not a classification forest.");
    }

    // Leer class_values
    readVector1D(class_values, infile);

    for (size_t i = 0; i < num_trees; ++i) {

        // Leer datos
        std::vector<std::vector<size_t>> child_nodeIDs;
        readVector2D(child_nodeIDs, infile);
        std::vector<size_t> split_varIDs;
        readVector1D(split_varIDs, infile);
        std::vector<double> split_values;
        readVector1D(split_values, infile);

        // Si la variable dependiente no está en los datos de prueba, lanzar error
        if (num_variables_saved != num_independent_variables) {
            throw std::runtime_error("Number of independent variables in data does not match with the loaded forest.");
        }

        // Crear árbol
        trees.push_back(std::unique_ptr<TreeClassification>(
            new TreeClassification(child_nodeIDs, split_varIDs, split_values, &class_values, &response_classIDs)));
    }
}

// Obtener predicción del árbol
double ForestClassification::getTreePrediction(size_t tree_idx, size_t sample_idx) const {
    const auto& tree = dynamic_cast<const TreeClassification&>(*trees[tree_idx]);
    return tree.getPrediction(sample_idx);
}

// Obtener el ID del nodo terminal de predicción del árbol
size_t ForestClassification::getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const {
    const auto& tree = dynamic_cast<const TreeClassification&>(*trees[tree_idx]);
    return tree.getPredictionTerminalNodeID(sample_idx);
}

// #nocov end

}// namespace ranger

