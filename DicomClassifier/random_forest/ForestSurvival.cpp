/*-------------------------------------------------------------------------------
This file is part of ranger.

Copyright (c) [2014-2018] [Marvin N. Wright]

This software may be modified and distributed under the terms of the MIT license.

Please note that the C++ core of ranger is distributed under MIT license and the
R package "ranger" under GPL3 license.
#-------------------------------------------------------------------------------*/

#include <set>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <memory> // Incluye <memory> para std::unique_ptr

#include "random_forest/utility.h"
#include "ForestSurvival.h"
#include "random_forest/Data.h"

namespace ranger {

// Función para cargar el bosque desde un archivo
void ForestSurvival::loadForest(size_t num_trees,
                                std::vector<std::vector<std::vector<size_t>> >& forest_child_nodeIDs,
                                std::vector<std::vector<size_t>>& forest_split_varIDs,
                                std::vector<std::vector<double>>& forest_split_values,
                                std::vector<std::vector<std::vector<double>> >& forest_chf,
                                std::vector<double>& unique_timepoints,
                                std::vector<bool>& is_ordered_variable) {

    this->num_trees = num_trees;
    this->unique_timepoints = unique_timepoints;
    data->setIsOrderedVariable(is_ordered_variable);

    // Crear árboles
    trees.reserve(num_trees);
    for (size_t i = 0; i < num_trees; ++i) {
        trees.push_back(std::unique_ptr<TreeSurvival>(new TreeSurvival(forest_child_nodeIDs[i], forest_split_varIDs[i],
                                                                       forest_split_values[i], forest_chf[i], &this->unique_timepoints, &response_timepointIDs)));
    }

    // Crear rangos de hilos
    equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
}

// Función para obtener la función de riesgo acumulada (CHF) de todos los árboles
std::vector<std::vector<std::vector<double>>> ForestSurvival::getChf() const {
    std::vector<std::vector<std::vector<double>>> result;
    result.reserve(num_trees);
    for (const auto& tree : trees) {
        const auto& temp = dynamic_cast<const TreeSurvival&>(*tree);
        result.push_back(temp.getChf());
    }
    return result;
}

// Función para inicializar parámetros internos del bosque
void ForestSurvival::initInternal() {

    // Si mtry no está establecido, usar la raíz cuadrada del número de variables independientes
    if (mtry == 0) {
        unsigned long temp = ceil(sqrt((double) num_independent_variables));
        mtry = std::max((unsigned long) 1, temp);
    }

    // Establecer tamaño mínimo del nodo
    if (min_node_size == 0) {
        min_node_size = DEFAULT_MIN_NODE_SIZE_SURVIVAL;
    }

    // Crear puntos de tiempo únicos
    if (!prediction_mode) {
        std::set<double> unique_timepoint_set;
        for (size_t i = 0; i < num_samples; ++i) {
            unique_timepoint_set.insert(data->get_y(i, 0));
        }
        unique_timepoints.reserve(unique_timepoint_set.size());
        for (auto& t : unique_timepoint_set) {
            unique_timepoints.push_back(t);
        }


            // Crear IDs de puntos de tiempo de respuesta
            for (size_t i = 0; i < num_samples; ++i) {
            double value = data->get_y(i, 0);

            // Si el punto de tiempo ya está en unique_timepoints, usar ID. De lo contrario, crear uno nuevo.
            uint timepointID = find(unique_timepoints.begin(), unique_timepoints.end(), value) - unique_timepoints.begin();
            response_timepointIDs.push_back(timepointID);
        }

    }

    // Ordenar datos si se usa extratrees y no se está en modo de ahorro de memoria
    if (splitrule == EXTRATREES && !memory_saving_splitting) {
        data->sort();
    }
}

// Función para hacer crecer el bosque
void ForestSurvival::growInternal() {
    trees.reserve(num_trees);
    for (size_t i = 0; i < num_trees; ++i) {
        trees.push_back(std::unique_ptr<TreeSurvival>(new TreeSurvival(&unique_timepoints, &response_timepointIDs)));
    }
}

// Función para asignar memoria para predicciones
void ForestSurvival::allocatePredictMemory() {
    size_t num_prediction_samples = data->getNumRows();
    size_t num_timepoints = unique_timepoints.size();
    if (predict_all) {
        predictions = std::vector<std::vector<std::vector<double>>>(num_prediction_samples,
                                                                    std::vector<std::vector<double>>(num_timepoints, std::vector<double>(num_trees, 0)));
    } else if (prediction_type == TERMINALNODES) {
        predictions = std::vector<std::vector<std::vector<double>>>(1,
                                                                    std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_trees, 0)));
    } else {
        predictions = std::vector<std::vector<std::vector<double>>>(1,
                                                                    std::vector<std::vector<double>>(num_prediction_samples, std::vector<double>(num_timepoints, 0)));
    }
}

// Función para realizar predicciones internas
void ForestSurvival::predictInternal(size_t sample_idx) {
    // Para cada punto de tiempo, sumar sobre los árboles
    if (predict_all) {
        for (size_t j = 0; j < unique_timepoints.size(); ++j) {
            for (size_t k = 0; k < num_trees; ++k) {
                predictions[sample_idx][j][k] = getTreePrediction(k, sample_idx)[j];
            }
        }
    } else if (prediction_type == TERMINALNODES) {
        for (size_t k = 0; k < num_trees; ++k) {
            predictions[0][sample_idx][k] = getTreePredictionTerminalNodeID(k, sample_idx);
        }
    } else {
        for (size_t j = 0; j < unique_timepoints.size(); ++j) {
            double sample_time_prediction = 0;
            for (size_t k = 0; k < num_trees; ++k) {
                sample_time_prediction += getTreePrediction(k, sample_idx)[j];
            }
            predictions[0][sample_idx][j] = sample_time_prediction / num_trees;
        }
    }
}

// Función para calcular el error de predicción interna
void ForestSurvival::computePredictionErrorInternal() {
    size_t num_timepoints = unique_timepoints.size();

    // Para cada muestra, sumar sobre los árboles donde la muestra es OOB
    std::vector<size_t> samples_oob_count;
    samples_oob_count.resize(num_samples, 0);
    predictions = std::vector<std::vector<std::vector<double>>>(1,
                                                                std::vector<std::vector<double>>(num_samples, std::vector<double>(num_timepoints, 0)));

    for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        for (size_t sample_idx = 0; sample_idx < trees[tree_idx]->getNumSamplesOob(); ++sample_idx) {
            size_t sampleID = trees[tree_idx]->getOobSampleIDs()[sample_idx];
            std::vector<double> tree_sample_chf = getTreePrediction(tree_idx, sample_idx);
                for (size_t time_idx = 0; time_idx < tree_sample_chf.size(); ++time_idx) {
                predictions[0][sampleID][time_idx] += tree_sample_chf[time_idx];
            }
            ++samples_oob_count[sampleID];
        }

    }

    // Dividir las predicciones de las muestras por el número de árboles donde la muestra es OOB y calcular el CHF sumado para las muestras
    std::vector<double> sum_chf;
    sum_chf.reserve(predictions[0].size());
    std::vector<size_t> oob_sampleIDs;
    oob_sampleIDs.reserve(predictions[0].size());
    for (size_t i = 0; i < predictions[0].size(); ++i) {
        if (samples_oob_count[i] > 0) {
            double sum = 0;
            for (size_t j = 0; j < predictions[0][i].size(); ++j) {
                predictions[0][i][j] /= samples_oob_count[i];
                sum += predictions[0][i][j];
            }
            sum_chf.push_back(sum);
            oob_sampleIDs.push_back(i);
        }
    }

    // Usar todas las muestras que son OOB al menos una vez
    overall_prediction_error = 1 - computeConcordanceIndex(*data, sum_chf, oob_sampleIDs, nullptr);
}

// Función para escribir la salida interna del bosque de supervivencia
void ForestSurvival::writeOutputInternal() {
    if (verbose_out) {
        *verbose_out << "Tree type: " << "Survival" << std::endl;
        if (dependent_variable_names.size() >= 2) {
            *verbose_out << "Status variable name: " << dependent_variable_names[1] << std::endl;
        }
    }
}

// Función para escribir el archivo de confusión
void ForestSurvival::writeConfusionFile() {
    // Abrir archivo de confusión para escritura
    std::string filename = output_prefix + ".confusion";
    std::ofstream outfile;
    outfile.open(filename, std::ios::out);
    if (!outfile.good()) {
        throw std::runtime_error("Could not write to confusion file: " + filename + ".");
    }

    // Escribir confusión en el archivo
    outfile << "Overall OOB prediction error (1 - C): " << overall_prediction_error << std::endl;

    outfile.close();
    if (verbose_out)
        *verbose_out << "Saved prediction error to file " << filename << "." << std::endl;
}

// Función para escribir el archivo de predicción
void ForestSurvival::writePredictionFile() {
    // Abrir archivo de predicción para escritura
    std::string filename = output_prefix + ".prediction";
    std::ofstream outfile;
    outfile.open(filename, std::ios::out);
    if (!outfile.good()) {
        throw std::runtime_error("Could not write to prediction file: " + filename + ".");
    }

    // Escribir
    outfile << "Unique timepoints: " << std::endl;
    for (auto& timepoint : unique_timepoints) {
        outfile << timepoint << " ";
    }
    outfile << std::endl << std::endl;

    outfile << "Cumulative hazard function, one row per sample: " << std::endl;
    if (predict_all) {
        for (size_t k = 0; k < num_trees; ++k) {
            outfile << "Tree " << k << ":" << std::endl;
            for (size_t i = 0; i < predictions.size(); ++i) {
                for (size_t j = 0; j < predictions[i].size(); ++j) {
                    outfile << predictions[i][j][k] << " ";
                }
                outfile << std::endl;
            }
            outfile << std::endl;
        }
    } else {
        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[i].size(); ++j) {
                for (size_t k = 0; k < predictions[i][j].size(); ++k) {
                    outfile << predictions[i][j][k] << " ";
                }
                outfile << std::endl;
            }
        }
    }

    if (verbose_out)
        *verbose_out << "Saved predictions to file " << filename << "." << std::endl;
}

// Función para guardar el bosque en un archivo
void ForestSurvival::saveToFileInternal(std::ofstream& outfile) {
    // Escribir num_variables
    outfile.write((char*) &num_independent_variables, sizeof(num_independent_variables));

    // Escribir treetype
    TreeType treetype = TREE_SURVIVAL;
    outfile.write((char*) &treetype, sizeof(treetype));

    // Escribir unique_timepoints
    saveVector1D(unique_timepoints, outfile);
}

// Función para cargar el bosque desde un archivo
void ForestSurvival::loadFromFileInternal(std::ifstream& infile) {
    // Leer número de variables
    size_t num_variables_saved;
    infile.read((char*) &num_variables_saved, sizeof(num_variables_saved));

    // Leer tipo de árbol
    TreeType treetype;
    infile.read((char*) &treetype, sizeof(treetype));
    if (treetype != TREE_SURVIVAL) {
        throw std::runtime_error("Wrong treetype. Loaded file is not a survival forest.");
    }

    // Leer unique_timepoints
    unique_timepoints.clear();
    readVector1D(unique_timepoints, infile);

    for (size_t i = 0; i < num_trees; ++i) {
        // Leer datos
        std::vector<std::vector<size_t>> child_nodeIDs;
        readVector2D(child_nodeIDs, infile);
        std::vector<size_t> split_varIDs;
        readVector1D(split_varIDs, infile);
        std::vector<double> split_values;
        readVector1D(split_values, infile);



                // Leer chf
                std::vector<size_t> terminal_nodes;
        readVector1D(terminal_nodes, infile);
        std::vector<std::vector<double>> chf_vector;
        readVector2D(chf_vector, infile);

        // Convertir chf a vector con elementos vacíos para nodos no terminales
        std::vector<std::vector<double>> chf;
        chf.resize(child_nodeIDs[0].size(), std::vector<double>());
        for (size_t j = 0; j < terminal_nodes.size(); ++j) {
            chf[terminal_nodes[j]] = chf_vector[j];
        }

        // Si la variable dependiente no está en los datos de prueba, lanzar un error
        if (num_variables_saved != num_independent_variables) {
            throw std::runtime_error("Number of independent variables in data does not match with the loaded forest.");
        }

        // Crear árbol
        trees.push_back(std::unique_ptr<TreeSurvival>(new TreeSurvival(child_nodeIDs, split_varIDs, split_values, chf,
                                                                       &unique_timepoints, &response_timepointIDs)));

    }
}

// Función para obtener la predicción de un árbol específico para una muestra específica
const std::vector<double>& ForestSurvival::getTreePrediction(size_t tree_idx, size_t sample_idx) const {
    const auto& tree = dynamic_cast<const TreeSurvival&>(*trees[tree_idx]);
    return tree.getPrediction(sample_idx);
}

// Función para obtener el ID del nodo terminal de la predicción de un árbol específico para una muestra específica
size_t ForestSurvival::getTreePredictionTerminalNodeID(size_t tree_idx, size_t sample_idx) const {
    const auto& tree = dynamic_cast<const TreeSurvival&>(*trees[tree_idx]);
    return tree.getPredictionTerminalNodeID(sample_idx);
}

// #nocov end

} // namespace ranger
