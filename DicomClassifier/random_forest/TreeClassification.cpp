/*-------------------------------------------------------------------------------
 This file is parte de ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 Este software puede ser modificado y distribuido bajo los términos de la licencia MIT.

 Tenga en cuenta que el núcleo en C++ de ranger se distribuye bajo la licencia MIT y el
 paquete R "ranger" bajo la licencia GPL3.
 #-------------------------------------------------------------------------------*/

#include <unordered_map>
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>

#include "TreeClassification.h"
#include "random_forest/utility.h"
#include "random_forest/Data.h"

namespace ranger {

// Constructor de TreeClassification, inicializando con valores de clase, IDs de clase de respuesta, IDs de muestra por clase y pesos de clase
TreeClassification::TreeClassification(std::vector<double>* class_values, std::vector<uint>* response_classIDs,
                                       std::vector<std::vector<size_t>>* sampleIDs_per_class, std::vector<double>* class_weights) :
    class_values(class_values), response_classIDs(response_classIDs), sampleIDs_per_class(sampleIDs_per_class), class_weights(
          class_weights), counter(0), counter_per_class(0) {
}

// Constructor para cargar un árbol desde un estado guardado
TreeClassification::TreeClassification(std::vector<std::vector<size_t>>& child_nodeIDs,
                                       std::vector<size_t>& split_varIDs, std::vector<double>& split_values, std::vector<double>* class_values,
                                       std::vector<uint>* response_classIDs) :
    Tree(child_nodeIDs, split_varIDs, split_values), class_values(class_values), response_classIDs(response_classIDs), sampleIDs_per_class(
          nullptr), class_weights(nullptr), counter { }, counter_per_class { } {
}

// Asignar memoria para contadores
void TreeClassification::allocateMemory() {
    // Inicializar contadores si no está en modo de ahorro de memoria
    if (!memory_saving_splitting) {
        size_t num_classes = class_values->size();
        size_t max_num_splits = data->getMaxNumUniqueValues();

        // Usar el número de divisiones aleatorias para extratrees
        if (splitrule == EXTRATREES && num_random_splits > max_num_splits) {
            max_num_splits = num_random_splits;
        }

        counter.resize(max_num_splits);
        counter_per_class.resize(num_classes * max_num_splits);
    }
}

// Estimar el valor para un nodo dado
double TreeClassification::estimate(size_t nodeID) {
    // Contar clases sobre muestras en el nodo y devolver la clase con el máximo conteo
    std::vector<double> class_count = std::vector<double>(class_values->size(), 0.0);

    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        size_t value = (*response_classIDs)[sampleID];
        class_count[value] += (*class_weights)[value];
    }

    if (end_pos[nodeID] > start_pos[nodeID]) {
        size_t result_classID = mostFrequentClass(class_count, random_number_generator);
        return ((*class_values)[result_classID]);
    } else {
        throw std::runtime_error("Error: Empty node.");
    }
}

// Función vacía para añadir a archivo
void TreeClassification::appendToFileInternal(std::ofstream& file) { // #nocov start
    // Intencionalmente vacía
} // #nocov end

// Dividir nodo basado en variables de división posibles
bool TreeClassification::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
    // Detener si se alcanza el tamaño máximo del nodo o la profundidad
    size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
    if (num_samples_node <= min_node_size || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
        split_values[nodeID] = estimate(nodeID);
        return true;
    }

    // Comprobar si el nodo es puro y establecer el valor de división para detener si es puro
    bool pure = true;
    double pure_value = 0;
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        double value = data->get_y(sampleID, 0);
        if (pos != start_pos[nodeID] && value != pure_value) {
            pure = false;
            break;
        }
        pure_value = value;
    }
    if (pure) {
        split_values[nodeID] = pure_value;
        return true;
    }

    // Encontrar la mejor división, detener si no hay disminución de la impureza
    bool stop;
    if (splitrule == EXTRATREES) {
        stop = findBestSplitExtraTrees(nodeID, possible_split_varIDs);
    } else {
        stop = findBestSplit(nodeID, possible_split_varIDs);
    }

    if (stop) {
        split_values[nodeID] = estimate(nodeID);
        return true;
    }

    return false;
}

// Función vacía para crear un nodo vacío
void TreeClassification::createEmptyNodeInternal() {
    // Intencionalmente vacía
}

// Calcular la precisión de la predicción
double TreeClassification::computePredictionAccuracyInternal(std::vector<double>* prediction_error_casewise) {
    size_t num_predictions = prediction_terminal_nodeIDs.size();
    size_t num_missclassifications = 0;
    for (size_t i = 0; i < num_predictions; ++i) {
        size_t terminal_nodeID = prediction_terminal_nodeIDs[i];
        double predicted_value = split_values[terminal_nodeID];
        double real_value = data->get_y(oob_sampleIDs[i], 0);
        if (predicted_value != real_value) {
            ++num_missclassifications;
            if (prediction_error_casewise) {
                (*prediction_error_casewise)[i] = 1;
            }
        } else {
            if (prediction_error_casewise) {
                (*prediction_error_casewise)[i] = 0;
            }
        }
    }
    return (1.0 - (double) num_missclassifications / (double) num_predictions);
}

// Encontrar la mejor división para el nodo dado
bool TreeClassification::findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
    size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
    size_t num_classes = class_values->size();
    double best_decrease = -1;
    size_t best_varID = 0;
    double best_value = 0;

    std::vector<size_t> class_counts(num_classes);
    // Calcular los conteos de clase generales
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        uint sample_classID = (*response_classIDs)[sampleID];
        ++class_counts[sample_classID];
    }

    // Para todas las variables de división posibles
    for (auto& varID : possible_split_varIDs) {
        // Encontrar el mejor valor de división, si está ordenado considerar todos los valores como valores de división, de lo contrario todas las particiones de 2
        if (data->isOrderedVariable(varID)) {
            // Usar el método de ahorro de memoria si la opción está configurada
            if (memory_saving_splitting) {
                findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID, best_decrease);
            } else {
                // Usar método más rápido para ambos casos
                double q = (double) num_samples_node / (double) data->getNumUniqueDataValues(varID);
                if (q < Q_THRESHOLD) {
                    findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID, best_decrease);
                } else {
                    findBestSplitValueLargeQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID, best_decrease);
                }
            }
        } else {
            findBestSplitValueUnordered(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID, best_decrease);
        }
    }

    // Detener si no se encuentra una buena división
    if (best_decrease < 0) {
        return true;
    }

    // Guardar los mejores valores
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;

    // Calcular el índice de gini para este nodo y añadir a la importancia de la variable si es necesario
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
        addGiniImportance(nodeID, best_varID, best_decrease);
    }

    // Regularización
    saveSplitVarID(best_varID);

    return false;
}

// Encontrar el mejor valor de división para variables ordenadas con pequeño Q
void TreeClassification::findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_classes,
                                                  const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
                                                  double& best_decrease) {

    // Crear valores de división posibles
    std::vector<double> possible_split_values;
    data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

    // Intentar con la siguiente variable si todos son iguales para esta
    if (possible_split_values.size() < 2) {
        return;
    }

    const size_t num_splits = possible_split_values.size();
    if (memory_saving_splitting) {
        std::vector<size_t> class_counts_right(num_splits * num_classes), n_right(num_splits);
        findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
                                 best_decrease, possible_split_values, class_counts_right, n_right);
    } else {
        std::fill_n(counter_per_class.begin(), num_splits * num_classes, 0);
        std::fill_n(counter.begin(), num_splits, 0);
        findBestSplitValueSmallQ(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
                                 best_decrease, possible_split_values, counter_per_class, counter);
    }
}

// Función auxiliar para findBestSplitValueSmallQ
void TreeClassification::findBestSplitValueSmallQ(size_t nodeID, size_t varID, size_t num_classes,
                                                  const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
                                                  double& best_decrease, const std::vector<double>& possible_split_values, std::vector<size_t>& counter_per_class,
                                                  std::vector<size_t>& counter) {

    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        uint sample_classID = (*response_classIDs)[sampleID];
        size_t idx = std::lower_bound(possible_split_values.begin(), possible_split_values.end(),
                                      data->get_x(sampleID, varID)) - possible_split_values.begin();

        ++counter_per_class[idx * num_classes + sample_classID];
        ++counter[idx];
    }

    size_t n_left = 0;
    std::vector<size_t> class_counts_left(num_classes);

    // Calcular la disminución de la impureza para cada división
    for (size_t i = 0; i < possible_split_values.size() - 1; ++i) {

        // Detener si no hay nada aquí
        if (counter[i] == 0) {
            continue;
        }

        n_left += counter[i];

        // Detener si el hijo derecho está vacío
        size_t n_right = num_samples_node - n_left;
        if (n_right == 0) {
            break;
        }

        double decrease;
        if (splitrule == HELLINGER) {
            for (size_t j = 0; j < num_classes; ++j) {
                class_counts_left[j] += counter_per_class[i * num_classes + j];
            }

            // TPR es el número de resultados 1 en un nodo / número total de 1s
            // FPR es el número de resultados 0 en un nodo / número total de 0s
            double tpr = (double) (class_counts[1] - class_counts_left[1]) / (double) class_counts[1];
            double fpr = (double) (class_counts[0] - class_counts_left[0]) / (double) class_counts[0];

            // Disminución de la impureza
            double a1 = sqrt(tpr) - sqrt(fpr);
            double a2 = sqrt(1 - tpr) - sqrt(1 - fpr);
            decrease = sqrt(a1 * a1 + a2 * a2);
        } else {
            // Suma de cuadrados
            double sum_left = 0;
            double sum_right = 0;
            for (size_t j = 0; j < num_classes; ++j) {
                class_counts_left[j] += counter_per_class[i * num_classes + j];
                size_t class_count_right = class_counts[j] - class_counts_left[j];

                sum_left += (*class_weights)[j] * class_counts_left[j] * class_counts_left[j];
                sum_right += (*class_weights)[j] * class_count_right * class_count_right;
            }

            // Disminución de la impureza
            decrease = sum_right / (double) n_right + sum_left / (double) n_left;
        }

        // Regularización
        regularize(decrease, varID);

        // Si es mejor que antes, usar esto
        if (decrease > best_decrease) {
            // Usar división en el punto medio
            best_value = (possible_split_values[i] + possible_split_values[i + 1]) / 2;
            best_varID = varID;
            best_decrease = decrease;

            // Usar valor más pequeño si el promedio es numéricamente el mismo que el valor más grande
            if (best_value == possible_split_values[i + 1]) {
                best_value = possible_split_values[i];
            }
        }
    }
}

// Encontrar el mejor valor de división para variables ordenadas con gran Q
void TreeClassification::findBestSplitValueLargeQ(size_t nodeID, size_t varID, size_t num_classes,
                                                  const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
                                                  double& best_decrease) {

    // Establecer contadores a 0
    size_t num_unique = data->getNumUniqueDataValues(varID);
    std::fill_n(counter_per_class.begin(), num_unique * num_classes, 0);
    std::fill_n(counter.begin(), num_unique, 0);

    // Contar valores
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        size_t index = data->getIndex(sampleID, varID);
        size_t classID = (*response_classIDs)[sampleID];

        ++counter[index];
        ++counter_per_class[index * num_classes + classID];
    }

    size_t n_left = 0;
    std::vector<size_t> class_counts_left(num_classes);

    // Calcular la disminución de la impureza para cada división
    for (size_t i = 0; i < num_unique - 1; ++i) {

        // Detener si no hay nada aquí
        if (counter[i] == 0) {
            continue;
        }

        n_left += counter[i];

        // Detener si el hijo derecho está vacío
        size_t n_right = num_samples_node - n_left;
        if (n_right == 0) {
            break;
        }

        double decrease;
        if (splitrule == HELLINGER) {
            for (size_t j = 0; j < num_classes; ++j) {
                class_counts_left[j] += counter_per_class[i * num_classes + j];
            }

            // TPR es el número de resultados 1 en un nodo / número total de 1s
            // FPR es el número de resultados 0 en un nodo / número total de 0s
            double tpr = (double) (class_counts[1] - class_counts_left[1]) / (double) class_counts[1];
            double fpr = (double) (class_counts[0] - class_counts_left[0]) / (double) class_counts[0];

            // Disminución de la impureza
            double a1 = sqrt(tpr) - sqrt(fpr);
            double a2 = sqrt(1 - tpr) - sqrt(1 - fpr);
            decrease = sqrt(a1 * a1 + a2 * a2);
        } else {
            // Suma de cuadrados
            double sum_left = 0;
            double sum_right = 0;
            for (size_t j = 0; j < num_classes; ++j) {
                class_counts_left[j] += counter_per_class[i * num_classes + j];
                size_t class_count_right = class_counts[j] - class_counts_left[j];

                sum_left += (*class_weights)[j] * class_counts_left[j] * class_counts_left[j];
                sum_right += (*class_weights)[j] * class_count_right * class_count_right;
            }

            // Disminución de la impureza
            decrease = sum_right / (double) n_right + sum_left / (double) n_left;
        }

        // Regularización
        regularize(decrease, varID);

        // Si es mejor que antes, usar esto
        if (decrease > best_decrease) {
            // Encontrar el siguiente valor en este nodo
            size_t j = i + 1;
            while (j < num_unique && counter[j] == 0) {
                ++j;
            }

            // Usar división en el punto medio
            best_value = (data->getUniqueDataValue(varID, i) + data->getUniqueDataValue(varID, j)) / 2;
            best_varID = varID;
            best_decrease = decrease;

            // Usar valor más pequeño si el promedio es numéricamente el mismo que el valor más grande
            if (best_value == data->getUniqueDataValue(varID, j)) {
                best_value = data->getUniqueDataValue(varID, i);
            }
        }
    }
}

// Encontrar el mejor valor de división para variables no ordenadas
void TreeClassification::findBestSplitValueUnordered(size_t nodeID, size_t varID, size_t num_classes,
                                                     const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
                                                     double& best_decrease) {

    // Crear valores de división posibles
    std::vector<double> factor_levels;
    data->getAllValues(factor_levels, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

    // Intentar con la siguiente variable si todos son iguales para esta
    if (factor_levels.size() < 2) {
        return;
    }

    // El número de posibles divisiones es 2^num_levels
    size_t num_splits = (1ULL << factor_levels.size());

    // Calcular la disminución de la impureza para cada división posible
    // Dividir donde todos los de la izquierda (0) o todos los de la derecha (1) están excluidos
    // La segunda mitad de los números es solo izquierda/derecha invertida de la primera mitad -> Excluir la segunda mitad
    for (size_t local_splitID = 1; local_splitID < num_splits / 2; ++local_splitID) {

        // Calcular splitID general desplazando local factorIDs a posiciones globales
        size_t splitID = 0;
        for (size_t j = 0; j < factor_levels.size(); ++j) {
            if ((local_splitID & (1ULL << j))) {
                double level = factor_levels[j];
                size_t factorID = floor(level) - 1;
                splitID = splitID | (1ULL << factorID);
            }
        }

        // Inicializar
        std::vector<size_t> class_counts_right(num_classes);
        size_t n_right = 0;

        // Contar clases en hijo izquierdo y derecho
        for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
            size_t sampleID = sampleIDs[pos];
            uint sample_classID = (*response_classIDs)[sampleID];
            double value = data->get_x(sampleID, varID);
            size_t factorID = floor(value) - 1;

            // Si está en el hijo derecho, contar
            // En el hijo derecho, si splitID bit a bit en la posición factorID es 1
            if ((splitID & (1ULL << factorID))) {
                ++n_right;
                ++class_counts_right[sample_classID];
            }
        }
        size_t n_left = num_samples_node - n_right;

        double decrease;
        if (splitrule == HELLINGER) {
            // TPR es el número de resultados 1 en un nodo / número total de 1s
            // FPR es el número de resultados 0 en un nodo / número total de 0s
            double tpr = (double) class_counts_right[1] / (double) class_counts[1];
            double fpr = (double) class_counts_right[0] / (double) class_counts[0];

            // Disminución de la impureza
            double a1 = sqrt(tpr) - sqrt(fpr);
            double a2 = sqrt(1 - tpr) - sqrt(1 - fpr);
            decrease = sqrt(a1 * a1 + a2 * a2);
        } else {
            // Suma de cuadrados
            double sum_left = 0;
            double sum_right = 0;
            for (size_t j = 0; j < num_classes; ++j) {
                size_t class_count_right = class_counts_right[j];
                size_t class_count_left = class_counts[j] - class_count_right;

                sum_right += (*class_weights)[j] * class_count_right * class_count_right;
                sum_left += (*class_weights)[j] * class_count_left * class_count_left;
            }

            // Disminución de la impureza
            decrease = sum_left / (double) n_left + sum_right / (double) n_right;
        }

        // Regularización
        regularize(decrease, varID);

        // Si es mejor que antes, usar esto
        if (decrease > best_decrease) {
            best_value = splitID;
            best_varID = varID;
            best_decrease = decrease;
        }
    }
}

// Encontrar la mejor división para ExtraTrees
bool TreeClassification::findBestSplitExtraTrees(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
    size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
    size_t num_classes = class_values->size();
    double best_decrease = -1;
    size_t best_varID = 0;
    double best_value = 0;

    std::vector<size_t> class_counts(num_classes);
    // Calcular los conteos de clase generales
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        uint sample_classID = (*response_classIDs)[sampleID];
        ++class_counts[sample_classID];
    }

    // Para todas las variables de división posibles
    for (auto& varID : possible_split_varIDs) {
        // Encontrar el mejor valor de división, si está ordenado considerar todos los valores como valores de división, de lo contrario todas las particiones de 2
        if (data->isOrderedVariable(varID)) {
            findBestSplitValueExtraTrees(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
                                         best_decrease);
        } else {
            findBestSplitValueExtraTreesUnordered(nodeID, varID, num_classes, class_counts, num_samples_node, best_value,
                                                  best_varID, best_decrease);
        }
    }

    // Detener si no se encuentra una buena división
    if (best_decrease < 0) {
        return true;
    }

    // Guardar los mejores valores
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;

    // Calcular el índice de gini para este nodo y añadir a la importancia de la variable si es necesario
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
        addGiniImportance(nodeID, best_varID, best_decrease);
    }

    // Regularización
    saveSplitVarID(best_varID);

    return false;
}

// Encontrar el mejor valor de división para ExtraTrees con variables ordenadas
void TreeClassification::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, size_t num_classes,
                                                      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
                                                      double& best_decrease) {

    // Obtener valores mínimos/máximos de covariables en el nodo
    double min;
    double max;
    data->getMinMaxValues(min, max, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

    // Intentar con la siguiente variable si todos son iguales para esta
    if (min == max) {
        return;
    }

    // Crear valores de división posibles: dibujar aleatoriamente entre min y max
    std::vector<double> possible_split_values;
    std::uniform_real_distribution<double> udist(min, max);
    possible_split_values.reserve(num_random_splits);
    for (size_t i = 0; i < num_random_splits; ++i) {
        possible_split_values.push_back(udist(random_number_generator));
    }
    if (num_random_splits > 1) {
        std::sort(possible_split_values.begin(), possible_split_values.end());
    }

    const size_t num_splits = possible_split_values.size();
    if (memory_saving_splitting) {
        std::vector<size_t> class_counts_right(num_splits * num_classes), n_right(num_splits);
        findBestSplitValueExtraTrees(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
                                     best_decrease, possible_split_values, class_counts_right, n_right);
    } else {
        std::fill_n(counter_per_class.begin(), num_splits * num_classes, 0);
        std::fill_n(counter.begin(), num_splits, 0);
        findBestSplitValueExtraTrees(nodeID, varID, num_classes, class_counts, num_samples_node, best_value, best_varID,
                                     best_decrease, possible_split_values, counter_per_class, counter);
    }
}

// Función auxiliar para findBestSplitValueExtraTrees
void TreeClassification::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, size_t num_classes,
                                                      const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
                                                      double& best_decrease, const std::vector<double>& possible_split_values, std::vector<size_t>& class_counts_right,
                                                      std::vector<size_t>& n_right) {
    const size_t num_splits = possible_split_values.size();

    // Contar muestras en hijo derecho por clase y división posible
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        double value = data->get_x(sampleID, varID);
        uint sample_classID = (*response_classIDs)[sampleID];

        // Contar muestras hasta que se alcance el valor de división
        for (size_t i = 0; i < num_splits; ++i) {
            if (value > possible_split_values[i]) {
                ++n_right[i];
                ++class_counts_right[i * num_classes + sample_classID];
            } else {
                break;
            }
        }
    }

    // Calcular la disminución de la impureza para cada división posible
    for (size_t i = 0; i < num_splits; ++i) {

        // Detener si un hijo está vacío
        size_t n_left = num_samples_node - n_right[i];
        if (n_left == 0 || n_right[i] == 0) {
            continue;
        }

        // Suma de cuadrados
        double sum_left = 0;
        double sum_right = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            size_t class_count_right = class_counts_right[i * num_classes + j];
            size_t class_count_left = class_counts[j] - class_count_right;

            sum_right += (*class_weights)[j] * class_count_right * class_count_right;
            sum_left += (*class_weights)[j] * class_count_left * class_count_left;
        }

        // Disminución de la impureza
        double decrease = sum_left / (double) n_left + sum_right / (double) n_right[i];

        // Regularización
        regularize(decrease, varID);

        // Si es mejor que antes, usar esto
        if (decrease > best_decrease) {
            best_value = possible_split_values[i];
            best_varID = varID;
            best_decrease = decrease;
        }
    }
}

// Encontrar el mejor valor de división para ExtraTrees con variables no ordenadas
void TreeClassification::findBestSplitValueExtraTreesUnordered(size_t nodeID, size_t varID, size_t num_classes,
                                                               const std::vector<size_t>& class_counts, size_t num_samples_node, double& best_value, size_t& best_varID,
                                                               double& best_decrease) {

    size_t num_unique_values = data->getNumUniqueDataValues(varID);

    // Obtener todos los índices de factores en el nodo
    std::vector<bool> factor_in_node(num_unique_values, false);
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
        size_t sampleID = sampleIDs[pos];
        size_t index = data->getIndex(sampleID, varID);
        factor_in_node[index] = true;
    }

    // Vector de índices dentro y fuera del nodo
    std::vector<size_t> indices_in_node;
    std::vector<size_t> indices_out_node;
    indices_in_node.reserve(num_unique_values);
    indices_out_node.reserve(num_unique_values);
    for (size_t i = 0; i < num_unique_values; ++i) {
        if (factor_in_node[i]) {
            indices_in_node.push_back(i);
        } else {
            indices_out_node.push_back(i);
        }
    }

    // Generar num_random_splits divisiones
    for (size_t i = 0; i < num_random_splits; ++i) {
        std::vector<size_t> split_subset;
        split_subset.reserve(num_unique_values);

        // Dibujar subconjuntos aleatorios, muestrear todas las particiones con igual probabilidad
        if (indices_in_node.size() > 1) {
            size_t num_partitions = (2ULL << (indices_in_node.size() - 1ULL)) - 2ULL; // 2^n-2 (no permitir lleno o vacío)
            std::uniform_int_distribution<size_t> udist(1, num_partitions);
            size_t splitID_in_node = udist(random_number_generator);
            for (size_t j = 0; j < indices_in_node.size(); ++j) {
                if ((splitID_in_node & (1ULL << j)) > 0) {
                    split_subset.push_back(indices_in_node[j]);
                }
            }
        }
        if (indices_out_node.size() > 1) {
            size_t num_partitions = (2ULL << (indices_out_node.size() - 1ULL)) - 1ULL; // 2^n-1 (permitir lleno o vacío)
            std::uniform_int_distribution<size_t> udist(0, num_partitions);
            size_t splitID_out_node = udist(random_number_generator);
            for (size_t j = 0; j < indices_out_node.size(); ++j) {
                if ((splitID_out_node & (1ULL << j)) > 0) {
                    split_subset.push_back(indices_out_node[j]);
                }
            }
        }

        // Asignar unión de los dos subconjuntos al hijo derecho
        size_t splitID = 0;
        for (auto& idx : split_subset) {
            splitID |= 1ULL << idx;
        }

        // Inicializar
        std::vector<size_t> class_counts_right(num_classes);
        size_t n_right = 0;

        // Contar clases en hijo izquierdo y derecho
        for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
            size_t sampleID = sampleIDs[pos];
            uint sample_classID = (*response_classIDs)[sampleID];
            double value = data->get_x(sampleID, varID);
            size_t factorID = floor(value) - 1;

            // Si está en el hijo derecho, contar
            // En el hijo derecho, si splitID bit a bit en la posición factorID es 1
            if ((splitID & (1ULL << factorID))) {
                ++n_right;
                ++class_counts_right[sample_classID];
            }
        }
        size_t n_left = num_samples_node - n_right;

        // Suma de cuadrados
        double sum_left = 0;
        double sum_right = 0;
        for (size_t j = 0; j < num_classes; ++j) {
            size_t class_count_right = class_counts_right[j];
            size_t class_count_left = class_counts[j] - class_count_right;

            sum_right += (*class_weights)[j] * class_count_right * class_count_right;
            sum_left += (*class_weights)[j] * class_count_left * class_count_left;
        }

        // Disminución de la impureza
        double decrease = sum_left / (double) n_left + sum_right / (double) n_right;

        // Regularización
        regularize(decrease, varID);

        // Si es mejor que antes, usar esto
        if (decrease > best_decrease) {
            best_value = splitID;
            best_varID = varID;
            best_decrease = decrease;
        }
    }
}

// Añadir importancia Gini para el nodo y la variable dados
void TreeClassification::addGiniImportance(size_t nodeID, size_t varID, double decrease) {

    double best_decrease = decrease;
    if (splitrule != HELLINGER) {
        size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
        std::vector<size_t> class_counts;
        class_counts.resize(class_values->size(), 0);

        for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
            size_t sampleID = sampleIDs[pos];
            uint sample_classID = (*response_classIDs)[sampleID];
            class_counts[sample_classID]++;
        }
        double sum_node = 0;
        for (size_t i = 0; i < class_counts.size(); ++i) {
            sum_node += (*class_weights)[i] * class_counts[i] * class_counts[i];
        }

        double impurity_node = (sum_node / (double) num_samples_node);

        // Tener en cuenta la regularización
        regularize(impurity_node, varID);

        best_decrease = decrease - impurity_node;
    }

    // No hay importancia de variable para variables no divididas
    size_t tempvarID = data->getUnpermutedVarID(varID);

    // Restar si importancia corregida y variable permutada, de lo contrario añadir
    if (importance_mode == IMP_GINI_CORRECTED && varID >= data->getNumCols()) {
        (*variable_importance)[tempvarID] -= best_decrease;
    } else {
        (*variable_importance)[tempvarID] += best_decrease;
    }
}

// Muestras de bootstrap por clase
void TreeClassification::bootstrapClassWise() {
    // El número de muestras es la suma de la fracción de muestra * número de muestras
    size_t num_samples_inbag = 0;
    double sum_sample_fraction = 0;
    for (auto& s : *sample_fraction) {
        num_samples_inbag += (size_t) num_samples * s;
        sum_sample_fraction += s;
    }

    // Reservar espacio, reservar un poco más para estar seguro
    sampleIDs.reserve(num_samples_inbag);
    oob_sampleIDs.reserve(num_samples * (exp(-sum_sample_fraction) + 0.1));

    // Comenzar con todas las muestras OOB
    inbag_counts.resize(num_samples, 0);

    // Dibujar muestras para cada clase
    for (size_t i = 0; i < sample_fraction->size(); ++i) {
        // Dibujar muestras de clase con reemplazo como inbag y marcar como no OOB
        size_t num_samples_class = (*sampleIDs_per_class)[i].size();
        size_t num_samples_inbag_class = round(num_samples * (*sample_fraction)[i]);
        std::uniform_int_distribution<size_t> unif_dist(0, num_samples_class - 1);
        for (size_t s = 0; s < num_samples_inbag_class; ++s) {
            size_t draw = (*sampleIDs_per_class)[i][unif_dist(random_number_generator)];
            sampleIDs.push_back(draw);
            ++inbag_counts[draw];
        }
    }

    // Guardar muestras OOB
    for (size_t s = 0; s < inbag_counts.size(); ++s) {
        if (inbag_counts[s] == 0) {
            oob_sampleIDs.push_back(s);
        }
    }
    num_samples_oob = oob_sampleIDs.size();

    if (!keep_inbag) {
        inbag_counts.clear();
        inbag_counts.shrink_to_fit();
    }
}

// Muestras de bootstrap por clase sin reemplazo
void TreeClassification::bootstrapWithoutReplacementClassWise() {
    // Dibujar muestras para cada clase
    for (size_t i = 0; i < sample_fraction->size(); ++i) {
        size_t num_samples_class = (*sampleIDs_per_class)[i].size();
        size_t num_samples_inbag_class = round(num_samples * (*sample_fraction)[i]);

        shuffleAndSplitAppend(sampleIDs, oob_sampleIDs, num_samples_class, num_samples_inbag_class,
                              (*sampleIDs_per_class)[i], random_number_generator);
    }
    num_samples_oob = oob_sampleIDs.size();

    if (keep_inbag) {
        // Todas las observaciones son 0 o 1 veces inbag
        inbag_counts.resize(num_samples, 1);
        for (size_t i = 0; i < oob_sampleIDs.size(); i++) {
            inbag_counts[oob_sampleIDs[i]] = 0;
        }
    }
}

} // namespace ranger

