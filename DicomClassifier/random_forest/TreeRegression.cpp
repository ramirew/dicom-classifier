/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ctime>

#include "random_forest/utility.h"
#include "TreeRegression.h"
#include "random_forest/Data.h"

namespace ranger {

// Constructor de la clase TreeRegression
TreeRegression::TreeRegression(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
    std::vector<double>& split_values) :
    Tree(child_nodeIDs, split_varIDs, split_values), counter(0), sums(0) {
}

// Método para asignar memoria
void TreeRegression::allocateMemory() {
  // Inicializar contadores si no se está en modo de ahorro de memoria
  if (!memory_saving_splitting) {
    size_t max_num_splits = data->getMaxNumUniqueValues();

    // Usar el número de divisiones aleatorias para extratrees
    if (splitrule == EXTRATREES && num_random_splits > max_num_splits) {
      max_num_splits = num_random_splits;
    }

    counter.resize(max_num_splits);
    sums.resize(max_num_splits);
  }
}

// Método para estimar el valor en un nodo
double TreeRegression::estimate(size_t nodeID) {
  // Promedio de las respuestas de las muestras en el nodo
  double sum_responses_in_node = 0;
  size_t num_samples_in_node = end_pos[nodeID] - start_pos[nodeID];
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    sum_responses_in_node += data->get_y(sampleID, 0);
  }
  return (sum_responses_in_node / (double) num_samples_in_node);
}

// Método para agregar información al archivo internamente (sin cobertura)
void TreeRegression::appendToFileInternal(std::ofstream& file) { // #nocov start
  // Vacío a propósito
} // #nocov end

// Método para dividir un nodo internamente
bool TreeRegression::splitNodeInternal(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  // Detener si se alcanza el tamaño máximo del nodo o la profundidad
  if (num_samples_node <= min_node_size || (nodeID >= last_left_nodeID && max_depth > 0 && depth >= max_depth)) {
    split_values[nodeID] = estimate(nodeID);
    return true;
  }

  // Verificar si el nodo es puro y establecer split_value para estimar y detener si es puro
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

  // Encontrar la mejor división, detener si no hay disminución de impureza
  bool stop;
  if (splitrule == MAXSTAT) {
    stop = findBestSplitMaxstat(nodeID, possible_split_varIDs);
  } else if (splitrule == EXTRATREES) {
    stop = findBestSplitExtraTrees(nodeID, possible_split_varIDs);
  } else if (splitrule == BETA) {
    stop = findBestSplitBeta(nodeID, possible_split_varIDs);
  } else {
    stop = findBestSplit(nodeID, possible_split_varIDs);
  }

  if (stop) {
    split_values[nodeID] = estimate(nodeID);
    return true;
  }

  return false;
}

// Método para crear un nodo vacío internamente
void TreeRegression::createEmptyNodeInternal() {
  // Vacío a propósito
}

// Método para calcular la precisión de la predicción internamente
double TreeRegression::computePredictionAccuracyInternal(std::vector<double>* prediction_error_casewise) {
  size_t num_predictions = prediction_terminal_nodeIDs.size();
  double sum_of_squares = 0;
  for (size_t i = 0; i < num_predictions; ++i) {
    size_t terminal_nodeID = prediction_terminal_nodeIDs[i];
    double predicted_value = split_values[terminal_nodeID];
    double real_value = data->get_y(oob_sampleIDs[i], 0);
    if (predicted_value != real_value) {
      double diff = (predicted_value - real_value) * (predicted_value - real_value);
      if (prediction_error_casewise) {
        (*prediction_error_casewise)[i] = diff;
      }
      sum_of_squares += diff;
    }
  }
  return (1.0 - sum_of_squares / (double) num_predictions);
}

// Método para encontrar la mejor división
bool TreeRegression::findBestSplit(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  double best_decrease = -1;
  size_t best_varID = 0;
  double best_value = 0;

  // Calcular la suma de respuestas en el nodo
  double sum_node = 0;
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    sum_node += data->get_y(sampleID, 0);
  }

  // Para todas las variables de división posibles
  for (auto& varID : possible_split_varIDs) {
    // Encontrar el mejor valor de división, si es ordenado considerar todos los valores como valores de división, de lo contrario, todas las particiones de 2
    if (data->isOrderedVariable(varID)) {
      // Usar método de ahorro de memoria si la opción está establecida
      if (memory_saving_splitting) {
        findBestSplitValueSmallQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
      } else {
        // Usar método más rápido para ambos casos
        double q = (double) num_samples_node / (double) data->getNumUniqueDataValues(varID);
        if (q < Q_THRESHOLD) {
          findBestSplitValueSmallQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
        } else {
          findBestSplitValueLargeQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
        }
      }
    } else {
      findBestSplitValueUnordered(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
    }
  }

  // Detener si no se encuentra una buena división
  if (best_decrease < 0) {
    return true;
  }

  // Guardar los mejores valores
  split_varIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

  // Calcular la disminución de la impureza para este nodo y agregar a la importancia de la variable si es necesario
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    addImpurityImportance(nodeID, best_varID, best_decrease);
  }

  // Regularización
  saveSplitVarID(best_varID);

  return false;
}

// Método para encontrar el mejor valor de división para pequeños valores de Q
void TreeRegression::findBestSplitValueSmallQ(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {
  // Crear posibles valores de división
  std::vector<double> possible_split_values;
  data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Intentar con la siguiente variable si todos son iguales para esta
  if (possible_split_values.size() < 2) {
    return;
  }

  const size_t num_splits = possible_split_values.size();
  if (memory_saving_splitting) {
    std::vector<double> sums_right(num_splits);
    std::vector<size_t> n_right(num_splits);
    findBestSplitValueSmallQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease,
        possible_split_values, sums_right, n_right);
  } else {
    std::fill_n(sums.begin(), num_splits, 0);
    std::fill_n(counter.begin(), num_splits, 0);
    findBestSplitValueSmallQ(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease,
        possible_split_values, sums, counter);
  }
}

// Helper para encontrar el mejor valor de división para pequeños valores de Q
void TreeRegression::findBestSplitValueSmallQ(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease, std::vector<double> possible_split_values,
    std::vector<double>& sums, std::vector<size_t>& counter) {

  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    size_t idx = std::lower_bound(possible_split_values.begin(), possible_split_values.end(),
        data->get_x(sampleID, varID)) - possible_split_values.begin();

    sums[idx] += data->get_y(sampleID, 0);
    ++counter[idx];
  }

  size_t n_left = 0;
  double sum_left = 0;

  // Calcular la disminución de la impureza para cada división
  for (size_t i = 0; i < possible_split_values.size() - 1; ++i) {
    // Detener si no hay nada aquí
    if (counter[i] == 0) {
      continue;
    }

    n_left += counter[i];
    sum_left += sums[i];

    // Detener si el hijo derecho está vacío
    size_t n_right = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    double sum_right = sum_node - sum_left;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    // Regularización
    regularize(decrease, varID);

    // Si es mejor que antes, usar esto
    if (decrease > best_decrease) {
      // Usar división de punto medio
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

// Método para encontrar el mejor valor de división para grandes valores de Q
void TreeRegression::findBestSplitValueLargeQ(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {
  // Establecer contadores a 0
  size_t num_unique = data->getNumUniqueDataValues(varID);
  std::fill_n(counter.begin(), num_unique, 0);
  std::fill_n(sums.begin(), num_unique, 0);

  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    size_t index = data->getIndex(sampleID, varID);

    sums[index] += data->get_y(sampleID, 0);
    ++counter[index];
  }

  size_t n_left = 0;
  double sum_left = 0;

  // Calcular la disminución de la impureza para cada división
  for (size_t i = 0; i < num_unique - 1; ++i) {
    // Detener si no hay nada aquí
    if (counter[i] == 0) {
      continue;
    }

    n_left += counter[i];
    sum_left += sums[i];

    // Detener si el hijo derecho está vacío
    size_t n_right = num_samples_node - n_left;
    if (n_right == 0) {
      break;
    }

    double sum_right = sum_node - sum_left;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

    // Regularización
    regularize(decrease, varID);

    // Si es mejor que antes, usar esto
    if (decrease > best_decrease) {
      // Encontrar el siguiente valor en este nodo
      size_t j = i + 1;
      while (j < num_unique && counter[j] == 0) {
        ++j;
      }

      // Usar división de punto medio
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

// Método para encontrar el mejor valor de división para variables no ordenadas
void TreeRegression::findBestSplitValueUnordered(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {
  // Crear posibles valores de división
  std::vector<double> factor_levels;
  data->getAllValues(factor_levels, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Intentar con la siguiente variable si todos son iguales para esta
  if (factor_levels.size() < 2) {
    return;
  }

  // El número de posibles divisiones es 2^num_levels
  size_t num_splits = (1ULL << factor_levels.size());

  // Calcular la disminución de la impureza para cada posible división
  // Dividir donde todo a la izquierda (0) o todo a la derecha (1) están excluidos
  // La segunda mitad de los números es solo izquierda/derecha invertida en la primera mitad -> Excluir la segunda mitad
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
    double sum_right = 0;
    size_t n_right = 0;

    // Suma en el hijo derecho
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      double response = data->get_y(sampleID, 0);
      double value = data->get_x(sampleID, varID);
      size_t factorID = floor(value) - 1;

      // Si está en el hijo derecho, contar
      // En el hijo derecho, si el splitID en posición factorID es 1
      if ((splitID & (1ULL << factorID))) {
        ++n_right;
        sum_right += response;
      }
    }
    size_t n_left = num_samples_node - n_right;

    // Suma de cuadrados
    double sum_left = sum_node - sum_right;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

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

// Método para encontrar la mejor división utilizando Maxstat
bool TreeRegression::findBestSplitMaxstat(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];

  // Calcular rangos
  std::vector<double> response;
  response.reserve(num_samples_node);
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    response.push_back(data->get_y(sampleID, 0));
  }
  std::vector<double> ranks = rank(response);

  // Guardar estadísticas de división
  std::vector<double> pvalues;
  pvalues.reserve(possible_split_varIDs.size());
  std::vector<double> values;
  values.reserve(possible_split_varIDs.size());
  std::vector<double> candidate_varIDs;
  candidate_varIDs.reserve(possible_split_varIDs.size());
  std::vector<double> test_statistics;
  test_statistics.reserve(possible_split_varIDs.size());

  // Calcular p-valores
  for (auto& varID : possible_split_varIDs) {
    // Obtener todas las observaciones
    std::vector<double> x;
    x.reserve(num_samples_node);
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      x.push_back(data->get_x(sampleID, varID));
    }

    // Ordenar por x
    std::vector<size_t> indices = order(x, false);

    // Calcular estadísticas de rango seleccionadas max
    double best_maxstat;
    double best_split_value;
    maxstat(ranks, x, indices, best_maxstat, best_split_value, minprop, 1 - minprop);

    if (best_maxstat > -1) {
      // Calcular el número de muestras a la izquierda de los puntos de corte
      std::vector<size_t> num_samples_left = numSamplesLeftOfCutpoint(x, indices);

      // Calcular p-valores
      double pvalue_lau92 = maxstatPValueLau92(best_maxstat, minprop, 1 - minprop);
      double pvalue_lau94 = maxstatPValueLau94(best_maxstat, minprop, 1 - minprop, num_samples_node, num_samples_left);

      // Usar el mínimo de Lau92 y Lau94
      double pvalue = std::min(pvalue_lau92, pvalue_lau94);

      // Guardar estadísticas de división
      pvalues.push_back(pvalue);
      values.push_back(best_split_value);
      candidate_varIDs.push_back(varID);
      test_statistics.push_back(best_maxstat);
    }
  }

  double adjusted_best_pvalue = std::numeric_limits<double>::max();
  size_t best_varID = 0;
  double best_value = 0;
  double best_maxstat = 0;

  if (pvalues.size() > 0) {
    // Ajustar p-valores con Benjamini/Hochberg
    std::vector<double> adjusted_pvalues = adjustPvalues(pvalues);

    // Usar el p-valor más pequeño
    double min_pvalue = std::numeric_limits<double>::max();
    for (size_t i = 0; i < pvalues.size(); ++i) {
      if (pvalues[i] < min_pvalue) {
        min_pvalue = pvalues[i];
        best_varID = candidate_varIDs[i];
        best_value = values[i];
        adjusted_best_pvalue = adjusted_pvalues[i];
        best_maxstat = test_statistics[i];
      }
    }
  }

  // Detener si no se encuentra una buena división (este es el nodo terminal).
  if (adjusted_best_pvalue > alpha) {
    return true;
  } else {
    // Si no es un nodo terminal, guardar los mejores valores
    split_varIDs[nodeID] = best_varID;
    split_values[nodeID] = best_value;

    // Calcular la disminución de la impureza para este nodo y agregar a la importancia de la variable si es necesario
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
      addImpurityImportance(nodeID, best_varID, best_maxstat);
    }

    return false;
  }
}

// Método para encontrar la mejor división utilizando Extra Trees
bool TreeRegression::findBestSplitExtraTrees(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  double best_decrease = -1;
  size_t best_varID = 0;
  double best_value = 0;

  // Calcular la suma de respuestas en el nodo
  double sum_node = 0;
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    sum_node += data->get_y(sampleID, 0);
  }

  // Para todas las variables de división posibles
  for (auto& varID : possible_split_varIDs) {
    // Encontrar el mejor valor de división, si es ordenado considerar todos los valores como valores de división, de lo contrario, todas las particiones de 2
    if (data->isOrderedVariable(varID)) {
      findBestSplitValueExtraTrees(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
    } else {
      findBestSplitValueExtraTreesUnordered(nodeID, varID, sum_node, num_samples_node, best_value, best_varID,
          best_decrease);
    }
  }

  // Detener si no se encuentra una buena división
  if (best_decrease < 0) {
    return true;
  }

  // Guardar los mejores valores
  split_varIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

  // Calcular la disminución de la impureza para este nodo y agregar a la importancia de la variable si es necesario
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    addImpurityImportance(nodeID, best_varID, best_decrease);
  }

  // Regularización
  saveSplitVarID(best_varID);

  return false;
}

// Método para encontrar el mejor valor de división para Extra Trees
void TreeRegression::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {
  // Obtener valores mínimos/máximos de la covariable en el nodo
  double min;
  double max;
  data->getMinMaxValues(min, max, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Intentar con la siguiente variable si todos son iguales para esta
  if (min == max) {
    return;
  }

  // Crear posibles valores de división: dibujar aleatoriamente entre min y max
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
    std::vector<double> sums_right(num_splits);
    std::vector<size_t> n_right(num_splits);
    findBestSplitValueExtraTrees(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease,
        possible_split_values, sums_right, n_right);
  } else {
    std::fill_n(sums.begin(), num_splits, 0);
    std::fill_n(counter.begin(), num_splits, 0);
    findBestSplitValueExtraTrees(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease,
        possible_split_values, sums, counter);
  }
}

// Helper para encontrar el mejor valor de división para Extra Trees
void TreeRegression::findBestSplitValueExtraTrees(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease, std::vector<double> possible_split_values,
    std::vector<double>& sums_right, std::vector<size_t>& n_right) {
  const size_t num_splits = possible_split_values.size();

  // Suma en el hijo derecho y posible división
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    double value = data->get_x(sampleID, varID);
    double response = data->get_y(sampleID, 0);

    // Contar muestras hasta que se alcance el valor de división
    for (size_t i = 0; i < num_splits; ++i) {
      if (value > possible_split_values[i]) {
        ++n_right[i];
        sums_right[i] += response;
      } else {
        break;
      }
    }
  }

  // Calcular la disminución de la impureza para cada posible división
  for (size_t i = 0; i < num_splits; ++i) {
    // Detener si un hijo está vacío
    size_t n_left = num_samples_node - n_right[i];
    if (n_left == 0 || n_right[i] == 0) {
      continue;
    }

    double sum_right = sums_right[i];
    double sum_left = sum_node - sum_right;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right[i];

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

// Método para encontrar el mejor valor de división no ordenado para Extra Trees
void TreeRegression::findBestSplitValueExtraTreesUnordered(size_t nodeID, size_t varID, double sum_node,
    size_t num_samples_node, double& best_value, size_t& best_varID, double& best_decrease) {
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

    // Asignar la unión de los dos subconjuntos al hijo derecho
    size_t splitID = 0;
    for (auto& idx : split_subset) {
      splitID |= 1ULL << idx;
    }

    // Inicializar
    double sum_right = 0;
    size_t n_right = 0;

    // Suma en el hijo derecho
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      double response = data->get_y(sampleID, 0);
      double value = data->get_x(sampleID, varID);
      size_t factorID = floor(value) - 1;

      // Si está en el hijo derecho, contar
      // En el hijo derecho, si el splitID en posición factorID es 1
      if ((splitID & (1ULL << factorID))) {
        ++n_right;
        sum_right += response;
      }
    }
    size_t n_left = num_samples_node - n_right;

    // Suma de cuadrados
    double sum_left = sum_node - sum_right;
    double decrease = sum_left * sum_left / (double) n_left + sum_right * sum_right / (double) n_right;

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

// Método para encontrar la mejor división utilizando Beta
bool TreeRegression::findBestSplitBeta(size_t nodeID, std::vector<size_t>& possible_split_varIDs) {
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  double best_decrease = -std::numeric_limits<double>::infinity();
  size_t best_varID = 0;
  double best_value = 0;

  // Calcular la suma de respuestas en el nodo
  double sum_node = 0;
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    sum_node += data->get_y(sampleID, 0);
  }

  // Para todas las variables de división posibles, encontrar el mejor valor de división
  for (auto& varID : possible_split_varIDs) {
    findBestSplitValueBeta(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease);
  }

  // Detener si no se encuentra una buena división
  if (std::isinf(-best_decrease)) {
    return true;
  }

  // Guardar los mejores valores
  split_varIDs[nodeID] = best_varID;
  split_values[nodeID] = best_value;

  // Calcular la disminución de la impureza para este nodo y agregar a la importancia de la variable si es necesario
  if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
    addImpurityImportance(nodeID, best_varID, best_decrease);
  }

  // Regularización
  saveSplitVarID(best_varID);

  return false;
}

// Método para encontrar el mejor valor de división para Beta
void TreeRegression::findBestSplitValueBeta(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease) {
  // Crear posibles valores de división
  std::vector<double> possible_split_values;
  data->getAllValues(possible_split_values, sampleIDs, varID, start_pos[nodeID], end_pos[nodeID]);

  // Intentar con la siguiente variable si todos son iguales para esta
  if (possible_split_values.size() < 2) {
    return;
  }

  // -1 porque no es posible dividir en el valor más grande
  size_t num_splits = possible_split_values.size() - 1;
  if (memory_saving_splitting) {
    std::vector<double> sums_right(num_splits);
    std::vector<size_t> n_right(num_splits);
    findBestSplitValueBeta(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease,
        possible_split_values, sums_right, n_right);
  } else {
    std::fill_n(sums.begin(), num_splits, 0);
    std::fill_n(counter.begin(), num_splits, 0);
    findBestSplitValueBeta(nodeID, varID, sum_node, num_samples_node, best_value, best_varID, best_decrease,
        possible_split_values, sums, counter);
  }
}

// Helper para encontrar el mejor valor de división para Beta
void TreeRegression::findBestSplitValueBeta(size_t nodeID, size_t varID, double sum_node, size_t num_samples_node,
    double& best_value, size_t& best_varID, double& best_decrease, std::vector<double> possible_split_values,
    std::vector<double>& sums_right, std::vector<size_t>& n_right) {
  // -1 porque no es posible dividir en el valor más grande
  const size_t num_splits = possible_split_values.size() - 1;

  // Suma en el hijo derecho y posible división
  for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
    size_t sampleID = sampleIDs[pos];
    double value = data->get_x(sampleID, varID);
    double response = data->get_y(sampleID, 0);

    // Contar muestras hasta que se alcance el valor de división
    for (size_t i = 0; i < num_splits; ++i) {
      if (value > possible_split_values[i]) {
        ++n_right[i];
        sums_right[i] += response;
      } else {
        break;
      }
    }
  }

  // Calcular la LogLik de la distribución beta para cada posible división
  for (size_t i = 0; i < num_splits; ++i) {
    // Detener si un hijo es demasiado pequeño
    size_t n_left = num_samples_node - n_right[i];
    if (n_left < 2 || n_right[i] < 2) {
      continue;
    }

    // Calcular la media
    double sum_right = sums_right[i];
    double mean_right = sum_right / (double) n_right[i];
    double sum_left = sum_node - sum_right;
    double mean_left = sum_left / (double) n_left;

    // Calcular la varianza
    double var_right = 0;
    double var_left = 0;
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      double value = data->get_x(sampleID, varID);
      double response = data->get_y(sampleID, 0);

      if (value > possible_split_values[i]) {
        var_right += (response - mean_right) * (response - mean_right);
      } else {
        var_left += (response - mean_left) * (response - mean_left);
      }
    }
    var_right /= (double) n_right[i] - 1;
    var_left /= (double) n_left - 1;

    // Detener si la varianza es cero
    if (var_right < std::numeric_limits<double>::epsilon() || var_left < std::numeric_limits<double>::epsilon()) {
      continue;
    }

    // Calcular phi para la distribución beta
    double phi_right = mean_right * (1 - mean_right) / var_right - 1;
    double phi_left = mean_left * (1 - mean_left) / var_left - 1;

    // Calcular la LogLik de la distribución beta
    double beta_loglik_right = 0;
    double beta_loglik_left = 0;
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      double value = data->get_x(sampleID, varID);
      double response = data->get_y(sampleID, 0);

      if (value > possible_split_values[i]) {
        beta_loglik_right += betaLogLik(response, mean_right, phi_right);
      } else {
        beta_loglik_left += betaLogLik(response, mean_left, phi_left);
      }
    }

    // La estadística de división es la suma de ambos log-likelihoods
    double decrease = beta_loglik_right + beta_loglik_left;

    // Detener si no hay resultado
    if (std::isnan(decrease)) {
      continue;
    }

    // Regularización (valores negativos)
    regularizeNegative(decrease, varID);

    // Si es mejor que antes, usar esto
    if (decrease > best_decrease) {
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

// Método para agregar importancia de la impureza
void TreeRegression::addImpurityImportance(size_t nodeID, size_t varID, double decrease) {
  size_t num_samples_node = end_pos[nodeID] - start_pos[nodeID];
  double best_decrease = decrease;
  if (splitrule != MAXSTAT) {
    double sum_node = 0;
    for (size_t pos = start_pos[nodeID]; pos < end_pos[nodeID]; ++pos) {
      size_t sampleID = sampleIDs[pos];
      sum_node += data->get_y(sampleID, 0);
    }

    double impurity_node = (sum_node * sum_node / (double) num_samples_node);

    // Considerar la regularización
    regularize(impurity_node, varID);

    best_decrease = decrease - impurity_node;
  }

  // No hay importancia de la variable para variables sin división
  size_t tempvarID = data->getUnpermutedVarID(varID);

  // Restar si la importancia es corregida y la variable está permutada, de lo contrario, sumar
  if (importance_mode == IMP_GINI_CORRECTED && varID >= data->getNumCols()) {
    (*variable_importance)[tempvarID] -= best_decrease;
  } else {
    (*variable_importance)[tempvarID] += best_decrease;
  }
}

} // namespace ranger

