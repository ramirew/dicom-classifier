/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <iterator>
#include <memory> // Incluimos la librería <memory> para usar punteros inteligentes

#include "Tree.h"
#include "random_forest/utility.h"

namespace ranger {

// Constructor por defecto
Tree::Tree() :
    mtry(0), num_samples(0), num_samples_oob(0), min_node_size(0), deterministic_varIDs(0), split_select_weights(0), case_weights(
          0), manual_inbag(0), oob_sampleIDs(0), holdout(false), keep_inbag(false), data(nullptr), regularization_factor(0), regularization_usedepth(
          false), split_varIDs_used(0), variable_importance(0), importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(
          true), sample_fraction(0), memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(
          DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(
          0) {
}

// Constructor con parámetros para inicialización con nodos hijos, variables de división y valores de división
Tree::Tree(std::vector<std::vector<size_t>>& child_nodeIDs, std::vector<size_t>& split_varIDs,
           std::vector<double>& split_values) :
    mtry(0), num_samples(0), num_samples_oob(0), min_node_size(0), deterministic_varIDs(0), split_select_weights(0), case_weights(
          0), manual_inbag(0), split_varIDs(split_varIDs), split_values(split_values), child_nodeIDs(child_nodeIDs), oob_sampleIDs(
          0), holdout(false), keep_inbag(false), data(nullptr), regularization_factor(0), regularization_usedepth(false), split_varIDs_used(
          0), variable_importance(0), importance_mode(DEFAULT_IMPORTANCE_MODE), sample_with_replacement(true), sample_fraction(
          0), memory_saving_splitting(false), splitrule(DEFAULT_SPLITRULE), alpha(DEFAULT_ALPHA), minprop(
          DEFAULT_MINPROP), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(DEFAULT_MAXDEPTH), depth(0), last_left_nodeID(
          0) {
}

// Inicialización del árbol con datos y parámetros
void Tree::init(const Data* data, uint mtry, size_t num_samples, uint seed, std::vector<size_t>* deterministic_varIDs,
                std::vector<double>* split_select_weights, ImportanceMode importance_mode, uint min_node_size,
                bool sample_with_replacement, bool memory_saving_splitting, SplitRule splitrule, std::vector<double>* case_weights,
                std::vector<size_t>* manual_inbag, bool keep_inbag, std::vector<double>* sample_fraction, double alpha,
                double minprop, bool holdout, uint num_random_splits, uint max_depth, std::vector<double>* regularization_factor,
                bool regularization_usedepth, std::vector<bool>* split_varIDs_used) {

    this->data = data;
    this->mtry = mtry;
    this->num_samples = num_samples;
    this->memory_saving_splitting = memory_saving_splitting;

    // Crear nodo raíz, asignar muestras bootstrap y muestras OOB
    child_nodeIDs.push_back(std::vector<size_t>());
    child_nodeIDs.push_back(std::vector<size_t>());
    createEmptyNode();

    // Inicializar generador de números aleatorios y establecer semilla
    random_number_generator.seed(seed);

    this->deterministic_varIDs = deterministic_varIDs;
    this->split_select_weights = split_select_weights;
    this->importance_mode = importance_mode;
    this->min_node_size = min_node_size;
    this->sample_with_replacement = sample_with_replacement;
    this->splitrule = splitrule;
    this->case_weights = case_weights;
    this->manual_inbag = manual_inbag;
    this->keep_inbag = keep_inbag;
    this->sample_fraction = sample_fraction;
    this->holdout = holdout;
    this->alpha = alpha;
    this->minprop = minprop;
    this->num_random_splits = num_random_splits;
    this->max_depth = max_depth;
    this->regularization_factor = regularization_factor;
    this->regularization_usedepth = regularization_usedepth;
    this->split_varIDs_used = split_varIDs_used;

    // Regularización
    if (regularization_factor->size() > 0) {
        regularization = true;
    } else {
        regularization = false;
    }
}

// Función para crecer el árbol
void Tree::grow(std::vector<double>* variable_importance) {
    // Asignar memoria para el crecimiento del árbol
    allocateMemory();

    this->variable_importance = variable_importance;

    // Bootstrap, dependiente de si es ponderado o no y con o sin reemplazo
    if (!case_weights->empty()) {
        if (sample_with_replacement) {
            bootstrapWeighted();
        } else {
            bootstrapWithoutReplacementWeighted();
        }
    } else if (sample_fraction->size() > 1) {
        if (sample_with_replacement) {
            bootstrapClassWise();
        } else {
            bootstrapWithoutReplacementClassWise();
        }
    } else if (!manual_inbag->empty()) {
        setManualInbag();
    } else {
        if (sample_with_replacement) {
            bootstrap();
        } else {
            bootstrapWithoutReplacement();
        }
    }

    // Inicializar posiciones de inicio y fin
    start_pos[0] = 0;
    end_pos[0] = sampleIDs.size();

    // Mientras no todos los nodos sean terminales, dividir el siguiente nodo
    size_t num_open_nodes = 1;
    size_t i = 0;
    depth = 0;
    while (num_open_nodes > 0) {
        // Dividir nodo
        bool is_terminal_node = splitNode(i);
        if (is_terminal_node) {
            --num_open_nodes;
        } else {
            ++num_open_nodes;
            if (i >= last_left_nodeID) {
                // Si es un nuevo nivel, aumentar la profundidad
                last_left_nodeID = split_varIDs.size() - 2;
                ++depth;
            }
        }
        ++i;
    }

    // Borrar vector de sampleID para ahorrar memoria
    sampleIDs.clear();
    sampleIDs.shrink_to_fit();
    cleanUpInternal();
}

// Función para realizar predicciones con el árbol
void Tree::predict(const Data* prediction_data, bool oob_prediction) {
    size_t num_samples_predict;
    if (oob_prediction) {
        num_samples_predict = num_samples_oob;
    } else {
        num_samples_predict = prediction_data->getNumRows();
    }

    prediction_terminal_nodeIDs.resize(num_samples_predict, 0);

    // Para cada muestra, iniciar en la raíz, descender por el árbol y devolver el valor final
    for (size_t i = 0; i < num_samples_predict; ++i) {
        size_t sample_idx;
        if (oob_prediction) {
            sample_idx = oob_sampleIDs[i];
        } else {
            sample_idx = i;
        }
        size_t nodeID = 0;
        while (1) {
            // Romper si es nodo terminal
            if (child_nodeIDs[0][nodeID] == 0 && child_nodeIDs[1][nodeID] == 0) {
                break;
            }

            // Mover al hijo
            size_t split_varID = split_varIDs[nodeID];
            double value = prediction_data->get_x(sample_idx, split_varID);
            if (prediction_data->isOrderedVariable(split_varID)) {
                if (value <= split_values[nodeID]) {
                    // Mover al hijo izquierdo
                    nodeID = child_nodeIDs[0][nodeID];
                } else {
                    // Mover al hijo derecho
                    nodeID = child_nodeIDs[1][nodeID];
                }
            } else {
                size_t factorID = floor(value) - 1;
                size_t splitID = floor(split_values[nodeID]);
                if (!(splitID & (1ULL << factorID))) {
                    // Mover al hijo izquierdo
                    nodeID = child_nodeIDs[0][nodeID];
                } else {
                    // Mover al hijo derecho
                    nodeID = child_nodeIDs[1][nodeID];
                }
            }
        }
        prediction_terminal_nodeIDs[i] = nodeID;
    }
}

// Función para calcular la importancia de las variables mediante permutaciones
void Tree::computePermutationImportance(std::vector<double>& forest_importance, std::vector<double>& forest_variance,
                                        std::vector<double>& forest_importance_casewise) {
    size_t num_independent_variables = data->getNumCols();

    // Calcular precisión de predicción normal para cada árbol
    double accuracy_normal;
    std::vector<double> prederr_normal_casewise;
    std::vector<double> prederr_shuf_casewise;
    if (importance_mode == IMP_PERM_CASEWISE) {
        prederr_normal_casewise.resize(num_samples_oob, 0);
        prederr_shuf_casewise.resize(num_samples_oob, 0);
        accuracy_normal = computePredictionAccuracyInternal(&prederr_normal_casewise);
    } else {
        accuracy_normal = computePredictionAccuracyInternal(nullptr);
    }

    prediction_terminal_nodeIDs.clear();
    prediction_terminal_nodeIDs.resize(num_samples_oob, 0);

    // Reservar espacio para permutaciones, inicializar con oob_sampleIDs
    std::vector<size_t> permutations(oob_sampleIDs);

    // Permutar aleatoriamente para todas las variables independientes
    for (size_t i = 0; i < num_independent_variables; ++i) {
        // Permutar y calcular precisión de predicción nuevamente para esta permutación y guardar diferencia
        permuteAndPredictOobSamples(i, permutations);
        double accuracy_permuted;
        if (importance_mode == IMP_PERM_CASEWISE) {
            accuracy_permuted = computePredictionAccuracyInternal(&prederr_shuf_casewise);
            for (size_t j = 0; j < num_samples_oob; ++j) {
                size_t pos = i * num_samples + oob_sampleIDs[j];
                forest_importance_casewise[pos] += prederr_shuf_casewise[j] - prederr_normal_casewise[j];
            }
        } else {
            accuracy_permuted = computePredictionAccuracyInternal(nullptr);
        }

        double accuracy_difference = accuracy_normal - accuracy_permuted;
        forest_importance[i] += accuracy_difference;

        // Calcular varianza
        if (importance_mode == IMP_PERM_BREIMAN) {
            forest_variance[i] += accuracy_difference * accuracy_difference;
        } else if (importance_mode == IMP_PERM_LIAW) {
            forest_variance[i] += accuracy_difference * accuracy_difference * num_samples_oob;
        }
    }
}

// #nocov start
// Función para agregar al archivo
void Tree::appendToFile(std::ofstream& file) {
    // Guardar campos generales
    saveVector2D(child_nodeIDs, file);
    saveVector1D(split_varIDs, file);
    saveVector1D(split_values, file);

    // Llamar a funciones especiales para subclases para guardar campos especiales
    appendToFileInternal(file);
}
// #nocov end

// Crear subconjunto de variables posibles para división
void Tree::createPossibleSplitVarSubset(std::vector<size_t>& result) {
    size_t num_vars = data->getNumCols();

    // Para la importancia corregida de Gini agregar variables ficticias
    if (importance_mode == IMP_GINI_CORRECTED) {
        num_vars += data->getNumCols();
    }

    // Agregar variables no deterministas aleatoriamente (según los pesos si es necesario)
    if (split_select_weights->empty()) {
        if (deterministic_varIDs->empty()) {
            drawWithoutReplacement(result, random_number_generator, num_vars, mtry);
        } else {
            drawWithoutReplacementSkip(result, random_number_generator, num_vars, (*deterministic_varIDs), mtry);
        }
    } else {
        drawWithoutReplacementWeighted(result, random_number_generator, num_vars, mtry, *split_select_weights);
    }

    // Usar siempre variables deterministas
    std::copy(deterministic_varIDs->begin(), deterministic_varIDs->end(), std::inserter(result, result.end()));
}

// Función para dividir un nodo
bool Tree::splitNode(size_t nodeID) {
    // Seleccionar subconjunto aleatorio de variables para dividir
    std::vector<size_t> possible_split_varIDs;
    createPossibleSplitVarSubset(possible_split_varIDs);

    // Llamar al método de la subclase, establece split_varIDs y split_values
    bool stop = splitNodeInternal(nodeID, possible_split_varIDs);
    if (stop) {
        // Nodo terminal
        return true;
    }

    size_t split_varID = split_varIDs[nodeID];
    double split_value = split_values[nodeID];

    // Guardar variable no permutada para predicción
    split_varIDs[nodeID] = data->getUnpermutedVarID(split_varID);

    // Crear nodos hijos
    size_t left_child_nodeID = split_varIDs.size();
    child_nodeIDs[0][nodeID] = left_child_nodeID;
    createEmptyNode();
    start_pos[left_child_nodeID] = start_pos[nodeID];

    size_t right_child_nodeID = split_varIDs.size();
    child_nodeIDs[1][nodeID] = right_child_nodeID;
    createEmptyNode();
    start_pos[right_child_nodeID] = end_pos[nodeID];

    // Para cada muestra en el nodo, asignar al hijo izquierdo o derecho
    if (data->isOrderedVariable(split_varID)) {
        // Ordenado: izquierda es <= splitval y derecha es > splitval
        size_t pos = start_pos[nodeID];
        while (pos < start_pos[right_child_nodeID]) {
            size_t sampleID = sampleIDs[pos];
            if (data->get_x(sampleID, split_varID) <= split_value) {
                // Si va a la izquierda, no hacer nada
                ++pos;
            } else {
                // Si va a la derecha, mover al extremo derecho
                --start_pos[right_child_nodeID];
                std::swap(sampleIDs[pos], sampleIDs[start_pos[right_child_nodeID]]);
            }
        }
    } else {
        // No ordenado: si el bit en la posición es 1 -> derecha, 0 -> izquierda
        size_t pos = start_pos[nodeID];
        while (pos < start_pos[right_child_nodeID]) {
            size_t sampleID = sampleIDs[pos];
            double level = data->get_x(sampleID, split_varID);
            size_t factorID = floor(level) - 1;
            size_t splitID = floor(split_value);

            if (!(splitID & (1ULL << factorID))) {
                // Si va a la izquierda, no hacer nada
                ++pos;
            } else {
                // Si va a la derecha, mover al extremo derecho
                --start_pos[right_child_nodeID];
                std::swap(sampleIDs[pos], sampleIDs[start_pos[right_child_nodeID]]);
            }
        }
    }

    // La posición final del hijo izquierdo es la posición inicial del hijo derecho
    end_pos[left_child_nodeID] = start_pos[right_child_nodeID];
    end_pos[right_child_nodeID] = end_pos[nodeID];

    // No es nodo terminal
    return false;
}

// Función para crear un nodo vacío
void Tree::createEmptyNode() {
    split_varIDs.push_back(0);
    split_values.push_back(0);
    child_nodeIDs[0].push_back(0);
    child_nodeIDs[1].push_back(0);
    start_pos.push_back(0);
    end_pos.push_back(0);

    createEmptyNodeInternal();
}

// Función para descender una muestra permutada por el árbol
size_t Tree::dropDownSamplePermuted(size_t permuted_varID, size_t sampleID, size_t permuted_sampleID) {
    size_t nodeID = 0;
    while (child_nodeIDs[0][nodeID] != 0 || child_nodeIDs[1][nodeID] != 0) {
        size_t split_varID = split_varIDs[nodeID];
        size_t sampleID_final = sampleID;
        if (split_varID == permuted_varID) {
            sampleID_final = permuted_sampleID;
        }

        double value = data->get_x(sampleID_final, split_varID);
        if (data->isOrderedVariable(split_varID)) {
            if (value <= split_values[nodeID]) {
                nodeID = child_nodeIDs[0][nodeID];
            } else {
                nodeID = child_nodeIDs[1][nodeID];
            }
        } else {
            size_t factorID = floor(value) - 1;
            size_t splitID = floor(split_values[nodeID]);
            if (!(splitID & (1ULL << factorID))) {
                nodeID = child_nodeIDs[0][nodeID];
            } else {
                nodeID = child_nodeIDs[1][nodeID];
            }
        }
    }
    return nodeID;
}

// Función para permutar y predecir muestras OOB
void Tree::permuteAndPredictOobSamples(size_t permuted_varID, std::vector<size_t>& permutations) {
    std::shuffle(permutations.begin(), permutations.end(), random_number_generator);
    for (size_t i = 0; i < num_samples_oob; ++i) {
        size_t nodeID = dropDownSamplePermuted(permuted_varID, oob_sampleIDs[i], permutations[i]);
        prediction_terminal_nodeIDs[i] = nodeID;
    }
}

// Función para realizar bootstrap con reemplazo
void Tree::bootstrap() {
    size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];
    sampleIDs.reserve(num_samples_inbag);
    oob_sampleIDs.reserve(num_samples * (exp(-(*sample_fraction)[0]) + 0.1));
    std::uniform_int_distribution<size_t> unif_dist(0, num_samples - 1);

    inbag_counts.resize(num_samples, 0);
    for (size_t s = 0; s < num_samples_inbag; ++s) {
        size_t draw = unif_dist(random_number_generator);
        sampleIDs.push_back(draw);
        ++inbag_counts[draw];
    }

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

// Función para realizar bootstrap ponderado con reemplazo
void Tree::bootstrapWeighted() {
    size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];
    sampleIDs.reserve(num_samples_inbag);
    oob_sampleIDs.reserve(num_samples * (exp(-(*sample_fraction)[0]) + 0.1));
    std::discrete_distribution<> weighted_dist(case_weights->begin(), case_weights->end());

    inbag_counts.resize(num_samples, 0);
    for (size_t s = 0; s < num_samples_inbag; ++s) {
        size_t draw = weighted_dist(random_number_generator);
        sampleIDs.push_back(draw);
        ++inbag_counts[draw];
    }

    if (holdout) {
        for (size_t s = 0; s < (*case_weights).size(); ++s) {
            if ((*case_weights)[s] == 0) {
                oob_sampleIDs.push_back(s);
            }
        }
    } else {
        for (size_t s = 0; s < inbag_counts.size(); ++s) {
            if (inbag_counts[s] == 0) {
                oob_sampleIDs.push_back(s);
            }
        }
    }
    num_samples_oob = oob_sampleIDs.size();

    if (!keep_inbag) {
        inbag_counts.clear();
        inbag_counts.shrink_to_fit();
    }
}

// Función para realizar bootstrap sin reemplazo
void Tree::bootstrapWithoutReplacement() {
    size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];
    shuffleAndSplit(sampleIDs, oob_sampleIDs, num_samples, num_samples_inbag, random_number_generator);
    num_samples_oob = oob_sampleIDs.size();

    if (keep_inbag) {
        inbag_counts.resize(num_samples, 1);
        for (size_t i = 0; i < oob_sampleIDs.size(); i++) {
            inbag_counts[oob_sampleIDs[i]] = 0;
        }
    }
}

// Función para realizar bootstrap ponderado sin reemplazo
void Tree::bootstrapWithoutReplacementWeighted() {
    size_t num_samples_inbag = (size_t) num_samples * (*sample_fraction)[0];
    drawWithoutReplacementWeighted(sampleIDs, random_number_generator, num_samples - 1, num_samples_inbag, *case_weights);

    inbag_counts.resize(num_samples, 0);
    for (auto& sampleID : sampleIDs) {
        inbag_counts[sampleID] = 1;
    }

    if (holdout) {
        for (size_t s = 0; s < (*case_weights).size(); ++s) {
            if ((*case_weights)[s] == 0) {
                oob_sampleIDs.push_back(s);
            }
        }
    } else {
        for (size_t s = 0; s < inbag_counts.size(); ++s) {
            if (inbag_counts[s] == 0) {
                oob_sampleIDs.push_back(s);
            }
        }
    }
    num_samples_oob = oob_sampleIDs.size();

    if (!keep_inbag) {
        inbag_counts.clear();
        inbag_counts.shrink_to_fit();
    }
}

// Funciones virtuales que se implementan en subclases
void Tree::bootstrapClassWise() {}
void Tree::bootstrapWithoutReplacementClassWise() {}

// Función para establecer manualmente las muestras inbag
void Tree::setManualInbag() {
    sampleIDs.reserve(manual_inbag->size());
    inbag_counts.resize(num_samples, 0);
    for (size_t i = 0; i < manual_inbag->size(); ++i) {
        size_t inbag_count = (*manual_inbag)[i];
        if ((*manual_inbag)[i] > 0) {
            for (size_t j = 0; j < inbag_count; ++j) {
                sampleIDs.push_back(i);
            }
            inbag_counts[i] = inbag_count;
        } else {
            oob_sampleIDs.push_back(i);
        }
    }
    num_samples_oob = oob_sampleIDs.size();
    std::shuffle(sampleIDs.begin(), sampleIDs.end(), random_number_generator);

    if (!keep_inbag) {
        inbag_counts.clear();
        inbag_counts.shrink_to_fit();
    }
}

} // namespace ranger
