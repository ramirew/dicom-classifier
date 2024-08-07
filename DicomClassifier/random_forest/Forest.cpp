/*-------------------------------------------------------------------------------
 This file is parte of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <ctime>
#include <functional>
#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#endif
#include <memory> // Asegúrate de incluir <memory>

#include "random_forest/utility.h"
#include "Forest.h"
#include "random_forest/DataChar.h"
#include "random_forest/DataDouble.h"
#include "random_forest/DataFloat.h"

namespace ranger {

// Constructor de la clase Forest
Forest::Forest() :
    verbose_out(0), num_trees(DEFAULT_NUM_TREE), mtry(0), min_node_size(0), num_independent_variables(0), seed(0), num_samples(
          0), prediction_mode(false), memory_mode(MEM_DOUBLE), sample_with_replacement(true), memory_saving_splitting(
          false), splitrule(DEFAULT_SPLITRULE), predict_all(false), keep_inbag(false), sample_fraction( { 1 }), holdout(
          false), prediction_type(DEFAULT_PREDICTIONTYPE), num_random_splits(DEFAULT_NUM_RANDOM_SPLITS), max_depth(
          DEFAULT_MAXDEPTH), alpha(DEFAULT_ALPHA), minprop(DEFAULT_MINPROP), num_threads(DEFAULT_NUM_THREADS), data { }, overall_prediction_error(
          NAN), importance_mode(DEFAULT_IMPORTANCE_MODE), regularization_usedepth(false), progress(0) {
}

// #nocov start
// Inicializa el bosque para el modo C++
void Forest::initCpp(std::string dependent_variable_name, MemoryMode memory_mode, std::string input_file, uint mtry,
                     std::string output_prefix, uint num_trees, std::ostream* verbose_out, uint seed, uint num_threads,
                     std::string load_forest_filename, ImportanceMode importance_mode, uint min_node_size,
                     std::string split_select_weights_file, const std::vector<std::string>& always_split_variable_names,
                     std::string status_variable_name, bool sample_with_replacement,
                     const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
                     std::string case_weights_file, bool predict_all, double sample_fraction, double alpha, double minprop, bool holdout,
                     PredictionType prediction_type, uint num_random_splits, uint max_depth,
                     const std::vector<double>& regularization_factor, bool regularization_usedepth) {

    this->memory_mode = memory_mode;
    this->verbose_out = verbose_out;

    if (!dependent_variable_name.empty()) {
        if (status_variable_name.empty()) {
            this->dependent_variable_names = {dependent_variable_name};
        } else {
            this->dependent_variable_names = {dependent_variable_name, status_variable_name};
        }
    }

    // Establece el modo de predicción
    bool prediction_mode = false;
    if (!load_forest_filename.empty()) {
        prediction_mode = true;
    }

    // Fracción de muestra predeterminada y convierte a vector
    if (sample_fraction == 0) {
        if (sample_with_replacement) {
            sample_fraction = DEFAULT_SAMPLE_FRACTION_REPLACE;
        } else {
            sample_fraction = DEFAULT_SAMPLE_FRACTION_NOREPLACE;
        }
    }
    std::vector<double> sample_fraction_vector = { sample_fraction };

    if (prediction_mode) {
        loadDependentVariableNamesFromFile(load_forest_filename);
    }

    // Llama a otra función de inicialización
    init(loadDataFromFile(input_file), mtry, output_prefix, num_trees, seed, num_threads, importance_mode,
         min_node_size, prediction_mode, sample_with_replacement, unordered_variable_names, memory_saving_splitting,
         splitrule, predict_all, sample_fraction_vector, alpha, minprop, holdout, prediction_type, num_random_splits,
         false, max_depth, regularization_factor, regularization_usedepth);

    if (prediction_mode) {
        loadFromFile(load_forest_filename);
    }
    // Establece variables que siempre se considerarán para dividir
    if (!always_split_variable_names.empty()) {
        setAlwaysSplitVariables(always_split_variable_names);
    }

    // Carga los pesos de selección de división del archivo
    if (!split_select_weights_file.empty()) {
        std::vector<std::vector<double>> split_select_weights;
        split_select_weights.resize(1);
        loadDoubleVectorFromFile(split_select_weights[0], split_select_weights_file);
        if (split_select_weights[0].size() != num_independent_variables) {
            throw std::runtime_error("Number of split select weights is not equal to number of independent variables.");
        }
        setSplitWeightVector(split_select_weights);
    }

    // Carga los pesos de los casos del archivo
    if (!case_weights_file.empty()) {
        loadDoubleVectorFromFile(case_weights, case_weights_file);
        if (case_weights.size() != num_samples) {
            throw std::runtime_error("Number of case weights is not equal to number of samples.");
        }
    }

    // Muestra de pesos no cero en modo holdout
    if (holdout && !case_weights.empty()) {
        size_t nonzero_weights = 0;
        for (auto& weight : case_weights) {
            if (weight > 0) {
                ++nonzero_weights;
            }
        }
        this->sample_fraction[0] = this->sample_fraction[0] * ((double) nonzero_weights / (double) num_samples);
    }

    // Verifica si todas las variables categóricas están codificadas en enteros que comienzan en 1
    if (!unordered_variable_names.empty()) {
        std::string error_message = checkUnorderedVariables(*data, unordered_variable_names);
        if (!error_message.empty()) {
            throw std::runtime_error(error_message);
        }
    }
}
// #nocov end

// Inicializa el bosque para el modo R
void Forest::initR(std::unique_ptr<Data> input_data, uint mtry, uint num_trees, std::ostream* verbose_out, uint seed,
                   uint num_threads, ImportanceMode importance_mode, uint min_node_size,
                   std::vector<std::vector<double>>& split_select_weights, const std::vector<std::string>& always_split_variable_names,
                   bool prediction_mode, bool sample_with_replacement, const std::vector<std::string>& unordered_variable_names,
                   bool memory_saving_splitting, SplitRule splitrule, std::vector<double>& case_weights,
                   std::vector<std::vector<size_t>>& manual_inbag, bool predict_all, bool keep_inbag,
                   std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout, PredictionType prediction_type,
                   uint num_random_splits, bool order_snps, uint max_depth, const std::vector<double>& regularization_factor,
                   bool regularization_usedepth) {

    this->verbose_out = verbose_out;

    // Llama a otra función de inicialización
    init(std::move(input_data), mtry, "", num_trees, seed, num_threads, importance_mode, min_node_size,
         prediction_mode, sample_with_replacement, unordered_variable_names, memory_saving_splitting, splitrule,
         predict_all, sample_fraction, alpha, minprop, holdout, prediction_type, num_random_splits, order_snps, max_depth,
         regularization_factor, regularization_usedepth);

    // Establece variables que siempre se considerarán para dividir
    if (!always_split_variable_names.empty()) {
        setAlwaysSplitVariables(always_split_variable_names);
    }

    // Establece pesos de selección de división
    if (!split_select_weights.empty()) {
        setSplitWeightVector(split_select_weights);
    }

    // Establece pesos de los casos
    if (!case_weights.empty()) {
        if (case_weights.size() != num_samples) {
            throw std::runtime_error("Number of case weights not equal to number of samples.");
        }
        this->case_weights = case_weights;
    }

    // Establece inbag manual
    if (!manual_inbag.empty()) {
        this->manual_inbag = manual_inbag;
    }

    // Mantiene los conteos inbag
    this->keep_inbag = keep_inbag;
}

// Inicializa el bosque con los datos proporcionados
void Forest::init(std::unique_ptr<Data> input_data, uint mtry, std::string output_prefix,
                  uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode, uint min_node_size,
                  bool prediction_mode, bool sample_with_replacement, const std::vector<std::string>& unordered_variable_names,
                  bool memory_saving_splitting, SplitRule splitrule, bool predict_all, std::vector<double>& sample_fraction,
                  double alpha, double minprop, bool holdout, PredictionType prediction_type, uint num_random_splits, bool order_snps,
                  uint max_depth, const std::vector<double>& regularization_factor, bool regularization_usedepth) {

    // Inicializa los datos con memmode
    this->data = std::move(input_data);

    // Inicializa el generador de números aleatorios y establece la semilla
    if (seed == 0) {
        std::random_device random_device;
        random_number_generator.seed(random_device());
    } else {
        random_number_generator.seed(seed);
    }

    // Establece el número de hilos
    if (num_threads == DEFAULT_NUM_THREADS) {
#ifdef OLD_WIN_R_BUILD
        this->num_threads = 1;
#else
        this->num_threads = std::thread::hardware_concurrency();
#endif
    } else {
        this->num_threads = num_threads;
    }

    // Establece las variables miembro
    this->num_trees = num_trees;
    this->mtry = mtry;
    this->seed = seed;
    this->output_prefix = output_prefix;
    this->importance_mode = importance_mode;
    this->min_node_size = min_node_size;
    this->prediction_mode = prediction_mode;
    this->sample_with_replacement = sample_with_replacement;
    this->memory_saving_splitting = memory_saving_splitting;
    this->splitrule = splitrule;
    this->predict_all = predict_all;
    this->sample_fraction = sample_fraction;
    this->holdout = holdout;
    this->alpha = alpha;
    this->minprop = minprop;
    this->prediction_type = prediction_type;
    this->num_random_splits = num_random_splits;
    this->max_depth = max_depth;
    this->regularization_factor = regularization_factor;
    this->regularization_usedepth = regularization_usedepth;

    // Establece el número de muestras y variables
    num_samples = data->getNumRows();
    num_independent_variables = data->getNumCols();

    // Establece variables de factor no ordenadas
    if (!prediction_mode) {
        data->setIsOrderedVariable(unordered_variable_names);
    }

    initInternal();

    // Inicializa los pesos de selección de división
    split_select_weights.push_back(std::vector<double>());

    // Inicializa el inbag manual
    manual_inbag.push_back(std::vector<size_t>());

    // Verifica si mtry está en el rango válido
    if (this->mtry > num_independent_variables) {
        throw std::runtime_error("mtry can not be larger than number of variables in data.");
    }

    // Verifica si se muestrearon observaciones
    if ((size_t) num_samples * sample_fraction[0] < 1) {
        throw std::runtime_error("sample_fraction too small, no observations sampled.");
    }

    // Permuta muestras para la importancia de Gini corregida
    if (importance_mode == IMP_GINI_CORRECTED) {
        data->permuteSampleIDs(random_number_generator);
    }

    // Ordena los niveles SNP si está en modo de división "order"
    if (!prediction_mode && order_snps) {
        data->orderSnpLevels((importance_mode == IMP_GINI_CORRECTED));
    }

    // Regularización
    if (regularization_factor.size() > 0) {
        if (regularization_factor.size() == 1 && num_independent_variables > 1) {
            double single_regularization_factor = regularization_factor[0];
            this->regularization_factor.resize(num_independent_variables, single_regularization_factor);
        } else if (regularization_factor.size() != num_independent_variables) {
            throw std::runtime_error("Use 1 or p (the number of predictor variables) regularization factors.");
        }

        // Establece todas las variables como no utilizadas
        split_varIDs_used.resize(num_independent_variables, false);
    }
}

// Ejecuta el proceso de crecimiento o predicción del bosque
void Forest::run(bool verbose, bool compute_oob_error) {

    if (prediction_mode) {
        if (verbose && verbose_out) {
            *verbose_out << "Predicting .." << std::endl;
        }
        predict();
    } else {
        if (verbose && verbose_out) {
            *verbose_out << "Growing trees .." << std::endl;
        }

        grow();

        if (verbose && verbose_out) {
            *verbose_out << "Computing prediction error .." << std::endl;
        }

        if (compute_oob_error) {
            computePredictionError();
        }

        if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW || importance_mode == IMP_PERM_RAW
            || importance_mode == IMP_PERM_CASEWISE) {
            if (verbose && verbose_out) {
                *verbose_out << "Computing permutation variable importance .." << std::endl;
            }
            computePermutationImportance();
        }
    }
}

// #nocov start
// Escribe la salida en un archivo
void Forest::writeOutput() {

    if (verbose_out)
        *verbose_out << std::endl;
    writeOutputInternal();
    if (verbose_out) {
        if (dependent_variable_names.size() >= 1) {
            *verbose_out << "Dependent variable name:           " << dependent_variable_names[0] << std::endl;
        }
        *verbose_out << "Number of trees:                   " << num_trees << std::endl;
        *verbose_out << "Sample size:                       " << num_samples << std::endl;
        *verbose_out << "Number of independent variables:   " << num_independent_variables << std::endl;
        *verbose_out << "Mtry:                              " << mtry << std::endl;
        *verbose_out << "Target node size:                  " << min_node_size << std::endl;
        *verbose_out << "Variable importance mode:          " << importance_mode << std::endl;
        *verbose_out << "Memory mode:                       " << memory_mode << std::endl;
        *verbose_out << "Seed:                              " << seed << std::endl;
        *verbose_out << "Number of threads:                 " << num_threads << std::endl;
        *verbose_out << std::endl;
    }

    if (prediction_mode) {
        writePredictionFile();
    } else {
        if (verbose_out) {
            *verbose_out << "Overall OOB prediction error:      " << overall_prediction_error << std::endl;
            *verbose_out << std::endl;
        }

        if (!split_select_weights.empty() & !split_select_weights[0].empty()) {
            if (verbose_out) {
                *verbose_out
                    << "Warning: Split select weights used. Variable importance measures are only comparable for variables with equal weights."
                    << std::endl;
            }
        }

        if (importance_mode != IMP_NONE) {
            writeImportanceFile();
        }

        writeConfusionFile();
    }
}

// Escribe el archivo de importancia de variables
void Forest::writeImportanceFile() {

    // Abre el archivo de importancia para escribir
    std::string filename = output_prefix + ".importance";
    std::ofstream importance_file;
    importance_file.open(filename, std::ios::out);
    if (!importance_file.good()) {
        throw std::runtime_error("Could not write to importance file: " + filename + ".");
    }

    if (importance_mode == IMP_PERM_CASEWISE) {
        // Escribe los nombres de las variables
        for (auto& variable_name : data->getVariableNames()) {
            importance_file << variable_name << " ";
        }
        importance_file << std::endl;

        // Escribe los valores de importancia
        for (size_t i = 0; i < num_samples; ++i) {
            for (size_t j = 0; j < num_independent_variables; ++j) {
                if (variable_importance_casewise.size() <= (j * num_samples + i)) {
                    throw std::runtime_error("Memory error in local variable importance.");
                }
                importance_file << variable_importance_casewise[j * num_samples + i] << " ";
            }
            importance_file << std::endl;
        }
    } else {
        // Escribe la importancia en el archivo
        for (size_t i = 0; i < variable_importance.size(); ++i) {
            std::string variable_name = data->getVariableNames()[i];
            importance_file << variable_name << ": " << variable_importance[i] << std::endl;
        }
    }

    importance_file.close();
    if (verbose_out)
        *verbose_out << "Saved variable importance to file " << filename << "." << std::endl;
}

// Guarda el bosque en un archivo
void Forest::saveToFile() {

    // Abre el archivo para escribir
    std::string filename = output_prefix + ".forest";
    std::ofstream outfile;
    outfile.open(filename, std::ios::binary);
    if (!outfile.good()) {
        throw std::runtime_error("Could not write to output file: " + filename + ".");
    }

    // Escribe los nombres de las variables dependientes
    uint num_dependent_variables = dependent_variable_names.size();
    if (num_dependent_variables >= 1) {
        outfile.write(reinterpret_cast<char*>(&num_dependent_variables), sizeof(num_dependent_variables));
        for (auto& var_name : dependent_variable_names) {
            size_t length = var_name.size();
            outfile.write(reinterpret_cast<char*>(&length), sizeof(length));
            outfile.write(var_name.c_str(), length * sizeof(char));
        }
    } else {
        throw std::runtime_error("Missing dependent variable name.");
    }

    // Escribe el número de árboles
    outfile.write(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));

    // Escribe las variables ordenadas
    saveVector1D(data->getIsOrderedVariable(), outfile);

    saveToFileInternal(outfile);

    // Escribe los datos de los árboles para cada árbol
    for (auto& tree : trees) {
        tree->appendToFile(outfile);
    }

    // Cierra el archivo
    outfile.close();
    if (verbose_out)
        *verbose_out << "Saved forest to file " << filename << "." << std::endl;
}
// #nocov end

// Hace crecer el bosque
void Forest::grow() {

    // Crea rangos de hilos
    equalSplit(thread_ranges, 0, num_trees - 1, num_threads);

    // Llama a funciones especiales de crecimiento de subclases. Aquí se deben crear los árboles.
    growInternal();

    // Inicializa los árboles, crea una semilla para cada árbol, basada en la semilla principal
    std::uniform_int_distribution<uint> udist;
    for (size_t i = 0; i < num_trees; ++i) {
        uint tree_seed;
        if (seed == 0) {
            tree_seed = udist(random_number_generator);
        } else {
            tree_seed = (i + 1) * seed;
        }

        // Obtiene los pesos de selección de división para el árbol
        std::vector<double>* tree_split_select_weights;
        if (split_select_weights.size() > 1) {
            tree_split_select_weights = &split_select_weights[i];
        } else {
            tree_split_select_weights = &split_select_weights[0];
        }

        // Obtiene los conteos inbag para el árbol
        std::vector<size_t>* tree_manual_inbag;
        if (manual_inbag.size() > 1) {
            tree_manual_inbag = &manual_inbag[i];
        } else {
            tree_manual_inbag = &manual_inbag[0];
        }

        trees[i]->init(data.get(), mtry, num_samples, tree_seed, &deterministic_varIDs, tree_split_select_weights,
                       importance_mode, min_node_size, sample_with_replacement, memory_saving_splitting, splitrule, &case_weights,
                       tree_manual_inbag, keep_inbag, &sample_fraction, alpha, minprop, holdout, num_random_splits, max_depth,
                       &regularization_factor, regularization_usedepth, &split_varIDs_used);
    }

    // Inicializa la importancia de las variables
    variable_importance.resize(num_independent_variables, 0);

// Crece los árboles en múltiples hilos
#ifdef OLD_WIN_R_BUILD
    // #nocov start
    progress = 0;
    clock_t start_time = clock();
    clock_t lap_time = clock();
    for (size_t i = 0; i < num_trees; ++i) {
        trees[i]->grow(&variable_importance);
        progress++;
        showProgress("Growing trees..", start_time, lap_time);
    }
    // #nocov end
#else
    progress = 0;
#ifdef R_BUILD
    aborted = false;
    aborted_threads = 0;
#endif

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Inicializa la importancia por hilo
    std::vector<std::vector<double>> variable_importance_threads(num_threads);

    for (uint i = 0; i < num_threads; ++i) {
        if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
            variable_importance_threads[i].resize(num_independent_variables, 0);
        }
        threads.emplace_back(&Forest::growTreesInThread, this, i, &(variable_importance_threads[i]));
    }
    showProgress("Growing trees..", num_trees);
    for (auto &thread : threads) {
        thread.join();
    }

#ifdef R_BUILD
    if (aborted_threads > 0) {
        throw std::runtime_error("User interrupt.");
    }
#endif

    // Suma las importancias de los hilos
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
        variable_importance.resize(num_independent_variables, 0);
        for (size_t i = 0; i < num_independent_variables; ++i) {
            for (uint j = 0; j < num_threads; ++j) {
                variable_importance[i] += variable_importance_threads[j][i];
            }
        }
        variable_importance_threads.clear();
    }

#endif

    // Divide la importancia por el número de árboles
    if (importance_mode == IMP_GINI || importance_mode == IMP_GINI_CORRECTED) {
        for (auto& v : variable_importance) {
            v /= num_trees;
        }
    }
}

// Realiza predicciones usando el bosque
void Forest::predict() {

// Predice árboles en múltiples hilos y une los hilos con el hilo principal
#ifdef OLD_WIN_R_BUILD
    // #nocov start
    progress = 0;
    clock_t start_time = clock();
    clock_t lap_time = clock();
    for (size_t i = 0; i < num_trees; ++i) {
        trees[i]->predict(data.get(), false);
        progress++;
        showProgress("Predicting..", start_time, lap_time);
    }

    // Para todas las muestras obtiene las predicciones de los árboles
    allocatePredictMemory();
    for (size_t sample_idx = 0; sample_idx < data->getNumRows(); ++sample_idx) {
        predictInternal(sample_idx);
    }
    // #nocov end
#else
    progress = 0;
#ifdef R_BUILD
    aborted = false;
    aborted_threads = 0;
#endif

    // Predice
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint i = 0; i < num_threads; ++i) {
        threads.emplace_back(&Forest::predictTreesInThread, this, i, data.get(), false);
    }
    showProgress("Predicting..", num_trees);
    for (auto &thread : threads) {
        thread.join();
    }

    // Agrega predicciones
    allocatePredictMemory();
    threads.clear();
    threads.reserve(num_threads);
    progress = 0;
    for (uint i = 0; i < num_threads; ++i) {
        threads.emplace_back(&Forest::predictInternalInThread, this, i);
    }
    showProgress("Aggregating predictions..", num_samples);
    for (auto &thread : threads) {
        thread.join();
    }

#ifdef R_BUILD
    if (aborted_threads > 0) {
        throw std::runtime_error("User interrupt.");
    }
#endif
#endif
}

// Calcula el error de predicción
void Forest::computePredictionError() {

// Predice árboles en múltiples hilos
#ifdef OLD_WIN_R_BUILD
    // #nocov start
    progress = 0;
    clock_t start_time = clock();
    clock_t lap_time = clock();
    for (size_t i = 0; i < num_trees; ++i) {
        trees[i]->predict(data.get(), true);
        progress++;
        showProgress("Predicting..", start_time, lap_time);
    }
    // #nocov end
#else
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    progress = 0;
    for (uint i = 0; i < num_threads; ++i) {
        threads.emplace_back(&Forest::predictTreesInThread, this, i, data.get(), true);
    }
    showProgress("Computing prediction error..", num_trees);
    for (auto &thread : threads) {
        thread.join();
    }

#ifdef R_BUILD
    if (aborted_threads > 0) {
        throw std::runtime_error("User interrupt.");
    }
#endif
#endif

    // Llama a la función especial para subclases
    computePredictionErrorInternal();
}

// Calcula la importancia de la permutación
void Forest::computePermutationImportance() {

// Calcula la importancia de la permutación de los árboles en múltiples hilos
#ifdef OLD_WIN_R_BUILD
    // #nocov start
    progress = 0;
    clock_t start_time = clock();
    clock_t lap_time = clock();

    // Inicializa la importancia y la varianza
    variable_importance.resize(num_independent_variables, 0);
    std::vector<double> variance;
    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
        variance.resize(num_independent_variables, 0);
    }
    if (importance_mode == IMP_PERM_CASEWISE) {
        variable_importance_casewise.resize(num_independent_variables * num_samples, 0);
    }

    // Calcula la importancia
    for (size_t i = 0; i < num_trees; ++i) {
        trees[i]->computePermutationImportance(variable_importance, variance, variable_importance_casewise);
        progress++;
        showProgress("Computing permutation importance..", start_time, lap_time);
    }

#else
    progress = 0;
#ifdef R_BUILD
    aborted = false;
    aborted_threads = 0;
#endif

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Inicializa la importancia y la varianza
    std::vector<std::vector<double>> variable_importance_threads(num_threads);
    std::vector<std::vector<double>> variance_threads(num_threads);
    std::vector<std::vector<double>> variable_importance_casewise_threads(num_threads);

    // Calcula la importancia
    for (uint i = 0; i < num_threads; ++i) {
        variable_importance_threads[i].resize(num_independent_variables, 0);
        if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
            variance_threads[i].resize(num_independent_variables, 0);
        }
        if (importance_mode == IMP_PERM_CASEWISE) {
            variable_importance_casewise_threads[i].resize(num_independent_variables * num_samples, 0);
        }
        threads.emplace_back(&Forest::computeTreePermutationImportanceInThread, this, i,
                             std::ref(variable_importance_threads[i]), std::ref(variance_threads[i]),
                             std::ref(variable_importance_casewise_threads[i]));
    }
    showProgress("Computing permutation importance..", num_trees);
    for (auto &thread : threads) {
        thread.join();
    }

#ifdef R_BUILD
    if (aborted_threads > 0) {
        throw std::runtime_error("User interrupt.");
    }
#endif

    // Suma las importancias de los hilos
    variable_importance.resize(num_independent_variables, 0);
    for (size_t i = 0; i < num_independent_variables; ++i) {
        for (uint j = 0; j < num_threads; ++j) {
            variable_importance[i] += variable_importance_threads[j][i];
        }
    }
    variable_importance_threads.clear();

    // Suma las varianzas de los hilos
    std::vector<double> variance(num_independent_variables, 0);
    if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
        for (size_t i = 0; i < num_independent_variables; ++i) {
            for (uint j = 0; j < num_threads; ++j) {
                variance[i] += variance_threads[j][i];
            }
        }
        variance_threads.clear();
    }

    // Suma las importancias caso por caso de los hilos
    if (importance_mode == IMP_PERM_CASEWISE) {
        variable_importance_casewise.resize(num_independent_variables * num_samples, 0);
        for (size_t i = 0; i < variable_importance_casewise.size(); ++i) {
            for (uint j = 0; j < num_threads; ++j) {
                variable_importance_casewise[i] += variable_importance_casewise_threads[j][i];
            }
        }
        variable_importance_casewise_threads.clear();
    }
#endif

    for (size_t i = 0; i < variable_importance.size(); ++i) {
        variable_importance[i] /= num_trees;

        // Normaliza por varianza para la importancia de la permutación escalada
        if (importance_mode == IMP_PERM_BREIMAN || importance_mode == IMP_PERM_LIAW) {
            if (variance[i] != 0) {
                variance[i] = variance[i] / num_trees - variable_importance[i] * variable_importance[i];
                variable_importance[i] /= sqrt(variance[i] / num_trees);
            }
        }
    }

    if (importance_mode == IMP_PERM_CASEWISE) {
        for (size_t i = 0; i < variable_importance_casewise.size(); ++i) {
            variable_importance_casewise[i] /= num_trees;
        }
    }
}

#ifndef OLD_WIN_R_BUILD
// Hace crecer los árboles en un hilo
void Forest::growTreesInThread(uint thread_idx, std::vector<double>* variable_importance) {
    if (thread_ranges.size() > thread_idx + 1) {
        for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
            trees[i]->grow(variable_importance);

// Verifica si hay interrupción del usuario
#ifdef R_BUILD
            if (aborted) {
                std::unique_lock<std::mutex> lock(mutex);
                ++aborted_threads;
                condition_variable.notify_one();
                return;
            }
#endif

            // Incrementa el progreso en 1 árbol
            std::unique_lock<std::mutex> lock(mutex);
            ++progress;
            condition_variable.notify_one();
        }
    }
}

// Predice los árboles en un hilo
void Forest::predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction) {
    if (thread_ranges.size() > thread_idx + 1) {
        for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
            trees[i]->predict(prediction_data, oob_prediction);

// Verifica si hay interrupción del usuario
#ifdef R_BUILD
            if (aborted) {
                std::unique_lock<std::mutex> lock(mutex);
                ++aborted_threads;
                condition_variable.notify_one();
                return;
            }
#endif

            // Incrementa el progreso en 1 árbol
            std::unique_lock<std::mutex> lock(mutex);
            ++progress;
            condition_variable.notify_one();
        }
    }
}

// Predice internamente en un hilo
void Forest::predictInternalInThread(uint thread_idx) {
    // Crea rangos de hilos
    std::vector<uint> predict_ranges;
    equalSplit(predict_ranges, 0, num_samples - 1, num_threads);

    if (predict_ranges.size() > thread_idx + 1) {
        for (size_t i = predict_ranges[thread_idx]; i < predict_ranges[thread_idx + 1]; ++i) {
            predictInternal(i);

// Verifica si hay interrupción del usuario
#ifdef R_BUILD
            if (aborted) {
                std::unique_lock<std::mutex> lock(mutex);
                ++aborted_threads;
                condition_variable.notify_one();
                return;
            }
#endif

            // Incrementa el progreso en 1 árbol
            std::unique_lock<std::mutex> lock(mutex);
            ++progress;
            condition_variable.notify_one();
        }
    }
}

// Calcula la importancia de la permutación del árbol en un hilo
void Forest::computeTreePermutationImportanceInThread(uint thread_idx, std::vector<double>& importance,
                                                      std::vector<double>& variance, std::vector<double>& importance_casewise) {
    if (thread_ranges.size() > thread_idx + 1) {
        for (size_t i = thread_ranges[thread_idx]; i < thread_ranges[thread_idx + 1]; ++i) {
            trees[i]->computePermutationImportance(importance, variance, importance_casewise);

// Verifica si hay interrupción del usuario
#ifdef R_BUILD
            if (aborted) {
                std::unique_lock<std::mutex> lock(mutex);
                ++aborted_threads;
                condition_variable.notify_one();
                return;
            }
#endif

            // Incrementa el progreso en 1 árbol
            std::unique_lock<std::mutex> lock(mutex);
            ++progress;
            condition_variable.notify_one();
        }
    }
}
#endif

// #nocov start
// Carga el bosque desde un archivo
void Forest::loadFromFile(std::string filename) {
    if (verbose_out)
        *verbose_out << "Loading forest from file " << filename << "." << std::endl;

    // Abre el archivo para leer
    std::ifstream infile;
    infile.open(filename, std::ios::binary);
    if (!infile.good()) {
        throw std::runtime_error("Could not read from input file: " + filename + ".");
    }

    // Omite los nombres de las variables dependientes (ya leídos)
    uint num_dependent_variables;
    infile.read(reinterpret_cast<char*>(&num_dependent_variables), sizeof(num_dependent_variables));
    for (size_t i = 0; i < num_dependent_variables; ++i) {
        size_t length;
        infile.read(reinterpret_cast<char*>(&length), sizeof(size_t));
        infile.ignore(length);
    }

    // Lee el número de árboles
    infile.read(reinterpret_cast<char*>(&num_trees), sizeof(num_trees));

    // Lee las variables ordenadas
    readVector1D(data->getIsOrderedVariable(), infile);

    // Lee los datos del árbol. Esto es diferente para tipos de árboles -> función virtual
    loadFromFileInternal(infile);

    infile.close();

    // Crea rangos de hilos
    equalSplit(thread_ranges, 0, num_trees - 1, num_threads);
}

// Carga los nombres de las variables dependientes desde un archivo
void Forest::loadDependentVariableNamesFromFile(std::string filename) {

    // Abre el archivo para leer
    std::ifstream infile;
    infile.open(filename, std::ios::binary);
    if (!infile.good()) {
        throw std::runtime_error("Could not read from input file: " + filename + ".");
    }

    // Lee los nombres de las variables dependientes
    dependent_variable_names.clear();
    uint num_dependent_variables = 0;
    infile.read(reinterpret_cast<char*>(&num_dependent_variables), sizeof(num_dependent_variables));
    for (size_t i = 0; i < num_dependent_variables; ++i) {
        size_t length;
        infile.read(reinterpret_cast<char*>(&length), sizeof(size_t));
        std::unique_ptr<char[]> temp(new char[length + 1]);
        infile.read(temp.get(), length * sizeof(char));
        temp[length] = '\0';
        dependent_variable_names.push_back(temp.get());
    }

    infile.close();
}

// Carga los datos desde un archivo
std::unique_ptr<Data> Forest::loadDataFromFile(const std::string& data_path) {
    std::unique_ptr<Data> result;
    switch (memory_mode) {
    case MEM_DOUBLE:
        result.reset(new DataDouble());
        break;
    case MEM_FLOAT:
        result.reset(new DataFloat());
        break;
    case MEM_CHAR:
        result.reset(new DataChar());
        break;
    }

    if (verbose_out)
        *verbose_out << "Loading input file: " << data_path << "." << std::endl;
    bool found_rounding_error = result->loadFromFile(data_path, dependent_variable_names);
    if (found_rounding_error && verbose_out) {
        *verbose_out << "Warning: Rounding or Integer overflow occurred. Use FLOAT or DOUBLE precision to avoid this."
                     << std::endl;
    }
    return result;
}
// #nocov end

// Establece el vector de pesos de selección de división
void Forest::setSplitWeightVector(std::vector<std::vector<double>>& split_select_weights) {

    // El tamaño debe ser 1 x num_independent_variables o num_trees x num_independent_variables
    if (split_select_weights.size() != 1 && split_select_weights.size() != num_trees) {
        throw std::runtime_error("Size of split select weights not equal to 1 or number of trees.");
    }

    // Reserva espacio
    size_t num_weights = num_independent_variables;
    if (importance_mode == IMP_GINI_CORRECTED) {
        num_weights = 2 * num_independent_variables;
    }
    if (split_select_weights.size() == 1) {
        this->split_select_weights[0].resize(num_weights);
    } else {
        this->split_select_weights.clear();
        this->split_select_weights.resize(num_trees, std::vector<double>(num_weights));
    }

    // Divide en variables deterministas y ponderadas, ignora los pesos cero
    for (size_t i = 0; i < split_select_weights.size(); ++i) {
        size_t num_zero_weights = 0;

        // El tamaño debe ser 1 x num_independent_variables o num_trees x num_independent_variables
        if (split_select_weights[i].size() != num_independent_variables) {
            throw std::runtime_error("Number of split select weights not equal to number of independent variables.");
        }

        for (size_t j = 0; j < split_select_weights[i].size(); ++j) {
            double weight = split_select_weights[i][j];

            if (weight == 0) {
                ++num_zero_weights;
            } else if (weight < 0 || weight > 1) {
                throw std::runtime_error("One or more split select weights not in range [0,1].");
            } else {
                this->split_select_weights[i][j] = weight;
            }
        }

        // Copia los pesos para la importancia de la impureza corregida
        if (importance_mode == IMP_GINI_CORRECTED) {
            std::vector<double>* sw = &(this->split_select_weights[i]);
            std::copy_n(sw->begin(), num_independent_variables, sw->begin() + num_independent_variables);
        }

        if (num_weights - num_zero_weights < mtry) {
            throw std::runtime_error("Too many zeros in split select weights. Need at least mtry variables to split at.");
        }
    }
}

// Establece las variables que siempre se considerarán para dividir
void Forest::setAlwaysSplitVariables(const std::vector<std::string>& always_split_variable_names) {

    deterministic_varIDs.reserve(num_independent_variables);

    for (auto& variable_name : always_split_variable_names) {
        size_t varID = data->getVariableID(variable_name);
        deterministic_varIDs.push_back(varID);
    }

    if (deterministic_varIDs.size() + this->mtry > num_independent_variables) {
        throw std::runtime_error(
            "Number of variables to be always considered for splitting plus mtry cannot be larger than number of independent variables.");
    }

    // También agrega variables para la importancia de la impureza corregida
    if (importance_mode == IMP_GINI_CORRECTED) {
        size_t num_deterministic_varIDs = deterministic_varIDs.size();
        for (size_t k = 0; k < num_deterministic_varIDs; ++k) {
            deterministic_varIDs.push_back(k + num_independent_variables);
        }
    }
}

#ifdef OLD_WIN_R_BUILD
// #nocov start
// Muestra el progreso del crecimiento
void Forest::showProgress(std::string operation, clock_t start_time, clock_t& lap_time) {

    // Verifica si hay interrupción del usuario
    if (checkInterrupt()) {
        throw std::runtime_error("User interrupt.");
    }

    double elapsed_time = (clock() - lap_time) / CLOCKS_PER_SEC;
    if (elapsed_time > STATUS_INTERVAL) {
        double relative_progress = (double) progress / (double) num_trees;
        double time_from_start = (clock() - start_time) / CLOCKS_PER_SEC;
        uint remaining_time = (1 / relative_progress - 1) * time_from_start;
        if (verbose_out) {
            *verbose_out << operation << " Progress: " << round(100 * relative_progress)
                         << "%. Estimated remaining time: " << beautifyTime(remaining_time) << "." << std::endl;
        }
        lap_time = clock();
    }
}
// #nocov end
#else
// Muestra el progreso del crecimiento
void Forest::showProgress(std::string operation, size_t max_progress) {
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::seconds;

    steady_clock::time_point start_time = steady_clock::now();
    steady_clock::time_point last_time = steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex);

    // Espera mensaje de los hilos y muestra salida si ha pasado suficiente tiempo
    while (progress < max_progress) {
        condition_variable.wait(lock);
        seconds elapsed_time = duration_cast<seconds>(steady_clock::now() - last_time);

// Verifica si hay interrupción del usuario
#ifdef R_BUILD
        if (!aborted && checkInterrupt()) {
            aborted = true;
        }
        if (aborted && aborted_threads >= num_threads) {
            return;
        }
#endif

        if (progress > 0 && elapsed_time.count() > STATUS_INTERVAL) {
            double relative_progress = (double) progress / (double) max_progress;
            seconds time_from_start = duration_cast<seconds>(steady_clock::now() - start_time);
            uint remaining_time = (1 / relative_progress - 1) * time_from_start.count();
            if (verbose_out) {
                *verbose_out << operation << " Progress: " << round(100 * relative_progress) << "%. Estimated remaining time: "
                             << beautifyTime(remaining_time) << "." << std::endl;
            }
            last_time = steady_clock::now();
        }
    }
}
#endif

} // namespace ranger

