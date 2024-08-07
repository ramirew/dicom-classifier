/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <math.h>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cstring>
#include "utility.h"
#include "globals.h"
#include "Data.h"

namespace ranger {

// Función para dividir de manera equitativa
void equalSplit(std::vector<uint>& result, uint start, uint end, uint num_parts) {
    result.reserve(num_parts + 1);

    // Devolver rango si solo hay 1 parte
    if (num_parts == 1) {
        result.push_back(start);
        result.push_back(end + 1);
        return;
    }

    // Devolver vector de inicio a fin+1 si hay más partes que elementos
    if (num_parts > end - start + 1) {
        for (uint i = start; i <= end + 1; ++i) {
            result.push_back(i);
        }
        return;
    }

    uint length = (end - start + 1);
    uint part_length_short = length / num_parts;
    uint part_length_long = (uint) ceil(length / ((double) num_parts));
    uint cut_pos = length % num_parts;

    // Agregar rangos largos
    for (uint i = start; i < start + cut_pos * part_length_long; i = i + part_length_long) {
        result.push_back(i);
    }

    // Agregar rangos cortos
    for (uint i = start + cut_pos * part_length_long; i <= end + 1; i = i + part_length_short) {
        result.push_back(i);
    }
}

// Función para cargar un array de dobles desde un archivo
void loadDoubleArrayFromFile(double*& result, size_t& size, const std::string& filename) { // #nocov start
    // Abrir archivo de entrada
    std::ifstream input_file;
    input_file.open(filename);
    if (!input_file.good()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    // Leer la primera línea, ignorar el resto
    std::string line;
    getline(input_file, line);
    std::stringstream line_stream(line);
    std::vector<double> temp_result;
    double token;
    while (line_stream >> token) {
        temp_result.push_back(token);
    }

    // Asignar memoria para el array resultante
    size = temp_result.size();
    result = new double[size];
    std::memcpy(result, temp_result.data(), size * sizeof(double));
} // #nocov end

// Función para realizar sorteos sin reemplazo usando un array dinámico
void drawWithoutReplacement(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator, size_t max, size_t num_samples) {
    std::vector<size_t> temp_result;
    if (num_samples < max / 10) {
        drawWithoutReplacementSimple(temp_result, random_number_generator, max, num_samples);
    } else {
        drawWithoutReplacementFisherYates(temp_result, random_number_generator, max, num_samples);
    }
    result_size = temp_result.size();
    result = new size_t[result_size];
    std::memcpy(result, temp_result.data(), result_size * sizeof(size_t));
}

// Función para realizar sorteos sin reemplazo considerando una lista de exclusión, usando un array dinámico
void drawWithoutReplacementSkip(size_t*& result, size_t& result_size, std::mt19937_64& random_number_generator, size_t max, const size_t* skip, size_t skip_size, size_t num_samples) {
    std::vector<size_t> temp_result;
    std::vector<size_t> temp_skip(skip, skip + skip_size);
    if (num_samples < max / 10) {
        drawWithoutReplacementSimple(temp_result, random_number_generator, max, temp_skip, num_samples);
    } else {
        drawWithoutReplacementFisherYates(temp_result, random_number_generator, max, temp_skip, num_samples);
    }
    result_size = temp_result.size();
    result = new size_t[result_size];
    std::memcpy(result, temp_result.data(), result_size * sizeof(size_t));
}

// Función para realizar sorteos sin reemplazo simples
void drawWithoutReplacementSimple(std::vector<size_t>& result, std::mt19937_64& random_number_generator, size_t max, size_t num_samples) {
    result.reserve(num_samples);

    // Establecer todos como no seleccionados
    std::vector<bool> temp;
    temp.resize(max, false);

    std::uniform_int_distribution<size_t> unif_dist(0, max - 1);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t draw;
        do {
            draw = unif_dist(random_number_generator);
        } while (temp[draw]);
        temp[draw] = true;
        result.push_back(draw);
    }
}

// Función para realizar sorteos sin reemplazo simples con exclusión
void drawWithoutReplacementSimple(std::vector<size_t>& result, std::mt19937_64& random_number_generator, size_t max, const std::vector<size_t>& skip, size_t num_samples) {
    result.reserve(num_samples);

    // Establecer todos como no seleccionados
    std::vector<bool> temp;
    temp.resize(max, false);

    std::uniform_int_distribution<size_t> unif_dist(0, max - 1 - skip.size());
    for (size_t i = 0; i < num_samples; ++i) {
        size_t draw;
        do {
            draw = unif_dist(random_number_generator);
            for (auto& skip_value : skip) {
                if (draw >= skip_value) {
                    ++draw;
                }
            }
        } while (temp[draw]);
        temp[draw] = true;
        result.push_back(draw);
    }
}

// Función para realizar sorteos sin reemplazo usando el algoritmo Fisher-Yates
void drawWithoutReplacementFisherYates(std::vector<size_t>& result, std::mt19937_64& random_number_generator, size_t max, size_t num_samples) {
    // Crear índices
    result.resize(max);
    std::iota(result.begin(), result.end(), 0);

    // Sortear sin reemplazo usando el algoritmo Fisher-Yates
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t j = i + distribution(random_number_generator) * (max - i);
        std::swap(result[i], result[j]);
    }

    result.resize(num_samples);
}

// Función para realizar sorteos sin reemplazo usando el algoritmo Fisher-Yates con exclusión
void drawWithoutReplacementFisherYates(std::vector<size_t>& result, std::mt19937_64& random_number_generator, size_t max, const std::vector<size_t>& skip, size_t num_samples) {
    // Crear índices
    result.resize(max);
    std::iota(result.begin(), result.end(), 0);

    // Excluir índices
    for (size_t i = 0; i < skip.size(); ++i) {
        result.erase(result.begin() + skip[skip.size() - 1 - i]);
    }

    // Sortear sin reemplazo usando el algoritmo Fisher-Yates
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < num_samples; ++i) {
        size_t j = i + distribution(random_number_generator) * (max - skip.size() - i);
        std::swap(result[i], result[j]);
    }

    result.resize(num_samples);
}

// Función para calcular el índice de concordancia
double computeConcordanceIndex(const Data& data, const std::vector<double>& sum_chf, const std::vector<size_t>& sample_IDs, std::vector<double>* prediction_error_casewise) {
    // Calcular índice de concordancia
    double concordance = 0;
    double permissible = 0;

    std::vector<double> concordance_casewise;
    std::vector<double> permissible_casewise;
    if (prediction_error_casewise) {
        concordance_casewise.resize(prediction_error_casewise->size(), 0);
        permissible_casewise.resize(prediction_error_casewise->size(), 0);
    }

    for (size_t i = 0; i < sum_chf.size(); ++i) {
        size_t sample_i = i;
        if (!sample_IDs.empty()) {
            sample_i = sample_IDs[i];
        }
        double time_i = data.get_y(sample_i, 0);
        double status_i = data.get_y(sample_i, 1);

        double conc, perm;
        if (prediction_error_casewise) {
            conc = concordance_casewise[i];
            perm = permissible_casewise[i];
        } else {
            conc = 0;
            perm = 0;
        }

        for (size_t j = i + 1; j < sum_chf.size(); ++j) {
            size_t sample_j = j;
            if (!sample_IDs.empty()) {
                sample_j = sample_IDs[j];
            }
            double time_j = data.get_y(sample_j, 0);
            double status_j = data.get_y(sample_j, 1);

            if (time_i < time_j && status_i == 0) {
                continue;
            }
            if (time_j < time_i && status_j == 0) {
                continue;
            }
            if (time_i == time_j && status_i == status_j) {
                continue;
            }

            double co;
            if (time_i < time_j && sum_chf[i] > sum_chf[j]) {
                co = 1;
            } else if (time_j < time_i && sum_chf[j] > sum_chf[i]) {
                co = 1;
            } else if (sum_chf[i] == sum_chf[j]) {
                co = 0.5;
            } else {
                co = 0;
            }

            conc += co;
            perm += 1;

            if (prediction_error_casewise) {
                concordance_casewise[j] += co;
                permissible_casewise[j] += 1;
            }
        }

        concordance += conc;
        permissible += perm;
        if (prediction_error_casewise) {
            concordance_casewise[i] = conc;
            permissible_casewise[i] = perm;
        }
    }

    if (prediction_error_casewise) {
        for (size_t i = 0; i < prediction_error_casewise->size(); ++i) {
            (*prediction_error_casewise)[i] = 1 - concordance_casewise[i] / permissible_casewise[i];
        }
    }

    return (concordance / permissible);
}

// Función para verificar variables no ordenadas
std::string checkUnorderedVariables(const Data& data, const std::vector<std::string>& unordered_variable_names) { // #nocov start
    size_t num_rows = data.getNumRows();
    std::vector<size_t> sampleIDs(num_rows);
    std::iota(sampleIDs.begin(), sampleIDs.end(), 0);

    // Verificar todas las variables no ordenadas
    for (auto& variable_name : unordered_variable_names) {
        size_t varID = data.getVariableID(variable_name);
        std::vector<double> all_values;
        data.getAllValues(all_values, sampleIDs, varID, 0, sampleIDs.size());

        // Verificar conteo de niveles
        size_t max_level_count = 8 * sizeof(size_t) - 1;
        if (all_values.size() > max_level_count) {
            return "Demasiados niveles en la variable categórica no ordenada " + variable_name + ". Solo se permiten "
                   + uintToString(max_level_count) + " niveles en este sistema.";
        }

        // Verificar enteros positivos
        if (!checkPositiveIntegers(all_values)) {
            return "No todos los valores en la variable categórica no ordenada " + variable_name + " son enteros positivos.";
        }
    }
    return "";
} // #nocov end

bool checkPositiveIntegers(const std::vector<double>& all_values) { // #nocov start
    for (auto& value : all_values) {
        if (value < 1 || !(floor(value) == value)) {
            return false;
        }
    }
    return true;
} // #nocov end

// Funciones adicionales de cálculo de valores p y otros métodos estadísticos

double maxstatPValueLau92(double b, double minprop, double maxprop) {
    if (b < 1) {
        return 1.0;
    }

    // Calcular una sola vez (minprop/maxprop no cambian durante la ejecución)
    static double logprop = log((maxprop * (1 - minprop)) / ((1 - maxprop) * minprop));

    double db = dstdnorm(b);
    double p = 4 * db / b + db * (b - 1 / b) * logprop;

    if (p > 0) {
        return p;
    } else {
        return 0;
    }
}

double maxstatPValueLau94(double b, double minprop, double maxprop, size_t N, const std::vector<size_t>& m) {
    double D = 0;
    for (size_t i = 0; i < m.size() - 1; ++i) {
        double m1 = m[i];
        double m2 = m[i + 1];

        double t = sqrt(1.0 - m1 * (N - m2) / ((N - m1) * m2));
        D += 1 / M_PI * exp(-b * b / 2) * (t - (b * b / 4 - 1) * (t * t * t) / 6);
    }

    return 2 * (1 - pstdnorm(b)) + D;
}

double maxstatPValueUnadjusted(double b) {
    return 2 * pstdnorm(-b);
}

double dstdnorm(double x) {
    return exp(-0.5 * x * x) / sqrt(2 * M_PI);
}

double pstdnorm(double x) {
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

// Función para ajustar valores p
std::vector<double> adjustPvalues(std::vector<double>& unadjusted_pvalues) {
    size_t num_pvalues = unadjusted_pvalues.size();
    std::vector<double> adjusted_pvalues(num_pvalues, 0);

    // Obtener el orden de los valores p
    std::vector<size_t> indices = order(unadjusted_pvalues, true);

    // Calcular valores p ajustados
    adjusted_pvalues[indices[0]] = unadjusted_pvalues[indices[0]];
    for (size_t i = 1; i < indices.size(); ++i) {
        size_t idx = indices[i];
        size_t idx_last = indices[i - 1];

        adjusted_pvalues[idx] = std::min(adjusted_pvalues[idx_last],
                                         (double) num_pvalues / (double) (num_pvalues - i) * unadjusted_pvalues[idx]);
    }
    return adjusted_pvalues;
}

// Función para calcular las puntuaciones de logrank
std::vector<double> logrankScores(const std::vector<double>& time, const std::vector<double>& status) {
    size_t n = time.size();
    std::vector<double> scores(n);

    // Obtener el orden de los puntos de tiempo
    std::vector<size_t> indices = order(time, false);

    // Calcular puntuaciones
    double cumsum = 0;
    size_t last_unique = -1;
    for (size_t i = 0; i < n; ++i) {
        // Continuar si el siguiente valor es el mismo
        if (i < n - 1 && time[indices[i]] == time[indices[i + 1]]) {
            continue;
        }

        // Calcular suma y puntuaciones para todos los valores no únicos en fila
        for (size_t j = last_unique + 1; j <= i; ++j) {
            cumsum += status[indices[j]] / (n - i);
        }
        for (size_t j = last_unique + 1; j <= i; ++j) {
            scores[indices[j]] = status[indices[j]] - cumsum;
        }

        // Guardar último valor calculado
        last_unique = i;
    }

    return scores;
}

// Función para calcular el valor máximo de una estadística
void maxstat(const std::vector<double>& scores, const std::vector<double>& x, const std::vector<size_t>& indices, double& best_maxstat, double& best_split_value, double minprop, double maxprop) {
    size_t n = x.size();

    double sum_all_scores = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_all_scores += scores[indices[i]];
    }

    // Calcular suma de diferencias con la media para la varianza
    double mean_scores = sum_all_scores / n;
    double sum_mean_diff = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_mean_diff += (scores[i] - mean_scores) * (scores[i] - mean_scores);
    }

    // Obtener el menor y mayor punto de división a considerar
    size_t minsplit = 0;
    if (n * minprop > 1) {
        minsplit = n * minprop - 1;
    }
    size_t maxsplit = n * maxprop - 1;

    // Para todos los valores únicos de x
    best_maxstat = -1;
    best_split_value = -1;
    double sum_scores = 0;
    size_t n_left = 0;
    for (size_t i = 0; i <= maxsplit; ++i) {
        sum_scores += scores[indices[i]];
        n_left++;

        // No considerar divisiones menores que minsplit para dividir
        if (i < minsplit) {
            continue;
        }

        // Considerar solo valores únicos
        if (i < n - 1 && x[indices[i]] == x[indices[i + 1]]) {
            continue;
        }

        // Si el valor es el mayor posible, detenerse
        if (x[indices[i]] == x[indices[n - 1]]) {
            break;
        }

        double S = sum_scores;
        double E = (double) n_left / (double) n * sum_all_scores;
        double V = (double) n_left * (double) (n - n_left) / (double) (n * (n - 1)) * sum_mean_diff;
        double T = fabs((S - E) / sqrt(V));

        if (T > best_maxstat) {
            best_maxstat = T;

            // Usar división en el punto medio si es posible
            if (i < n - 1) {
                best_split_value = (x[indices[i]] + x[indices[i + 1]]) / 2;
            } else {
                best_split_value = x[indices[i]];
            }
        }
    }
}

// Función para calcular el número de muestras a la izquierda del punto de corte usando un array dinámico
size_t* numSamplesLeftOfCutpoint(double* x, size_t x_size, const size_t* indices, size_t indices_size) {
    size_t* num_samples_left = new size_t[x_size]; // Asignar memoria para el array resultante
    size_t num_samples = 1;
    num_samples_left[0] = num_samples;
    for (size_t i = 1; i < x_size; ++i) {
        if (x[indices[i]] == x[indices[i - 1]]) {
            num_samples++;
        } else {
            num_samples++;
        }
        num_samples_left[i] = num_samples; // Actualizar el array resultante
    }
    return num_samples_left;
}

// Función para leer desde un flujo de entrada con manejo de errores
std::stringstream& readFromStream(std::stringstream& in, double& token) {
    if (!(in >> token) && (std::fpclassify(token) == FP_SUBNORMAL)) {
        in.clear();
    }
    return in;
}

// Función para calcular la verosimilitud logarítmica beta
double betaLogLik(double y, double mean, double phi) {
    if (y < std::numeric_limits<double>::epsilon()) {
        y = std::numeric_limits<double>::epsilon();
    } else if (y >= 1) {
        y = 1 - std::numeric_limits<double>::epsilon();
    }
    if (mean < std::numeric_limits<double>::epsilon()) {
        mean = std::numeric_limits<double>::epsilon();
    } else if (mean >= 1) {
        mean = 1 - std::numeric_limits<double>::epsilon();
    }
    if (phi < std::numeric_limits<double>::epsilon()) {
        phi = std::numeric_limits<double>::epsilon();
    } else if (mean >= 1) {
        phi = 1 - std::numeric_limits<double>::epsilon();
    }

    return (lgamma(phi) - lgamma(mean * phi) - lgamma((1 - mean) * phi) + (mean * phi - 1) * log(y)
            + ((1 - mean) * phi - 1) * log(1 - y));
}

} // namespace ranger
