/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iterator>

#include "Data.h"
#include "utility.h"

namespace ranger {

// Constructor de la clase Data
Data::Data() :
    num_rows(0), num_rows_rounded(0), num_cols(0), snp_data(nullptr), num_cols_no_snp(0), externalData(true), max_num_unique_values(0), order_snps(false) {
}

// Obtiene el ID de una variable dado su nombre
size_t Data::getVariableID(const std::string& variable_name) const {
    auto it = std::find(variable_names.cbegin(), variable_names.cend(), variable_name);
    if (it == variable_names.cend()) {
        throw std::runtime_error("Variable " + variable_name + " not found.");
    }
    return (std::distance(variable_names.cbegin(), it));
}

// #nocov start (no se puede probar porque GenABEL no está en CRAN)
// Agrega datos SNP al conjunto de datos
void Data::addSnpData(unsigned char* snp_data, size_t num_cols_snp) {
    num_cols = num_cols_no_snp + num_cols_snp;
    num_rows_rounded = roundToNextMultiple(num_rows, 4);
    this->snp_data = snp_data;
}
// #nocov end

// #nocov start
// Carga datos desde un archivo
bool Data::loadFromFile(std::string filename, std::vector<std::string>& dependent_variable_names) {

    bool result;

    // Abre el archivo de entrada
    std::ifstream input_file;
    input_file.open(filename);
    if (!input_file.good()) {
        throw std::runtime_error("Could not open input file.");
    }

    // Cuenta el número de filas
    size_t line_count = 0;
    std::string line;
    while (getline(input_file, line)) {
        ++line_count;
    }
    num_rows = line_count - 1;
    input_file.close();
    input_file.open(filename);

    // Verifica si está separado por comas, punto y coma o espacios
    std::string header_line;
    getline(input_file, header_line);

    // Determina el separador y llama al método apropiado
    if (header_line.find(',') != std::string::npos) {
        result = loadFromFileOther(input_file, header_line, dependent_variable_names, ',');
    } else if (header_line.find(';') != std::string::npos) {
        result = loadFromFileOther(input_file, header_line, dependent_variable_names, ';');
    } else {
        result = loadFromFileWhitespace(input_file, header_line, dependent_variable_names);
    }

    externalData = false;
    input_file.close();
    return result;
}

// Carga datos desde un archivo separado por espacios
bool Data::loadFromFileWhitespace(std::ifstream& input_file, std::string header_line,
                                  std::vector<std::string>& dependent_variable_names) {

    size_t num_dependent_variables = dependent_variable_names.size();
    std::vector<size_t> dependent_varIDs;
    dependent_varIDs.resize(num_dependent_variables);

    // Lee la cabecera
    std::string header_token;
    std::stringstream header_line_stream(header_line);
    size_t col = 0;
    while (header_line_stream >> header_token) {
        bool is_dependent_var = false;
        for (size_t i = 0; i < dependent_variable_names.size(); ++i) {
            if (header_token == dependent_variable_names[i]) {
                dependent_varIDs[i] = col;
                is_dependent_var = true;
            }
        }
        if (!is_dependent_var) {
            variable_names.push_back(header_token);
        }
        ++col;
    }

    num_cols = variable_names.size();
    num_cols_no_snp = num_cols;

    // Lee el cuerpo
    reserveMemory(num_dependent_variables);
    bool error = false;
    std::string line;
    size_t row = 0;
    while (getline(input_file, line)) {
        double token;
        std::stringstream line_stream(line);
        size_t column = 0;
        while (readFromStream(line_stream, token)) {
            size_t column_x = column;
            bool is_dependent_var = false;
            for (size_t i = 0; i < dependent_varIDs.size(); ++i) {
                if (column == dependent_varIDs[i]) {
                    set_y(i, row, token, error);
                    is_dependent_var = true;
                    break;
                } else if (column > dependent_varIDs[i]) {
                    --column_x;
                }
            }
            if (!is_dependent_var) {
                set_x(column_x, row, token, error);
            }
            ++column;
        }
        if (column > (num_cols + num_dependent_variables)) {
            throw std::runtime_error(
                std::string("Could not open input file. Too many columns in row ") + std::to_string(row) + std::string("."));
        } else if (column < (num_cols + num_dependent_variables)) {
            throw std::runtime_error(
                std::string("Could not open input file. Too few columns in row ") + std::to_string(row)
                + std::string(". Are all values numeric?"));
        }
        ++row;
    }
    num_rows = row;
    return error;
}

// Carga datos desde un archivo separado por un caracter específico
bool Data::loadFromFileOther(std::ifstream& input_file, std::string header_line,
                             std::vector<std::string>& dependent_variable_names, char separator) {

    size_t num_dependent_variables = dependent_variable_names.size();
    std::vector<size_t> dependent_varIDs;
    dependent_varIDs.resize(num_dependent_variables);

    // Lee la cabecera
    std::string header_token;
    std::stringstream header_line_stream(header_line);
    size_t col = 0;
    while (getline(header_line_stream, header_token, separator)) {
        bool is_dependent_var = false;
        for (size_t i = 0; i < dependent_variable_names.size(); ++i) {
            if (header_token == dependent_variable_names[i]) {
                dependent_varIDs[i] = col;
                is_dependent_var = true;
            }
        }
        if (!is_dependent_var) {
            variable_names.push_back(header_token);
        }
        ++col;
    }

    num_cols = variable_names.size();
    num_cols_no_snp = num_cols;

    // Lee el cuerpo
    reserveMemory(num_dependent_variables);
    bool error = false;
    std::string line;
    size_t row = 0;
    while (getline(input_file, line)) {
        std::string token_string;
        double token;
        std::stringstream line_stream(line);
        size_t column = 0;
        while (getline(line_stream, token_string, separator)) {
            std::stringstream token_stream(token_string);
            readFromStream(token_stream, token);

            size_t column_x = column;
            bool is_dependent_var = false;
            for (size_t i = 0; i < dependent_varIDs.size(); ++i) {
                if (column == dependent_varIDs[i]) {
                    set_y(i, row, token, error);
                    is_dependent_var = true;
                    break;
                } else if (column > dependent_varIDs[i]) {
                    --column_x;
                }
            }
            if (!is_dependent_var) {
                set_x(column_x, row, token, error);
            }
            ++column;
        }
        ++row;
    }
    num_rows = row;
    return error;
}
// #nocov end

// Obtiene todos los valores de una variable para un conjunto de muestras
void Data::getAllValues(std::vector<double>& all_values, std::vector<size_t>& sampleIDs, size_t varID, size_t start,
                        size_t end) const {

    // Todos los valores para varID (sin duplicados) para las sampleIDs dadas
    if (getUnpermutedVarID(varID) < num_cols_no_snp) {

        all_values.reserve(end - start);
        for (size_t pos = start; pos < end; ++pos) {
            all_values.push_back(get_x(sampleIDs[pos], varID));
        }
        std::sort(all_values.begin(), all_values.end());
        all_values.erase(std::unique(all_values.begin(), all_values.end()), all_values.end());
    } else {
        // Si los datos son GWA, solo usar 0, 1, 2
        all_values = std::vector<double>( { 0, 1, 2 });
    }
}

// Obtiene los valores mínimo y máximo de una variable para un conjunto de muestras
void Data::getMinMaxValues(double& min, double&max, std::vector<size_t>& sampleIDs, size_t varID, size_t start,
                           size_t end) const {
    if (sampleIDs.size() > 0) {
        min = get_x(sampleIDs[start], varID);
        max = min;
    }
    for (size_t pos = start; pos < end; ++pos) {
        double value = get_x(sampleIDs[pos], varID);
        if (value < min) {
            min = value;
        }
        if (value > max) {
            max = value;
        }
    }
}

// Ordena los datos
void Data::sort() {

    // Reserva memoria
    index_data.resize(num_cols_no_snp * num_rows);

    // Para todas las columnas, obtiene valores únicos y guarda el índice para cada observación
    for (size_t col = 0; col < num_cols_no_snp; ++col) {

        // Obtiene todos los valores únicos
        std::vector<double> unique_values(num_rows);
        for (size_t row = 0; row < num_rows; ++row) {
            unique_values[row] = get_x(row, col);
        }
        std::sort(unique_values.begin(), unique_values.end());
        unique_values.erase(unique(unique_values.begin(), unique_values.end()), unique_values.end());

        // Obtiene el índice del valor único
        for (size_t row = 0; row < num_rows; ++row) {
            size_t idx = std::lower_bound(unique_values.begin(), unique_values.end(), get_x(row, col))
                         - unique_values.begin();
            index_data[col * num_rows + row] = idx;
        }

        // Guarda los valores únicos
        unique_data_values.push_back(unique_values);
        if (unique_values.size() > max_num_unique_values) {
            max_num_unique_values = unique_values.size();
        }
    }
}

// TODO: Implementar la ordenación para multiclass y survival
// #nocov start (no se puede probar porque GenABEL no está en CRAN)
// Ordena los niveles SNP
void Data::orderSnpLevels(bool corrected_importance) {
    // Detiene si no hay datos SNP
    if (snp_data == nullptr) {
        return;
    }

    size_t num_snps;
    if (corrected_importance) {
        num_snps = 2 * (num_cols - num_cols_no_snp);
    } else {
        num_snps = num_cols - num_cols_no_snp;
    }

    // Reserva espacio
    snp_order.resize(num_snps, std::vector<size_t>(3));

    // Para cada SNP
    for (size_t i = 0; i < num_snps; ++i) {
        size_t col = i;
        if (i >= (num_cols - num_cols_no_snp)) {
            // Obtiene el ID SNP no permutado
            col = i - num_cols + num_cols_no_snp;
        }

        // Ordena por la respuesta media
        std::vector<double> means(3, 0);
        std::vector<double> counts(3, 0);
        for (size_t row = 0; row < num_rows; ++row) {
            size_t row_permuted = row;
            if (i >= (num_cols - num_cols_no_snp)) {
                row_permuted = getPermutedSampleID(row);
            }
            size_t idx = col * num_rows_rounded + row_permuted;
            size_t value = (((snp_data[idx / 4] & mask[idx % 4]) >> offset[idx % 4]) - 1);

            // TODO: Mejor manera de tratar los valores faltantes?
            if (value > 2) {
                value = 0;
            }

            means[value] += get_y(row, 0);
            ++counts[value];
        }

        for (size_t value = 0; value < 3; ++value) {
            means[value] /= counts[value];
        }

        // Guarda el orden
        snp_order[i] = order(means, false);
    }

    order_snps = true;
}
// #nocov end

} // namespace ranger

