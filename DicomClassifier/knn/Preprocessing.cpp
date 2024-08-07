#include "Preprocessing.h"
#include <cmath>

using namespace std;

// Normalización por Media
MatrixPointer MeanNormalize(DatasetPointer data) {
    // Crea una matriz de resultados de dos filas (min y max) con el mismo número de columnas que el conjunto de datos.
    MatrixPointer results = MatrixPointer(new matrix_base(2, data->cols));
    results->clear();

    // Encuentra el valor mínimo y máximo para cada columna.
    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            // Actualiza el mínimo de la columna si el valor actual es menor.
            results->pos(0, j) = min(results->pos(0, j), data->pos(i, j));
            // Actualiza el máximo de la columna si el valor actual es mayor.
            results->pos(1, j) = max(results->pos(1, j), data->pos(i, j));
        }
    }

    // Normaliza cada elemento de la matriz.
    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            // Normaliza el valor usando min-max scaling.
            data->pos(i, j) = (data->pos(i, j) - results->pos(0, j)) / (results->pos(1, j) - results->pos(0, j));
        }
    }

    // Devuelve la matriz de resultados que contiene los mínimos y máximos por columna.
    return results;
}

// Aplicar Normalización por Media
void ApplyMeanNormalization(DatasetPointer data, MatrixPointer meanData) {
    // Aplica la normalización min-max a los datos usando los valores precomputados.
    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            // Normaliza el valor usando los mínimos y máximos proporcionados.
            data->pos(i, j) = (data->pos(i, j) - meanData->pos(0, j)) / (meanData->pos(1, j) - meanData->pos(0, j));
        }
    }
}

// Normalización Z-Score
MatrixPointer ZScore(DatasetPointer data) {
    /*
     * X = (X - X_mean) / X_std
     * Devuelve una matriz: la primera fila contiene la "media" para cada columna.
     * La segunda fila contiene la "desviación estándar" para cada columna.
     */

    // Crea una matriz de resultados de dos filas (media y desviación estándar) con el mismo número de columnas que el conjunto de datos.
    MatrixPointer results = MatrixPointer(new matrix_base(2, data->cols));
    results->clear();

    // Suma todos los valores para calcular la media.
    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            results->pos(0, j) += data->pos(i, j);  // Suma el valor a la media acumulada.
        }
    }

    // Calcula la media para cada columna.
    for (size_t j = 0; j < data->cols; j++) {
        results->pos(0, j) /= data->rows;  // Divide la suma total entre el número de filas para obtener la media.
    }

    // Calcula la desviación estándar.
    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            double data_minus_mean = data->pos(i, j) - results->pos(0, j);  // Calcula la diferencia con la media.
            results->pos(1, j) += data_minus_mean * data_minus_mean;  // Acumula el cuadrado de la diferencia.
        }
    }
    for (size_t j = 0; j < data->cols; j++) {
        results->pos(1, j) = sqrt(results->pos(1, j) / data->rows);  // Calcula la desviación estándar para cada columna.
    }

    // Aplica la normalización Z-Score a los datos.
    for (size_t i = 0; i < data->rows; i++) {
        for (size_t j = 0; j < data->cols; j++) {
            data->pos(i, j) = (data->pos(i, j) - results->pos(0, j)) / results->pos(1, j);  // Normaliza cada valor.
        }
    }

    // Devuelve la matriz de resultados que contiene las medias y desviaciones estándar por columna.
    return results;
}
