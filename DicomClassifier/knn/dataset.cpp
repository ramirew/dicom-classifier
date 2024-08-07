#include "dataset.h"          // Incluye la definición de dataset_base y DatasetPointer.
#include "debug.h"           // Incluye macros para la depuración.
#include <algorithm>         // Incluye funciones estándar de algoritmos como std::random_shuffle.
#include <random>            // Incluye funcionalidades para generar números aleatorios.

// Constructor para inicializar el dataset_base.
// Asigna memoria para las etiquetas y llama al constructor de matrix_base.
dataset_base::dataset_base(size_t rows, size_t cols, size_t numLabels)
    : matrix_base(rows, cols),        // Inicializa la base de la matriz con el número de filas y columnas.
    numLabels(numLabels),           // Inicializa el número de etiquetas.
    labels(new int[rows]) {         // Asigna memoria dinámica para el array de etiquetas.
    DEBUGMEM("dataset: allocated int: %lu in %p\n", rows, labels);
    // Imprime un mensaje de depuración que muestra el tamaño y la dirección del array de etiquetas.
}

// Destructor para liberar la memoria asignada a las etiquetas.
dataset_base::~dataset_base() {
    DEBUGMEM("dataset: deleting: %lu in %p\n", rows, labels);
    // Imprime un mensaje de depuración que muestra el tamaño y la dirección del array de etiquetas.
    delete[] labels; // Libera la memoria dinámica asignada para el array de etiquetas.
}

// Función auxiliar que llena un array con índices aleatorios.
// Utiliza punteros crudos para manejar el array de índices.
void fillWithRandomIndices(std::vector<int>& indices, size_t nIndices) {
    // Llenar el vector con números secuenciales
    std::iota(indices.begin(), indices.end(), 0);
    // Crear un generador de números aleatorios
    std::random_device rd;
    std::mt19937 g(rd());
    // Mezclar los números secuenciales aleatoriamente
    std::shuffle(indices.begin(), indices.end(), g);
}


// Función para dividir el dataset en entrenamiento y validación.
// Utiliza punteros inteligentes para gestionar los nuevos datasets.
void dataset_base::splitDataset(DatasetPointer& train, DatasetPointer& valid, double train_percent) {
    // Vector para almacenar índices aleatorios
    std::vector<int> randomIndices(rows);
    // Llenar el vector con índices aleatorios
    fillWithRandomIndices(randomIndices, rows);

    // Calcular el umbral basado en el porcentaje de entrenamiento
    size_t threshold = static_cast<size_t>(train_percent * rows);

    // Crear conjuntos de entrenamiento y validación con las filas correspondientes
    train = std::make_shared<dataset_base>(threshold, cols, numLabels);
    valid = std::make_shared<dataset_base>(rows - threshold, cols, numLabels);

    size_t currentRowTrain = 0; // Índice actual para el conjunto de entrenamiento
    size_t currentRowValid = 0; // Índice actual para el conjunto de validación

    // Iterar sobre todas las filas del dataset original
    for (size_t i = 0; i < rows; ++i) {
        const auto& sourceRow = randomIndices[i];
        if (i < threshold) {
            // Copiar los datos de la fila al conjunto de entrenamiento
            for (size_t j = 0; j < cols; ++j) {
                train->pos(currentRowTrain, j) = pos(sourceRow, j);
            }
            train->label(currentRowTrain) = label(sourceRow);
            ++currentRowTrain;
        } else {
            // Copiar los datos de la fila al conjunto de validación
            for (size_t j = 0; j < cols; ++j) {
                valid->pos(currentRowValid, j) = pos(sourceRow, j);
            }
            valid->label(currentRowValid) = label(sourceRow);
            ++currentRowValid;
        }
    }
}
