#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <vector>

// Función para obtener la distancia al cuadrado entre dos ejemplos
double GetSquaredDistance(std::shared_ptr<dataset_base> train, size_t trainExample, std::shared_ptr<dataset_base> target, size_t targetExample) {
    assert(train->cols == target->cols); // Verifica que los conjuntos de datos tengan el mismo número de columnas

    double sum = 0.0;
    for (size_t col = 0; col < train->cols; ++col) {
        double difference = train->pos(trainExample, col) - target->pos(targetExample, col); // Calcula la diferencia en la dimensión actual
        sum += difference * difference; // Suma el cuadrado de la diferencia a la suma total
    }
    return sum; // Devuelve la suma de las diferencias al cuadrado (distancia euclidiana al cuadrado)
}

// Función para ejecutar el algoritmo KNN
KNNResults KNN::run(int k, std::shared_ptr<dataset_base> target) {
    // Crea un nuevo conjunto de datos para almacenar los resultados
    std::shared_ptr<dataset_base> results = std::make_shared<dataset_base>(target->rows, target->numLabels, target->numLabels);
    results->clear(); // Limpia el conjunto de datos de resultados

    // Crea un vector para almacenar las distancias cuadradas y los índices de los ejemplos de entrenamiento
    std::vector<std::pair<double, size_t>> squaredDistances(data->rows);

    // Recorre cada ejemplo objetivo en el conjunto de datos objetivo
    for (size_t targetExample = 0; targetExample < target->rows; ++targetExample) {

#ifdef DEBUG_KNN
        if (targetExample % 100 == 0)
            DEBUGKNN("Target %lu of %lu\n", targetExample, target->rows); // Mensaje de depuración para rastrear el progreso
#endif

        // Calcula la distancia al cuadrado entre el ejemplo objetivo y todos los ejemplos de entrenamiento
        for (size_t trainExample = 0; trainExample < data->rows; ++trainExample) {
            squaredDistances[trainExample].first = GetSquaredDistance(data, trainExample, target, targetExample); // Calcula la distancia
            squaredDistances[trainExample].second = trainExample; // Almacena el índice del ejemplo de entrenamiento
        }

        // Ordena los ejemplos de entrenamiento por distancia al cuadrado (más cercano primero)
        std::sort(squaredDistances.begin(), squaredDistances.end());

        // Crea un vector para contar las clases de los k vecinos más cercanos
        size_t nClasses = target->numLabels;
        std::vector<int> countClosestClasses(nClasses, 0); // Inicializa el contador a cero

        // Cuenta las instancias de cada clase entre los k vecinos más cercanos
        for (int i = 0; i < k; ++i) {
            int currentClass = data->label(squaredDistances[i].second); // Obtiene la clase del vecino más cercano
            countClosestClasses[currentClass]++; // Incrementa el contador de la clase
        }

        // Calcula la probabilidad de cada clase para el ejemplo objetivo
        for (size_t i = 0; i < nClasses; ++i) {
            results->pos(targetExample, i) = static_cast<double>(countClosestClasses[i]) / k; // Probabilidad de la clase
        }
    }

    // Copia las etiquetas esperadas desde el conjunto objetivo al conjunto de resultados
    for (size_t i = 0; i < target->rows; ++i) {
        results->label(i) = target->label(i);
    }

    return KNNResults(results); // Devuelve los resultados del KNN
}
