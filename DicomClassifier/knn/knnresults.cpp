#include "knn.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <limits>

using namespace std;


//Ejecucion de resultados
SingleExecutionResults KNNResults::top1Result(){
    int nSuccess = 0;
    int nRejected = 0;

    //definicion de la matriz de punteros
    MatrixPointer pred = getPredictions();

    //0Prediccion de etiquetas en base a los ejemplos obtenidos
    for (size_t currentExample = 0; currentExample < results->rows; currentExample++) {
        int predictedLabel = static_cast<int>(pred->pos(currentExample, 0));
        int actualLabel = results->label(currentExample);
        //Numero de etiquetas rechazadas y etiquetas aprobadas
        if (predictedLabel == -1) {
            nRejected++;
        } else if (predictedLabel == actualLabel) {
            nSuccess++;
        }
    }

    return SingleExecutionResults(results->rows, nSuccess, nRejected);
}

//Ejecucion de resultados basados en el algoritmo KNN
SingleExecutionResults KNNResults::topXResult(int n) {
    int nSuccess = 0;
    int nRejected = 0;
    // Calculo de resultados en base a los ejemplos por filas y columnas de la resolucion actual
    for (size_t currentExample = 0; currentExample < results->rows; currentExample++) {
        vector<pair<double, int>> resultsForExample(results->cols);

        //CAlculo parcial de resultado en base al ejemplo actual
        for (size_t j = 0; j < results->cols; j++) {
            resultsForExample[j] = {results->pos(currentExample, j), j};
        }

        //Ordenamiento parcial de resultados parciales
        partial_sort(resultsForExample.begin(), resultsForExample.begin() + n, resultsForExample.end(), greater<pair<double, int>>());
        //Etiquetado en base al ejemplo total
        int actualLabel = results->label(currentExample);
        if (any_of(resultsForExample.begin(), resultsForExample.begin() + n, [actualLabel](const pair<double, int>& result) {
                return result.second == actualLabel;
            })) {
            nSuccess++;
        }
    }


    return SingleExecutionResults(results->rows, nSuccess, nRejected);
}

//Obtencion de resultados de la matrix de punteros
MatrixPointer KNNResults::getPredictions() {

    //predicciones de las matrices en base a las ejecuciones parciales compartidas
    MatrixPointer predictions = make_shared<matrix_base>(results->rows, 1);

    //CAlculo de matrices restantes en el ejemplo actual
    for (size_t currentExample = 0; currentExample < results->rows; currentExample++) {
        double maxProbability = -numeric_limits<double>::infinity();
        int maxIndex = -1;
        bool rejecting = false;
        for (size_t j = 0; j < results->cols; j++) {
            double currentProbability = results->pos(currentExample, j);
            //Probabilidad actual en base a la probabilidad maxima para rechazo
            if (currentProbability > maxProbability) {
                maxIndex = j;
                maxProbability = currentProbability;
                rejecting = false;
            } else if (currentProbability == maxProbability) {
                rejecting = true;
            }
        }
        if (rejecting) {
            maxIndex = -1;
        }

        predictions->pos(currentExample, 0) = maxIndex;
    }

    return predictions;
}

//Obtencion de matriz de confusion
MatrixPointer KNNResults::getConfusionMatrix() {
    MatrixPointer pred = getPredictions();
    MatrixPointer confusion = make_shared<matrix_base>(results->cols, results->cols);
    confusion->clear();
    //Obtener datos de la matriz de confusion actual
    for (size_t currentExample = 0; currentExample < results->rows; currentExample++) {
        int predicted = static_cast<int>(pred->pos(currentExample, 0));
        int actual = results->label(currentExample);

        if (predicted != -1 && predicted != actual) {
            confusion->pos(actual, predicted)++;
        }
    }

    return confusion;
}
