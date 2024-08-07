#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include<chrono>

#include "dicom/DicomReader.h"
#include "dicom/dicomutils.h"

#include "knn/dataset.h"
#include "knn/knn.h"
#include "knn/knnUtils.h"
#include "knn/Preprocessing.h"
#include "benchmark.h"

using namespace std;
using namespace std::chrono;


// Function to calculate precision
double calculatePrecision(const vector<int>& trueLabels, const vector<int>& predictedLabels) {
    if (trueLabels.size() != predictedLabels.size()) {
        cerr << "Error: Size of true labels and predicted labels must be equal." << endl;
        return -1;
    }

    int truePositive = 0;
    int falsePositive = 0;

    for (size_t i = 0; i < trueLabels.size(); ++i) {
        if (predictedLabels[i] == 1) {
            if (trueLabels[i] == 1) {
                truePositive++;
            } else {
                falsePositive++;
            }
        }
    }

    return truePositive / static_cast<double>(truePositive + falsePositive);
}

int main()
{

    auto start = high_resolution_clock::now();
    // La ruta depende de cada maquina
    DicomReader dicomObj("/home/will/Projects/dicom-classifier/data/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm");


    vector<vector<double>> data = dicomObj.getDoubleImageMatrix(12);
    vector<int> dataLabels = DicomUtils::genTargetValues(data, 2);

    for (const auto& row : data) {
        for (const auto& value : row) {
            if (std::isnan(value) || std::isinf(value)) {
                cerr << "Invalid value found in data" << endl;
                return -1;
            }
        }
    }

    vector<int> testIdx = DicomUtils::genTestDataIdx(data, 50);
    vector<int> testLabels = DicomUtils::getTestingLabels(dataLabels, testIdx);
    vector<vector<double>> testData = DicomUtils::getTestingValues(data, testIdx);

    double* parsedTrainData = DicomUtils::parseKNNData(data);
    int* parsedTrainLabel = DicomUtils::parseKNNLabels(dataLabels);

    double* parsedTestData = DicomUtils::parseKNNData(testData);
    int* parsedTestLabel = DicomUtils::parseKNNLabels(testLabels);

    int rows = data.size();
    int cols = data[0].size();

    int numLabels = 2;
    DatasetPointer dataset = makeDataset(rows, cols, numLabels, parsedTrainData, parsedTrainLabel);
    DatasetPointer test = makeDataset(testData.size(), cols, numLabels, parsedTestData, parsedTestLabel);

    MatrixPointer meanData = MeanNormalize(dataset);
    ApplyMeanNormalization(test, meanData);

    KNN knn(dataset);
    int bestK = 2;
    KNNResults results = knn.run(bestK, test);

    vector<int> predictedLabels;
    MatrixPointer predictions = results.getPredictions();

    for (size_t i = 0; i < predictions->rows; ++i) {
        for (size_t j = 0; j < predictions->cols; ++j) {
            double pred = predictions->pos(i, j);
            if (std::isnan(pred)) {
                cerr << "NaN found in predictions at (" << i << ", " << j << ")" << endl;
                return -1;
            }
            predictedLabels.push_back(static_cast<int>(pred));
        }
    }

    double precision = calculatePrecision(testLabels, predictedLabels);
    cout << "Precision of the predictions: " << precision << endl;

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start );
    cout << "Execution time: " << duration.count() << " milliseconds" << endl;

    return 0;
}


