#include "dicom/DicomReader.h"
using namespace std;
#ifndef CLAHEH_H
#define CLAHEH_H
	vector<vector<int>> newMatriz(DicomReader dicomObj);
	vector<vector<int>> newCLAHE_GO(vector<vector<int>> data);
	vector<vector<int>> claheGo(vector<vector<int>> CLAHE_GO,vector<vector<int>> data,DicomReader dicomObj);
#endif // CLAHEH_H
