#ifndef CANNY_H
#define CANNY_H

#include <iostream>
#include <vector>

using namespace std;

void jhon();

int jhonsuma(int a,int b);

vector<vector<double>> createFilter(int row, int column, double sigmaIn);
vector<vector<int>> useFilter(vector<vector<int>> img_in, vector<vector<double>> filterIn);
vector<vector<int>> sobel(vector<vector<int>> gFiltered);
vector<vector<int>> anglesMap(vector<vector<int>> gFiltered);
vector<vector<int>> nonMaxSupp(vector<vector<int>> sFiltered,vector<vector<int>> angles2);
vector<vector<int>> threshold(vector<vector<int>> imgin,int low, int high);
vector<vector<int>> canny(vector<vector<int>> img_in,int low, int high);
#endif
