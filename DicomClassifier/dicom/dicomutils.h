#ifndef DICOMUTILS_H
#define DICOMUTILS_H

#include <vector>
#include <string>
#include <iostream>

using namespace std;

class DicomUtils
{
public:
    DicomUtils();
    static vector<string> getDicomFilesPath(const char *path);
    static int getDataWidth(vector<string> dicomFiles);
    static int getDataHeight(vector<string> dicomFiles);
    static string base_name(string const & path, string const &delims);
    static double **asFCMPointsData(int **arr, int numPoints, int numDims);
};

#endif // DICOMUTILS_H