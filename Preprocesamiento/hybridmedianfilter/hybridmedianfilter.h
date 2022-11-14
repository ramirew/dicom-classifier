#ifndef _HYBRIDMEDIANFILTER_H_
#define _HYBRIDMEDIANFILTER_H_

using namespace std;
typedef int element;
//metodo del filtro mediano hibrido
vector<vector<element>> hybridmedianfilterN(vector<vector<element>> image, int ventana=3);
//metodo para pasar un archivo .csv de una imagen dicom sin cabecera a vector de vectores int
vector<vector<element>> csvAvector(string path, char delimitador=',');
#endif

