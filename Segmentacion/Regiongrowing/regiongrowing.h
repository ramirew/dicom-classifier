//REGION GROWING
//Autores: Jonathan Adriano, Kevin Jimenez
//FECHA: 26 Julio 2022

#ifndef _REGIONGROWING_H_
#define _REGIONGROWING_H_
#include <vector>
using namespace std;
typedef int element;
int reg_size;
int reg_mean;
int x;
int y;
vector<vector<element>> regiongrowing(vector<vector<element>> src,int x,int y, int reg_size, int reg_mean);

#endif
