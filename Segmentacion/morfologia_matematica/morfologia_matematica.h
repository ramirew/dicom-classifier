#ifndef _MORFOLOGIAMATEMATICA_H_
#define _MORFOLOGIAMATEMATICA_H_

using namespace std;
typedef int element;

//INSTANCIACIACIÓN DEL FILTRO DE DILATACIÓN
vector<vector<element>> dilatacion(vector<vector<element>> image,vector<vector<element>> elementoEstructurante, int mostrar=1);
//INSTANCIACIACIÓN DEL FILTRO DE EROSIÓN
vector<vector<element>> erosion(vector<vector<element>> image,vector<vector<element>> elementoEstructurante, int mostrar=1);
//INSTANCIACIACIÓN DEL FILTRO DE APERTURA
vector<vector<element>> apertura(vector<vector<element>> image,vector<vector<element>> elementoEstructurante);
//INSTANCIACIACIÓN DEL FILTRO DE CIERRE
vector<vector<element>> cierre(vector<vector<element>> image,vector<vector<element>> elementoEstructurante);
//INSTANCIAR LA FUNCIÓN PARA DETERMINAR EL TIPO DE KERNEL A UTILIZAR
vector<vector<element>> determinarKernel(int dimension, int tipo);

#endif
