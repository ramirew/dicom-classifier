#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <vector>
#include <math.h>
#include <memory.h>
#include <omp.h>

//   Image element type
using namespace std;
typedef int element;

//   MEDIAN calculation
//     elements - input elements
//     N        - number of input elements
//obtener la mediana de elementos
//elements son los numeros int
//N numero de elementos 
element median(element* elements, int N)
{
   //   Order elements (only half of them)
   for (int i = 0; i < (N >> 1) + 1; ++i)
   {
      //   Find position of minimum element
      int min = i;
      for (int j = i + 1; j < N; ++j)
         if (elements[j] < elements[min])
            min = j;
      //   Put found minimum element in its place
      const element temp = elements[i];
      elements[i] = elements[min];
      elements[min] = temp;
   }
   //   Get result - the middle element
   return elements[N >> 1];
}


//entrada del metodo hybridmedianfilterN
// image: vector de vectores int 
// ventana: valor de la longitud de la caja con la que se aplica el filtro
vector<vector<element>> hybridmedianfilterN(vector<vector<element>> image, int ventana)
{

//se define el MN dependiendo de la longitud de la caja
    int MN_vecindad;
    if(ventana < 3){
        MN_vecindad=3;
    }else{
        if(ventana % 2 ==0){
            ventana = ventana + 1;
        }
    }

    MN_vecindad = (ventana-1)/2;


//se crea una imagen resultante igual a la imagen de entrada
     vector<vector<element>> img_result = image;
    //N width, M height
    //N ancho de la imagen, es decir, numero de columnas
    //M largo de la imagen, es decir, numero de filas
    element N = image[0].size();
    element M = image.size();
    //tamano de los elementos de la ventana
    int tW = (MN_vecindad*4) + 1;

//creando arrays para la ventana X y CRUZ
    element window[tW];
//creando arrays para guardar mediana de X, de CRUZ y el elemento central
    element results[3];

//definiendo variables privadas para utilizar en los for 
    int x,y,posicion;

   //   Move window through all elements of the image
   #pragma omp parallel for collapse(2) shared(img_result,image,M,N,MN_vecindad) private(window,results,x,y,posicion) schedule(static)
   for (y = MN_vecindad; y < M - MN_vecindad; ++y)
      for (x = MN_vecindad; x < N - MN_vecindad; ++x)
      {


         //   Pick up cross-window elements
         posicion=0;
         window[0] = image[y][x];
         for(int i=1; i<=MN_vecindad; i++){
             window[1+posicion] = image[(y-i)][x];
             window[2+posicion] = image[y][(x-i)];
             window[3+posicion] = image[y][(x+i)];
             window[4+posicion] = image[(y+i)][x];
             posicion=i*4;
         }
         //   Get median
         results[0] = median(window, tW);

         //   Pick up x-window elements
         window[0] = image[y][x];
         posicion=0;
         for(int i=1; i<=MN_vecindad; i++){
             window[1+posicion] = image[(y-i)][(x-i)];
             window[2+posicion] = image[(y-i)][(x+i)];
             window[3+posicion] = image[(y+i)][(x-i)];
             window[4+posicion] = image[(y+i)][(x+i)];
             posicion=i*4;
         }
         //   Get median
         results[1] = median(window, tW);
         //   Pick up leading element
         results[2] = image[y][x];
         //   Get result
         img_result[y][x] = median(results, 3);
      }

   return img_result;
}


//pasar csv a vector<vector<element>>
vector<vector<element>> csvAvector(string path, char delimitador){
    vector<vector<element>> vec;

    //leer el archivo
    //ejemplo string path="/home/jair/Documents/ejemplo.csv";
    ifstream archivo(path);
    //delimitador ejemplo
    //char delimitador = ',';

    //linea que sera leida, va leyendo cada fila del csv
    string linea;

    //funcion getline
    //getline(variable tipo stringstream, variable a guardar la linea, delimitador)
    //stringstream -> el archivo o un string se lo puede convertir
    //variable a guardar -> se guarda lo que encuentre hasta que llegue al delimitador
    //delimitador -> el delimitador por defecto es salto de linea, se puede poner coma o punto y coma o otro


    while(getline(archivo,linea)){
        //en linea se guarda cada linea en cada pasada
        //cout << linea << endl;
        //la linea se pasa a un stringstream para poder usar getline
        stringstream stream(linea);
        //este while para cada columna de la fila
        string col;
        vector<element> v1;
        while(getline(stream,col,delimitador)){
            element p = stoi(col);
            v1.push_back(p);
        }
        vec.push_back(v1);
    }

    return vec;
}
