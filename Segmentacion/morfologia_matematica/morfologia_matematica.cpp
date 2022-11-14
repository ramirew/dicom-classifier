#include <stdio.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <ctime>
#include "stdlib.h"
#include "string.h"
#include "stdio.h"
#include "sys/times.h"
#include "sys/vtimes.h"

using namespace std;
typedef int element;

int obtenerUsoDeRAM();
int conversorDatos(char* line);

double obtenerUsoDeCPU();
void init();

// Autores: Alvaro Lomas, Jhony Pillajo, Joel Romo
// Descripción del filtro: Este filtro se encarga de eliminar algún tipo de ruido presente en determinada imagen, ya sea utilizando una dilatación
// (obtener el mayor valor de una ventana superpuesta sobre la imagen) o una erosión(obtener el menor valor de una ventana superpuesta sobre la imagen),
// de tal forma que se eliminan pequeñas zonas de la imagen donde exista información poco relevante. Además, también se cuenta con filtros mixtos como
// son apertura(aplica primero una erosión y después una dilatación) y cierre(aplica primero una dilatación y después una erosión).

/*MÉTODOS*/
//FUNCIÓN QUE HAYA EL VALOR MÁXIMO AL RECORRER KERNEL SOBRE LA MATRIZ DE IMAGEN
//ENVÍA COMO PARÁMETRO EL KERNEL DE TIPO vector<element>
// PARÁMETROS DE LA FUNCIÓN: elements
element maximo(vector<element> elements) {
   // PARÁMETRO 1: elements -> Matriz de elementos que corresponde a la ventana recorrida sobre la imagen.
   //DECLARA E INICIALIZA LA VARIABLE DE COMPARACIÓN
   element max = -1;
   //COMPARACIÓN DE ELEMENTOS
   for(int i = 0; i < elements.size() ; ++i) {
      if(elements[i] > max) {
          //GUARDAR EL MAYOR VALOR DE LA VENTANA
          max = elements[i];
      }
   }
   //RETORNAR EL VALOR MÁXIMO DE LA VENTANA
   return max;
}

//FUNCIÓN QUE HAYA EL VALOR MÍNIMO AL RECORRER KERNEL SOBRE LA MATRIZ DE IMAGEN
//ENVÍA COMO PARÁMETRO EL KERNEL DE TIPO vector<element>
// PARÁMETROS DE LA FUNCIÓN: elements
element minimo(vector<element> elements) {
   // PARÁMETRO 1: elements -> Matriz de elementos que corresponde a la ventana recorrida sobre la imagen.
   //DECLARA E INICIALIZA LA VARIABLE DE COMPARACIÓN
   element min = 1000000;
   //COMPARACIÓN DE ELEMENTOS
   for(int i = 0; i < elements.size(); ++i) {
      if(elements[i] < min) {
          //GUARDAR EL MENOR VALOR DE LA VENTANA
          min = elements[i];
      }
   }
   //RETORNAR EL VALOR MÍNIMO DE LA VENTANA
   return min;
}

//CREAR UN ELEMENTO ESTRUCTURANTE DEPENDIENDO DE LA DIMENSIÓN Y TIPO SOLICITADO:
//1 -> KERNEL CUADRADO
//2 -> KERNEL HORIZONTAL
//3 -> KERNEL VERTICAL
//4 -> KERNEL CRUZADO
// PARÁMETROS DE LA FUNCIÓN: dimension y tipo
vector<vector<element>> determinarKernel(int dimension, int tipo){
    // PARÁMETRO 1: dimension -> Valor entero que representa la dimensión del kernel a crear.
    // PARÁMETRO 2: tipo -> Valor entero que representa el tipo de kernel a crear según los tipos antes mencionados. 
   
    //CREAR E INICIALIZAR KERNEL
    vector<vector<int>> matriz(dimension, vector<int> (dimension));
   
    //OBTENER LA MITAD DEL KERNEL PARA POSTERIORMENTE LLENARLA DE VALORES 1 SEGÚN CORRESPONDA EL TIPO DE KERNEL
    double mitad = round(matriz.size()/2.0)-1;

    int i, j;
   //SI LA DIMENSIÓN INGRESADA ES UN NÚMERO IMPAR, ENTONCES SE PERMITE LA CREACIÓN DE UN KERNEL
    if(dimension % 2 != 0){
        //SE CREA UN KERNEL DEPENDIENDO EL TIPO SOLICITADO
        if(tipo==1){
            //KERNEL CUADRADO
            #pragma omp parallel for collapse(2) shared(matriz) private(i,j) schedule(static)
            for (i = 0; i < matriz.size(); ++i) {
                for (j = 0; j < matriz[0].size(); ++j) {
                    //LLENA TODA LA MATRIZ DE VALORES 1
                    matriz[i][j]=1;
                }
            }
        } else if(tipo==2){
            //KERNER HORIZONTAL
            #pragma omp parallel for collapse(2) shared(matriz) private(i,j) schedule(static)
            for (i = 0; i < matriz.size(); ++i) {
                for (j = 0; j < matriz[0].size(); ++j) {
                    //SI LA POSICIÓN CORRESPONDE A LA MITAD DEL ALTO ENTONCES SE LLENA DE VALORES 1
                    if(i==mitad){
                        matriz[i][j]=1;
                    } else {
                        matriz[i][j]=0;
                    }
                }
            }
        } else if(tipo==3){
            //KERNEL VERTICAL
            #pragma omp parallel for collapse(2) shared(matriz) private(i,j) schedule(static)
            for (i = 0; i < matriz.size(); ++i) {
                for (j = 0; j < matriz[0].size(); ++j) {
                    //SI LA POSICIÓN CORRESPONDE A LA MITAD DEL ANCHO ENTONCES SE LLENA DE VALORES 1
                    if(j==mitad){
                        matriz[i][j]=1;
                    } else {
                        matriz[i][j]=0;
                    }
                }
            }
        } else if(tipo==4){
            //KERNEL CRUZADO
            #pragma omp parallel for collapse(2) shared(matriz) private(i,j) schedule(static)
            for (i = 0; i < matriz.size(); ++i) {
                for (j = 0; j < matriz[0].size(); ++j) {
                   //SI LA POSICIÓN CORRESPONDE YA SEA A LA MITAD DEL ALTO O LA MITAD DEL ANCHO ENTONCES SE LLENA DE VALORES 1
                    if(j==mitad || i==mitad){
                        matriz[i][j]=1;
                    } else {
                        matriz[i][j]=0;
                    }
                }
            }
        } else {
            //SI SE INGRESA UN KERNEL QUE NO EXISTE, ENTONCES SE ENVÍA UN MENSAJE DE ERROR Y TERMINA EL PROGRAMA
            cout << "Debe ingresar un tipo de kernel válido" << endl;
            exit(-1);
        }
    } else {
       //SI LA DIMENSIÓN INGRESADA NO ES UN NÚMERO IMPAR, ENTONCES SE ENVÍA UN MENSAJE DE ERROR Y TERMINA EL PROGRAMA
        cout << "Debe ingresar una dimensión de número impar" << endl;
        exit(-1);
    }
    //RETORNAR LA MATRIZ DEL ELEMENTO ESTRUCTURANTE SEGÚN LOS PARÁMETROS ESPECIFICADOS
    return matriz;
}

//RECIBE LA IMAGEN Y EL ELEMENTO ESTRUCTURANTE O KERNEL, AMBOS DE TIPO (vector<vector<element>>)
// PARÁMETROS DE LA FUNCIÓN: image y elementoEstructurante
vector<vector<element>> dilatacion(vector<vector<element>> image, vector<vector<element>> elementoEstructurante, int mostrar=1) {   
   // PARÁMETRO 1: image -> Matriz de imagen que representa los píxeles que posee la imagen dicom.
   // PARÁMETRO 2: elementoEstructurante -> Matriz que representa un tipo de ventana a recorrer sobre la matriz de imagen.
   //VARIABLES PARA MEDIR EL TIEMPO DE EJECUCIÓN
    double tbegin;
    if(mostrar == 1){
        tbegin = omp_get_wtime();
    }
    
    //DECLARA E INICIALIZA LA VARIABLE DONDE SE ALMACENARÁ LA IMAGEN YA FILTRADA
    vector<vector<element>> img_result = image;
    
    //OBTENER ANCHO (N) Y ALTO (M) DE LA IMAGEN ENVIADA
    element N = image[0].size();
    element M = image.size();
    
    //OBTENER ANCHO (N) Y ALTO (M) DEL ELEMENTO ESTRUCTURANTE
    element Nestructurante = elementoEstructurante[0].size();
    element Mestructurante = elementoEstructurante.size();

    int x,y,i,j;

    //OBTENER LA VECINDAD DEL ELEMENTO ESTRUCTURANTE (m: ANCHO y n: ALTO)
    int m = Mestructurante/2;
    int n = Nestructurante/2;

   //MOVER KERNEL A TRAVÉS DE TODOS LOS ELEMENTOS DE LA IMAGEN
   #pragma omp parallel for collapse(2) shared(img_result,image,m,n,M,N) private(x,y,i,j) schedule(static)
   for (y = m; y < M - m ; ++y)
      for (x = n; x < N - n; ++x) {
         //CREAR VENTANA A RECORRER CON EL KERNEL
         vector<element> window;
         int xx = x-n; int yy = y-m;
         for(i = 0; i < Mestructurante; i++) {
            for(j = 0; j < Nestructurante; j++) {
                if(elementoEstructurante[i][j] == 1) {
                    //GUARDAR ELEMENTOS DE LA IMAGEN RECORRIDA EN LA VENTANA SUPERPUESTA
                    window.push_back(image[yy][xx]);
                }
                xx++;
            }
            xx = x-n;
            yy++;
         }
         //OBTENER EL VALOR MÁXIMO DE LA VENTANA QUE RECORRE EL KERNEL
         img_result[y][x] = maximo(window);
      }
      
   //TOMAR TIEMPO FINAL Y CALCULARLO
   if(mostrar == 1){
       double time = omp_get_wtime() - tbegin;
       cout << "Execution Time Del Filtro de Dilatación: " << time << "; Max Threads:" << omp_get_max_threads() <<endl;
       cout << "RAM " << obtenerUsoDeRAM() << " KB" << endl;
       cout << "CPU " << obtenerUsoDeCPU() << " %" << endl;
   }
   //RETORNAR LA MATRIZ DE IMAGEN RESULTANTE
   return img_result;
}

//RECIBE LA IMAGEN Y EL ELEMENTO ESTRUCTURANTE O KERNEL, AMBOS DE TIPO (vector<vector<element>>)
// PARÁMETROS DE LA FUNCIÓN: image y elementoEstructurante
vector<vector<element>> erosion(vector<vector<element>> image, vector<vector<element>> elementoEstructurante, int mostrar=1) {
    // PARÁMETRO 1: image -> Matriz de imagen que representa los píxeles que posee la imagen dicom.
    // PARÁMETRO 2: elementoEstructurante -> Matriz que representa un tipo de ventana a recorrer sobre la matriz de imagen.
    //VARIABLES PARA MEDIR EL TIEMPO DE EJECUCIÓN
    double tbegin;
    if(mostrar == 1){
        tbegin = omp_get_wtime();
    }
    
    //DECLARA E INICIALIZA LA VARIABLE DONDE SE ALMACENARÁ LA IMAGEN YA FILTRADA
    vector<vector<element>> img_result = image;
    
    //OBTENER ANCHO (N) Y ALTO (M) DE LA IMAGEN ENVIADA
    element N = image[0].size();
    element M = image.size();
    
    //OBTENER ANCHO (N) Y ALTO (M) DEL ELEMENTO ESTRUCTURANTE
    element Nestructurante = elementoEstructurante[0].size();
    element Mestructurante = elementoEstructurante.size();

    int x,y,i,j;

    //OBTENER LA VECINDAD DEL ELEMENTO ESTRUCTURANTE (m: ANCHO y n: ALTO)
    int m = Mestructurante/2;
    int n = Nestructurante/2;

   //MOVER KERNEL A TRAVÉS DE TODOS LOS ELEMENTOS DE LA IMAGEN
   #pragma omp parallel for collapse(2) shared(img_result,image,m,n,M,N) private(x,y,i,j) schedule(static)
   for (y = m; y < M - m ; ++y)
      for (x = n; x < N - n; ++x) {
         //CREAR VENTANA A RECORRER CON EL KERNEL
         vector<element> window;
         int xx = x-n; int yy = y-m;
         for(i = 0; i < Mestructurante; i++) {
            for(j = 0; j < Nestructurante; j++) {
                if(elementoEstructurante[i][j] == 1) {
                    //GUARDAR ELEMENTOS DE LA IMAGEN RECORRIDA EN LA VENTANA SUPERPUESTA
                    window.push_back(image[yy][xx]);
                }
                xx++;
            }
            xx = x-n;
            yy++;
         }
         //OBTENER EL VALOR MÍNIMO DE LA VENTANA QUE RECORRE EL KERNEL
         img_result[y][x] = minimo(window);
      }

   //TOMAR TIEMPO FINAL Y CALCULARLO
   if(mostrar == 1){
       double time = omp_get_wtime() - tbegin;
       cout << "Execution Time Del Filtro de Erosión: " << time << "; Max Threads:" << omp_get_max_threads() <<endl;
       cout << "RAM " << obtenerUsoDeRAM() << " KB" << endl;
       cout << "CPU " << obtenerUsoDeCPU() << " %" << endl;
   }
   //RETORNAR LA MATRIZ DE IMAGEN RESULTANTE
   return img_result;
}

//RECIBE LA IMAGEN Y EL ELEMENTO ESTRUCTURANTE O KERNEL, AMBOS DE TIPO (vector<vector<element>>)
// PARÁMETROS DE LA FUNCIÓN: image y elementoEstructurante
vector<vector<element>> apertura(vector<vector<element>> image, vector<vector<element>> elementoEstructurante) {
    // PARÁMETRO 1: image -> Matriz de imagen que representa los píxeles que posee la imagen dicom.
    // PARÁMETRO 2: elementoEstructurante -> Matriz que representa un tipo de ventana a recorrer sobre la matriz de imagen.
    //VARIABLES PARA MEDIR EL TIEMPO DE EJECUCIÓN
    double tbegin = omp_get_wtime();
    
    //DECLARAR LA VARIABLE DONDE SE ALMACENARÁ LA IMAGEN YA FILTRADA
    vector<vector<element>> img_result;
    
    //APLICAR EL FILTRO DE EROSIÓN Y A ESTA IMAGEN RESULTANTE EL FILTRO DE DILATACIÓN, SE ENVÍA 0 PARA NO MOSTRAR EL BENCHMARK
    img_result = erosion(image, elementoEstructurante, 0);
    img_result = dilatacion(img_result, elementoEstructurante,0);

   //TOMAR TIEMPO FINAL Y CALCULARLO
    double time = omp_get_wtime() - tbegin;
    cout << "Execution Time Del Filtro de Apertura: " << time << "; Max Threads:" << omp_get_max_threads() << endl;
    cout << "RAM " << obtenerUsoDeRAM() << " KB" << endl;
    cout << "CPU " << obtenerUsoDeCPU() << " %" << endl;
    
    //RETORNAR LA MATRIZ DE IMAGEN RESULTANTE
    return img_result;
}

//RECIBE LA IMAGEN Y EL ELEMENTO ESTRUCTURANTE O KERNEL, AMBOS DE TIPO (vector<vector<element>>)
// PARÁMETROS DE LA FUNCIÓN: image y elementoEstructurante
vector<vector<element>> cierre(vector<vector<element>> image,vector<vector<element>> elementoEstructurante) {
    // PARÁMETRO 1: image -> Matriz de imagen que representa los píxeles que posee la imagen dicom.
    // PARÁMETRO 2: elementoEstructurante -> Matriz que representa un tipo de ventana a recorrer sobre la matriz de imagen.
    //VARIABLES PARA MEDIR EL TIEMPO DE EJECUCIÓN
    double tbegin = omp_get_wtime();
 
    //DECLARAR LA VARIABLE DONDE SE ALMACENARÁ LA IMAGEN YA FILTRADA
    vector<vector<element>> img_result;

    //APLICAR EL FILTRO DE DILATACIÓN Y A ESTA IMAGEN RESULTANTE EL FILTRO DE EROSIÓN, SE ENVÍA 0 PARA NO MOSTRAR EL BENCHMARK
    img_result = dilatacion(image, elementoEstructurante, 0);
    img_result = erosion(img_result, elementoEstructurante, 0);

   //TOMAR TIEMPO FINAL Y CALCULARLO
    double time = omp_get_wtime() - tbegin;
    cout << "Execution Time Del Filtro de Cierre: " << time <<"; Max Threads:" << omp_get_max_threads() <<endl;
    cout << "RAM " << obtenerUsoDeRAM() << " KB" << endl;
    cout << "CPU " << obtenerUsoDeCPU() << " %" << endl;
    
    //RETORNAR LA MATRIZ DE IMAGEN RESULTANTE
    return img_result;
}

//CONVIERTE LOS DATOS DE TEXTO EN ENTEROS
int conversorDatos(char* line){
    int i = strlen(line);
    while (*line < '0' || *line > '9') line++; line[i-3] = '\0'; i = atoi(line);
    return i;
}

//CALCULAR EL USO DE MEMORIA RAM
int obtenerUsoDeRAM(){ //NOTA: VALORES EN KB
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = conversorDatos(line);
            break;
        }
    }
    fclose(file);
    return result;
}

//DECLARAR VARIABLES PARA OBTENER EL USO DE CPU
static unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;

//TOMAR VARIABLES DEL SISTEMA E INICIALIZAR VARIABLES A UTILIZAR PARA OBTENER EL USO DE CPU
void init(){
    FILE* file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %Ld %Ld %Ld %Ld", &lastTotalUser, &lastTotalUserLow, &lastTotalSys, &lastTotalIdle);
    fclose(file);
}

//CALCULAR EL PORCENTAJE DE USO DE CPU
double obtenerUsoDeCPU(){
    double percent;
    FILE* file;
    unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;
    file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %Ld %Ld %Ld %Ld", &totalUser, &totalUserLow, &totalSys, &totalIdle);
    fclose(file);
    if (totalUser < lastTotalUser || totalUserLow < lastTotalUserLow || totalSys < lastTotalSys || totalIdle < lastTotalIdle){
        percent = -1.0;
    } else {
        total = (totalUser - lastTotalUser) + (totalUserLow - lastTotalUserLow) + (totalSys - lastTotalSys);
        percent = total;
        total += (totalIdle - lastTotalIdle);
        percent /= total;
        percent *= 100;
    }
    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;
    return percent;
}
