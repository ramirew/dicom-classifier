#define _USE_MATH_DEFINES

#include <iostream>

#include <cmath>

#include <vector>
#include "omp.h"


using namespace std;

//Autores: Jhon Idrovo, Andres Ponces, Edison López
/*Descripción del método:El algoritmo de detección de bordes de Canny utiliza un filtro basado en la primera derivada de una gaussiana.
Ya que es susceptible al ruido presente en datos de imagen sin procesar, la imagen original es transformada con un filtro gaussiano
El borde de una imagen puede apuntar en diferentes direcciones, por lo que el algoritmo de Canny utiliza cuatro filtros para detectar horizontal, vertical y diagonal en los bordes de la imagen borrosa.
Además utiliza la Supresión de falsos máximos siendo que esta técnica es utilizada para afinar los bordes encontrados en el paso anterior. 
Básicamente, consiste en escanear la imagen para eliminar los píxeles que no formen parte de los bordes. 
Para esto se compara el valor de cada píxel con sus vecinos cercanos en la dirección del gradiente (perpendicular al borde).
El procedimiento anterior logra determinar los píxeles que conforman los bordes con bastante precisión. 
Sin embargo, aún pueden quedar algunos píxeles provenientes del ruido o  de variaciones en los colores de la imagen. 
En esta cuarta estapa se decide cuáles píxeles pertenecen realmente a bordes y cuáles no.*/

/*Metodos*/
//Función que permite crear una mascara de 3x3, siendo útil de el desarrollo del filtro gaussiano
vector<vector<double>> createFilter(int row, int column, double sigmaIn)

    {
		//Declara la variable donde se va a guardar la matriz principal
        vector<vector<double>> filter;
	#pragma opm parallel num_threads(2) 
	{	
		//Comparación entre filas y colunas
        for (int i = 0; i < row; i++)
        {	
			//Declara las columnas
            vector<double> col;
            for (int j = 0; j < column; j++)

            {
				//Retorna la última posición de la columna
                col.push_back(-1);

            }
			//Retorna todas las columnas del filtro
            filter.push_back(col);
		}
		//Declara la variable para sumar las posiciones de cada uno de los valores de la matriz
        float coordSum = 0;
		// Dentro de la variable s, se guarda la optimización del sigma.
        float constant = 2.0 * sigmaIn * sigmaIn;



        // Sumatoria para la normalización

        float sum = 0.0;
		//
        for (int x = - row/2; x <= row/2; x++)

        {

            for (int y = -column/2; y <= column/2; y++)

            {

                coordSum = (x*x + y*y);

                filter[x + row/2][y + column/2] = (exp(-(coordSum) / constant)) / (M_PI * constant);

                sum += filter[x + row/2][y + column/2];

            }

        }



        // Normalización del Filtro
		//#pragma opm parallel for collapse (2)
        for (int i = 0; i < row; i++)

            for (int j = 0; j < column; j++)

                filter[i][j] /= sum;
	}
        return filter;
	
    }
	
//Función que establece el filtro gaussiano en base a la mascará requerida(3x3)  
    vector<vector<int>> useFilter(vector<vector<int>> img_in, vector<vector<double>> filterIn)

    {
		//Establece el limite del tamaño de la matriz
        int size = (int)filterIn.size()/2;
        vector<vector<int>> filteredImg=img_in;
	#pragma omp parallel num_threads(2)
    {
        #pragma opm parallel for collapse (2)
		//Aplicamos la operación de convolución
        for (int i = size; i < img_in.size() - size; i++)

        {

            for (int j = size; j < img_in[i].size() - size; j++)

            {

                double sum = 0;

                for (int x = 0; x < filterIn.size(); x++)

                    for (int y = 0; y < filterIn.size(); y++)

                    {
						//En la variable sum guardamos los datos de los pixeles resultantes de aplicar el filtro
                        sum += filterIn[x][y] * (double)(img_in[i + x - size][j + y - size]);

                    }



                filteredImg[i-size][j-size] = sum;

            }



        }
	}
	//Retorna la matriz resultante del filtro gaussiano
        return filteredImg;
	
    }

	//Función de sobel es utilizado en procesamiento de imágenes, especialmente en algoritmos de detección de bordes. 
	//Técnicamente es un operador diferencial discreto que calcula una aproximación al gradiente de la función de intensidad de una imagen.
    vector<vector<int>> sobel(vector<vector<int>> gFiltered){
		//Establecemos un kernel de 3x3 elementos para aplicar convolución a la imagen original
		//Filtro de Sobel X
        double x1[] = {-1.0, 0, 1.0};

        double x2[] = {-2.0, 0, 2.0};

        double x3[] = {-1.0, 0, 1.0};


		//Definimos la variable de la gradiente horizontal
        vector<vector<double>> xFilter(3);

        xFilter[0].assign(x1, x1+3);

        xFilter[1].assign(x2, x2+3);

        xFilter[2].assign(x3, x3+3);

        //Filtro de Sobel Y

        double y1[] = {1.0, 2.0, 1.0};

        double y2[] = {0, 0, 0};

        double y3[] = {-1.0, -2.0, -1.0};


		//Definimos la variable de la gradiente vertical
        vector<vector<double>> yFilter(3);

        yFilter[0].assign(y1, y1+3);

        yFilter[1].assign(y2, y2+3);

        yFilter[2].assign(y3, y3+3);



        //Nuevamente limitamos el tamaño de la matriz

        int size = (int)xFilter.size()/2;
		
		//Llamamos al anterior filtro(Filtro Gaussiano)
        vector<vector<int>> filteredImg=gFiltered;
		
		//Declaramos una variable para definir la dirección de la gradiente
        vector<vector<int>> angles=gFiltered;

        //vector<vector<int>> angles;

		//int i,j,x,y;
	#pragma opm parallel num_threads(32) 
	{
		#pragma opm parallel for collapse (2)
        for (int i = size; i < gFiltered.size() - size; i++)

        {

            for (int j = size; j < gFiltered[i].size() - size; j++)

            {

                double sumx = 0;

                double sumy = 0;

                for (int x = 0; x < xFilter.size(); x++)

                    for (int y = 0; y < xFilter.size(); y++)

                    {

                        sumx += xFilter[x][y] * (double)(gFiltered[i + x - size][j + y - size]); //Sobel_X Filter Value

                        sumy += yFilter[x][y] * (double)(gFiltered[i + x - size][j + y - size]); //Sobel_Y Filter Value

                    }

                double sumxsq = sumx*sumx;

                double sumysq = sumy*sumy;



                double sq2 = sqrt(sumxsq + sumysq);



                if(sq2 > 4096) //Unsigned Char Fix

                    sq2 =4096;

                filteredImg[i-size][j-size] = sq2;



                if(sumx==0) //Arctan Fix

                    angles[i-size][j-size] = 90;  //Magnitud del vector Gradiente.

                else

                    angles[i-size][j-size] = atan(sumy/sumx); //Direccion del vector Gradiente.

            }

        }
	}
	//Retorno la matriz de la imagen con la función de sobel
        return filteredImg;
	
    }

//Función para encontrar la dirección del vector gradiente, es similar al función anterior de sobel.
    vector<vector<int>> anglesMap(vector<vector<int>> gFiltered){

        double x1[] = {-1.0, 0, 1.0};

        double x2[] = {-2.0, 0, 2.0};

        double x3[] = {-1.0, 0, 1.0};



        vector<vector<double>> xFilter(3);

        xFilter[0].assign(x1, x1+3);

        xFilter[1].assign(x2, x2+3);

        xFilter[2].assign(x3, x3+3);

        //Sobel Y Filter

        double y1[] = {1.0, 2.0, 1.0};

        double y2[] = {0, 0, 0};

        double y3[] = {-1.0, -2.0, -1.0};



        vector<vector<double>> yFilter(3);

        yFilter[0].assign(y1, y1+3);

        yFilter[1].assign(y2, y2+3);

        yFilter[2].assign(y3, y3+3);



        //Limitar el tamaño de la matriz.

        int size = (int)xFilter.size()/2;

        vector<vector<int>> filteredImg=gFiltered;


        vector<vector<int>> angles=gFiltered;
		
		//int i,j,x,y;
	#pragma opm parallel num_threads(32) 
	{
		#pragma opm parallel for collapse (2)
        for (int i = size; i < gFiltered.size() - size; i++)

        {

            for (int j = size; j < gFiltered[i].size() - size; j++)

            {

                double sumx = 0;

                double sumy = 0;

                for (int x = 0; x < xFilter.size(); x++)

                    for (int y = 0; y < xFilter.size(); y++)

                    {

                        sumx += xFilter[x][y] * (double)(gFiltered[i + x - size][j + y - size]); //Sobel_X Filter Value

                        sumy += yFilter[x][y] * (double)(gFiltered[i + x - size][j + y - size]); //Sobel_Y Filter Value

                    }

                double sumxsq = sumx*sumx;

                double sumysq = sumy*sumy;



                double sq2 = sqrt(sumxsq + sumysq);



                if(sq2 > 4096) //Unsigned Char Fix

                    sq2 =4096;

                filteredImg[i-size][j-size] = sq2;



                if(sumx==0) //Arctan Fix

                    angles[i-size][j-size] = 90;

                else

                    angles[i-size][j-size] = atan(sumy/sumx);

            }

        }
	}
	//Retorno la matriz de la imagen con la función de gradiente de intensidad
        return angles;
	
    }

/*Función de Supresión de no máximos: en este paso se logra el adelgazamiento del ancho de los bordes, obtenidos con el gradiente, hasta lograr bordes de un píxel de ancho*/
    vector<vector<int>> nonMaxSupp(vector<vector<int>> sFiltered,vector<vector<int>> angles2)

   {

       vector<vector<int>> nonMaxSupped=sFiltered;

       vector<vector<int>> angles=angles2;
	#pragma opm parallel num_threads(32) 
	{
		#pragma opm parallel for collapse (2)
          for (int i=1; i< sFiltered.size() - 1; i++) {

              for (int j=1; j<sFiltered[i].size() - 1; j++) {

                  float Tangent = angles[i][j];



                  nonMaxSupped[i-1][j-1] = sFiltered[i][j];

                  //Bordes Horizontales

                  if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))

                  {

                      if ((sFiltered[i][j] < sFiltered[i][j+1]) || (sFiltered[i][j] < sFiltered[i][j-1]))

                          nonMaxSupped[i-1][j-1] = 0;

                  }

                  //Bordes Verticales

                  if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))

                  {

                      if ((sFiltered[i][j] < sFiltered[i+1][j]) || (sFiltered[i][j] < sFiltered[i-1][j]))

                          nonMaxSupped[i-1][j-1] = 0;

                  }



                  //Bordes que estan posicionados en -45 grados

                  if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))

                  {

                      if ((sFiltered[i][j] < sFiltered[i-1][j+1]) || (sFiltered[i][j] < sFiltered[i+1][j-1]))

                          nonMaxSupped[i-1][j-1] = 0;

                  }



                  //Bordes que estan posicionados en 45 grados

                  if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))

                  {

                      if ((sFiltered[i][j] < sFiltered[i+1][j+1]) || (sFiltered[i][j]< sFiltered[i-1][j-1]))

                          nonMaxSupped[i-1][j-1] = 0;

                  }

              }

          }
	}
		//Retorno una matriz de imagen con bordes muy finos
       return nonMaxSupped;
	
   }

//Función threshold: Los píxeles correspondientes a bordes con valores de gradientes que se encuentren entre estos dos umbrales son etiquetados como píxeles débiles. 
//Estos últimos serán o no aceptados, dependiendo de su conectividad. Si están conectados a píxeles “fuertes”, se consideran parte de los bordes; de lo contrario, también se descartan.
   vector<vector<int>> threshold(vector<vector<int>> imgin,int low, int high)

   {

       if(low > 4096)

           low = 1;
       if(high > 4096)

           high = 1;



       vector<vector<int>> EdgeMat=imgin;

	#pragma opm parallel num_threads(32) 
	{
		#pragma opm parallel for collapse (2)
       for (int i=0; i<imgin.size(); i++)

       {

           for (int j = 0; j<imgin[i].size(); j++)

           {

               EdgeMat[i][j] = imgin[i][j];
				//Si el pixelMayor es superior al valor de la matriz, entonces los valores de la matriz son reemplazados por 1
               if(EdgeMat[i][j] > high)

                   EdgeMat[i][j] = 1;
				//Si el pixelMenor es menor al valor de la matriz, entonces los valores de la matriz son reemplazados por 0
               else if(EdgeMat[i][j] < low)

                   EdgeMat[i][j] = 0;

               else

               {

                   bool anyHigh = false;

                   bool anyBetween = false;
					
                   for (int x=i-1; x < i+2; x++)

                   {

                       for (int y = j-1; y<j+2; y++)

                       {

                           if(x <= 0 || y <= 0 || x > EdgeMat.size() || y > EdgeMat[x].size()) //Fuera de los límites

                               continue;

                           else

                           {
							   //Si el pixelMayor es superior al valor de la matriz, entonces los valores de la matriz son reemplazados por 1

                               if(EdgeMat[x][y] > high)

                               {

                                   EdgeMat[i][j] = 1;

                                   anyHigh = true;

                                   break;

                               }
								//Si el pixelMenor es superior al valor de la matriz, entonces los valores de la matriz son reemplazados por 1
                               else if(EdgeMat[x][y] >= low)

                                   anyBetween = true;

                           }

                       }

                       if(anyHigh)

                           break;

                   }

                   if(!anyHigh && anyBetween)
					   
                       for (int x=i-2; x < i+3; x++)

                       {

                           for (int y = j-1; y<j+3; y++)

                           {

                               if(x < 0 || y < 0 || x > EdgeMat.size() || y > EdgeMat[x].size()) //Fuera de los límites

                                   continue;

                               else

                               {
								//Si el pixelMayor es superior al valor de la matriz, entonces los valores de la matriz son reemplazados por 1
                                   if(EdgeMat[x][y] > high)

                                   {

                                       EdgeMat[i][j] = 1;

                                       anyHigh = true;

                                       break;

                                   }

                               }

                           }

                           if(anyHigh)

                               break;

                       }
					//Si los valores de la matriz son diferentes al valor de pixelMayor, entonces va a tomar valores de 0  
                   if(!anyHigh)

                       EdgeMat[i][j] = 0;

               }

           }

       }
	}
	//Retorna la matriz con valores de 0 y 1 donde "0" es la pixel débil y "1" es el pixel mayor. 
       return EdgeMat;
	
   }
vector<vector<int>>canny(vector<vector<int>> img_in,int low, int high){
	vector<vector<double>> filter=createFilter(3,3,1);

	vector<vector<int>> gFiltered=useFilter(img_in,filter);

	vector<vector<int>> sFiltered=sobel(gFiltered);

	vector<vector<int>> angles=anglesMap(gFiltered);

	vector<vector<int>> non=nonMaxSupp(sFiltered,angles);

	vector<vector<int>> thres=threshold(non,low,high);
	
	return thres;
	
}

