#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <vector>
#include <math.h>

#include <omp.h>
#include "performance/benchmark.h"
#include "hybridmedianfilter/hybridmedianfilter.h"
#include "dicom/DicomReader.h"

using namespace std;

int main()
{
    cout << "Hello World!" << endl;
    //enviando numero de hilos en los subprocesos posteriores
    omp_set_num_threads(8);

    //vector que tendra el path de las imagenes originales dicom
    vector<string> src_origen;
    //vector que tendra el path donde se guardaran los archivos csv de las imagenes aplicadas el filtro
    vector<string> src_destino;
    //vector donde se guardara la data, es decir, la imagen
    vector<vector<int>> data;

    //creando variables char para los path de las imagenes, para poder aplicar en los objetos de la libreria Dicom
    char src_img_filtrada_char[300];
    char src_img_char[300];

    //definir archivo donde se escribiran las metricas
    string filename("/home/jair/Documents/metricasHibridMedianFilter/metricasParaleloProbandoConLibreriaW.txt");
    //abrir archivo
    fstream file_out;
    file_out.open(filename, std::ios_base::app);


    //creando objeto para metricas, en este caso de la libreria metrics
    systemMetrics metricas("filtro");



    //creando variables para capturar metricas
    //obteniendo el numero de hilos que existite en los subprocesos
    int nHilos = omp_get_max_threads();
    int ram;
    double cpu;
    int procesadores=6;
    double tbegin, time;
    double cpu_time;



    //agregar path de imagen original
    src_origen.push_back("/home/jair/Pictures/datasetDicom/MasaMicro1.dcm");
    //agregar path donde se guardara el archivo de aplicar el filtro
    src_destino.push_back("/home/jair/Pictures/datasetDicomR/MasaMicro1hmf7.csv");

    src_origen.push_back("/home/jair/Pictures/datasetDicom/MasaMicro2.dcm");
    src_destino.push_back("/home/jair/Pictures/datasetDicomR/MasaMicro2hmf7.csv");

    src_origen.push_back("/home/jair/Pictures/datasetDicom/MasaMicro3.dcm");
    src_destino.push_back("/home/jair/Pictures/datasetDicomR/MasaMicro3hmf7.csv");

    src_origen.push_back("/home/jair/Pictures/datasetDicom/MasaMicro4.dcm");
    src_destino.push_back("/home/jair/Pictures/datasetDicomR/MasaMicro4hmf7.csv");

    src_origen.push_back("/home/jair/Pictures/datasetDicom/MasaMicro5.dcm");
    src_destino.push_back("/home/jair/Pictures/datasetDicomR/MasaMicro5hmf7.csv");


    int tamanoCaja=7;




    //for para recorrer las imagenes
    for(int i=0; i<src_origen.size(); i++){
        //cambiar de string a char[] de la imagen de origen
        strcpy(src_img_char,src_origen[i].c_str());
        //leyendo imagen de origen Dicom
        DicomReader dicomObj(src_img_char);
        //la convierte en vector de vectores int
        vector<vector<int>> data = dicomObj.getIntImageMatrix(12);
        for(int j=0; j<1 ;j++){
            //medicion de metricas
            //resetear las metricas en 0
            metricas.resetCounters();
            //inicio del tiempo de ejecucion del filtro
           tbegin = omp_get_wtime();
                //ejecucion del filtro//
                data=hybridmedianfilterN(data,tamanoCaja);
           //tomar tiempo final de ejecucion y calcularlo
           time = omp_get_wtime() - tbegin;
           //calcular las metricas del objeto de metrics
           metricas.calculate();
           //obtener el tiempo de cpu del objeto metricas
           cpu_time=metricas.getDurationInSeconds();
           //obtener el uso de ram de metricas
           ram=getRamUsage();
           //obtener porcentaje de cpu
           cpu=metricas.getCpuPercent();

           //imprimiendo metricas obtenidas
           cout<<"-----"<<endl<<"Metricas:"<<endl;
           cout << "Tiempo de ejecucion Del Filtro: " << time <<endl;
           cout <<"Tiempo de cpu:"<<cpu_time<<endl;
           cout<<"ram:"<<ram<<"KB"<<endl;
           cout<<"cpu%:"<<cpu<<endl;
           cout<<"N hilos:"<<nHilos<<endl;
           cout<<"Procesadores:"<<6<<endl;

           if (!file_out.is_open()) {
               //si el archivo no se ha abrido
               cout << "failed to open " << filename << '\n';
           } else {
               //escribir metricas en archivo csv
               file_out<<src_origen[i]<<","<<tamanoCaja<<","<<ram<<","<<cpu<<","<<time<<","<<cpu_time<<","<<nHilos<<","<<procesadores<<","<<j<<endl;
           }


           //guardar el en archivo csv
           //convertir path de imagen donde se guardara de string a char[]
           strcpy(src_img_filtrada_char,src_destino[i].c_str());
           //escribiendo archivo
           //saveData(vector de vectores, path de imagen en char, separador entre comillas, si se quiere cabecera en booleano)
           dicomObj.saveData(data,src_img_filtrada_char,",",false);

        }
    }

    return 0;
}
