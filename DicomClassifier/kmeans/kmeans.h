#ifndef KMEANS_H
#define KMEANS_H


#include <omp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

// Clase para representar un punto en el espacio de características
class Point
{
private:
    int pointId, clusterId; // ID del punto y ID del clúster al que pertenece
    int dimensions;        // Número de dimensiones del punto
    vector<double> values; // Valores de las dimensiones del punto

    // Convierte una línea de texto en un vector de valores numéricos
    vector<double> lineToVec(string &line)
    {
        vector<double> values; // Vector para almacenar los valores convertidos
        string tmp = "";       // Cadena temporal para construir valores numéricos

        for (int i = 0; i < (int)line.length(); i++)
        {
            // Verifica si el carácter es un dígito, punto decimal, signo o 'e' para notación científica
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e')
            {
                tmp += line[i]; // Construye el número en formato string
            }
            else if (tmp.length() > 0)
            {
                values.push_back(stod(tmp)); // Convierte el string a double y lo añade al vector
                tmp = ""; // Resetea la cadena temporal
            }
        }
        if (tmp.length() > 0)
        {
            values.push_back(stod(tmp)); // Añade el último valor si existe
        }

        return values;
    }

public:
    // Constructor que inicializa un punto a partir de una línea de texto
    Point(int id, string line)
    {
        pointId = id;
        values = lineToVec(line);
        dimensions = values.size();
        clusterId = 0; // Inicialmente no asignado a ningún clúster
    }

    // Constructor que inicializa un punto a partir de un vector de valores
    Point(int id, vector<double> dataPoints)
    {
        pointId = id;
        values = dataPoints;
        dimensions = dataPoints.size();
        clusterId = 0;
    }

    int getDimensions() const { return dimensions; } // Obtiene el número de dimensiones

    int getCluster() const { return clusterId; } // Obtiene el ID del clúster al que pertenece el punto
    int getID() const { return pointId; } // Obtiene el ID del punto

    void setCluster(int val) { clusterId = val; } // Establece el ID del clúster al que pertenece el punto

    double getVal(int pos) const { return values[pos]; } // Obtiene el valor en la posición dada
};

// Clase para representar un clúster
class Cluster
{
private:
    int clusterId;                // ID del clúster
    vector<double> centroid;      // Centroide del clúster
    vector<Point*> points;        // Punteros a los puntos que pertenecen al clúster

public:
    // Constructor que inicializa un clúster con un punto como centroide
    Cluster(int clusterId, Point* centroid)
    {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid->getDimensions(); i++)
        {
            this->centroid.push_back(centroid->getVal(i)); // Copia las coordenadas del centroide
        }
        this->addPoint(centroid); // Añade el punto de centroide al clúster
    }

    // Añade un punto al clúster
    void addPoint(Point* p)
    {
        p->setCluster(this->clusterId); // Establece el ID del clúster del punto
        points.push_back(p); // Añade el puntero al vector de puntos
    }

    // Elimina un punto del clúster por su ID
    bool removePoint(int pointId)
    {
        int size = points.size();
        for (int i = 0; i < size; i++)
        {
            if (points[i]->getID() == pointId)
            {
                points.erase(points.begin() + i); // Elimina el punto del vector
                return true;
            }
        }
        return false; // No se encontró el punto
    }

    // Elimina todos los puntos del clúster
    void removeAllPoints() { points.clear(); }

    int getId() const { return clusterId; } // Obtiene el ID del clúster

    Point* getPoint(int pos) const { return points[pos]; } // Obtiene el puntero al punto en la posición dada
    int getSize() const { return points.size(); } // Obtiene el número de puntos en el clúster

    double getCentroidByPos(int pos) const { return centroid[pos]; } // Obtiene el valor del centroide en la posición dada

    void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; } // Establece el valor del centroide en la posición dada

    // Destructor
    ~Cluster() {
        // En este caso, no destruimos los punteros a `Point` aquí para evitar eliminar objetos compartidos
    }
};

// Clase para gestionar el algoritmo de K-Means
class KMeans
{
private:
    int K, iters, dimensions, total_points; // Número de clústeres, iteraciones, dimensiones y puntos totales
    vector<Cluster*> clusters; // Punteros a los clústeres
    string output_dir; // Directorio de salida para los archivos

    // Limpia todos los puntos de todos los clústeres
    void clearClusters()
    {
        for (auto cluster : clusters)
        {
            cluster->removeAllPoints(); // Limpia los puntos de cada clúster
        }
    }

    // Obtiene el ID del clúster más cercano al punto dado
    int getNearestClusterId(const Point &point)
    {
        double min_dist = numeric_limits<double>::max(); // Inicializa la distancia mínima a infinito
        int NearestClusterId = -1;

        for (auto cluster : clusters)
        {
            double sum = 0.0;
            for (int j = 0; j < dimensions; j++)
            {
                double diff = cluster->getCentroidByPos(j) - point.getVal(j); // Calcula la diferencia en cada dimensión
                sum += diff * diff; // Suma de las distancias cuadradas
            }
            double dist = sqrt(sum); // Distancia euclidiana

            if (dist < min_dist) // Si la distancia es menor que la mínima encontrada
            {
                min_dist = dist;
                NearestClusterId = cluster->getId(); // Actualiza el ID del clúster más cercano
            }
        }
        return NearestClusterId; // Devuelve el ID del clúster más cercano
    }

public:
    // Constructor que inicializa el número de clústeres, iteraciones y directorio de salida
    KMeans(int K, int iterations, const string &output_dir)
        : K(K), iters(iterations), output_dir(output_dir)
    {
        // No se necesita asignación dinámica aquí
    }

    // Ejecuta el algoritmo de K-Means
    void run(vector<Point> &all_points)
    {
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions(); // Asume que todos los puntos tienen la misma dimensión

        vector<int> used_pointIds; // Vector para almacenar los IDs de los puntos ya usados
        for (auto cluster : clusters) {
            delete cluster; // Limpia los punteros a clústeres existentes
        }
        clusters.clear(); // Asegura que no queden clústeres sobrantes

        // Inicializa los clústeres con puntos aleatorios
        for (int i = 1; i <= K; i++)
        {
            while (true)
            {
                int index = rand() % total_points; // Selecciona un índice aleatorio
                if (find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end())
                {
                    used_pointIds.push_back(index); // Añade el índice a los usados
                    all_points[index].setCluster(i); // Asigna el clúster al punto
                    Cluster* cluster = new Cluster(i, &all_points[index]); // Crea un nuevo clúster
                    clusters.push_back(cluster); // Añade el clúster al vector de clústeres
                    break;
                }
            }
        }

        int iter = 1;
        while (iter <= iters)
        {
            cout << "Iteration " << iter << "/" << iters << endl;
            bool done = true;

            // Sección paralela para actualizar las asignaciones de clústeres
            //#pragma omp parallel for reduction(&& : done)
            for (int i = 0; i < total_points; i++)
            {
                int currentClusterId = all_points[i].getCluster(); // Obtiene el clúster actual del punto
                int nearestClusterId = getNearestClusterId(all_points[i]); // Encuentra el clúster más cercano

                if (currentClusterId != nearestClusterId) // Si el clúster del punto cambió
                {
                    all_points[i].setCluster(nearestClusterId); // Actualiza el clúster del punto
                    done = false; // Marca el algoritmo como no convergido
                }
            }

            clearClusters(); // Limpia los puntos de los clústeres

            // Reasigna los puntos a los clústeres
            for (int i = 0; i < total_points; i++)
            {
                clusters[all_points[i].getCluster() - 1]->addPoint(&all_points[i]); // Añade el punto al clúster correspondiente
            }

            // Actualiza los centroides de los clústeres
            for (auto cluster : clusters)
            {
                int ClusterSize = cluster->getSize();
                for (int j = 0; j < dimensions; j++)
                {
                    double sum = 0.0;
                    if (ClusterSize > 0)
                    {
                        //#pragma omp parallel for reduction(+: sum)
                        for (int p = 0; p < ClusterSize; p++)
                        {
                            sum += cluster->getPoint(p)->getVal(j); // Suma los valores en la dimensión dada
                        }
                        cluster->setCentroidByPos(j, sum / ClusterSize); // Calcula el nuevo centroide
                    }
                }
            }

            if (done) // Si no hubo cambios en los clústeres
            {
                cout << "Converged in iteration " << iter << endl;
                break;
            }
            iter++; // Incrementa el número de iteración
        }
    }

    // Obtiene los clústeres
    vector<Cluster*> getClustersValues() const
    {
        return clusters;
    }

    // Guarda los puntos en un archivo
    void savePoints(const vector<Point> &all_points, const string &prefix) const
    {
        ofstream pointsFile(output_dir + "/" + prefix + to_string(K) + "-points.txt");
        if (pointsFile.is_open())
        {
            for (const auto &point : all_points)
            {
                pointsFile << point.getCluster() << endl; // Guarda el ID del clúster de cada punto
            }
            pointsFile.close();
        }
    }

    // Guarda los centroides de los clústeres en un archivo
    void saveClusters(const string &prefix) const
    {
        ofstream outfile(output_dir + "/" + prefix + to_string(K) + "-clusters.txt");
        if (outfile.is_open())
        {
            for (const auto &cluster : clusters)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    outfile << cluster->getCentroidByPos(j) << " "; // Guarda los valores del centroide
                }
                outfile << endl;
            }
            outfile.close();
        }
    }

    // Obtiene el valor promedio de los centroides
    double getAvgClusters() const
    {
        double accum = 0;
        for (const auto &cluster : clusters)
        {
            for (int j = 0; j < dimensions; j++)
            {
                accum += cluster->getCentroidByPos(j); // Suma los valores de los centroides
            }
        }
        return accum / (K * dimensions); // Devuelve el promedio
    }

    // Destructor
    ~KMeans()
    {
        // Limpia los punteros a clústeres para evitar fugas de memoria
        for (auto cluster : clusters)
        {
            delete cluster;
        }
    }
};
#endif // KMEANS_H
