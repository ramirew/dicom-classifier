#include "fcm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

FCM::FCM(double fuzziness, double epsilon)
{
    this->fuzziness = fuzziness;
    this->epsilon = epsilon;
}

void FCM::init(double **data, int clusters, int num_points, int num_dimensions)
{
    if (num_clusters > MAX_CLUSTER) {
        printf("Number of clusters should be < %d\n", MAX_CLUSTER);
        exit(1);
    }
    if (num_data_points > MAX_DATA_POINTS) {
        printf("Number of data points should be < %d\n", MAX_DATA_POINTS);
        exit(1);
    }
    if (num_dimensions > MAX_DATA_DIMENSION) {
        printf("Number of dimensions should be >= 1.0 and < %d\n",MAX_DATA_DIMENSION);
        exit(1);
    }

    this->num_clusters = clusters;
    this->num_data_points = num_points;
    this->num_dimensions = num_dimensions;

    for (int i = 0; i < this->num_data_points; i++) {
        for (int j = 0; j < this->num_dimensions; j++) {
            this->data_point[i][j] = data[i][j];
            if (data[i][j] < low_high[j][0])
                low_high[j][0] = data[i][j];
            if (data[i][j] > low_high[j][1])
                low_high[j][1] = data[i][j];
        }
    }

    double s;
    int i, j, r, rval;

    for (i = 0; i < num_data_points; i++) {
        s = 0.0;
        r = 100;
        for (j = 1; j < num_clusters; j++) {
            rval = rand() % (r + 1);
            r -= rval;
            degree_of_memb[i][j] = rval / 100.0;
            s += degree_of_memb[i][j];
        }
        degree_of_memb[i][0] = 1.0 - s;
    }
}

void FCM::eval() {
    double max_diff;
    do {
        calculate_centre_vectors();
        max_diff = update_degree_of_membership();
    } while (max_diff > epsilon);
}

double **FCM::getCenters()
{
    static double **centers;
    centers= new double*[num_clusters];

    for (int i = 0; i < this->num_clusters; ++i) {
        centers[i] = new double[num_dimensions];
        for (int j = 0; j < this->num_dimensions; ++j) {
            centers[i][j] = cluster_centre[i][j];
        }
    }

    return centers;
}

double **FCM::getMembershipMatrix()
{
    double **membershipMatrix;
    membershipMatrix = new double*[num_data_points];

    for (int i = 0; i < this->num_data_points; ++i) {
        membershipMatrix[i] = new double[num_clusters];
        for (int j = 0; j < this->num_clusters; ++j) {
            membershipMatrix[i][j] = degree_of_memb[i][j];
//            cout << membershipMatrix[i][j] << " " ;
        }

//        cout << endl;
    }
    return membershipMatrix;
}

void FCM::saveMembershipMatrixU(const char* name)
{
        double **U;
        U = getMembershipMatrix();
        std::ofstream output_file2(name);
        for (int i = 0; i < num_dimensions; ++i) {
            for (int j = 0; j < num_clusters; ++j) {
    //            cout << U[i][j] << " " ;
                output_file2 << U[i][j] << "," ;
            }
            output_file2  << "\n";
    //        cout << endl;
        }
        output_file2.close();

}

void FCM::saveCenters(const char *name)
{
    double **centers;
    centers = getCenters();

    std::ofstream output_file(name);

//     Save results as csv
    for (int i = 0; i < num_clusters; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            output_file << centers[i][j] << ",";
        }
        output_file << "\n";
    }
    output_file.close();

}

double FCM::getCenterAVG()
{
    double avg = 0;
    double **centers;
    centers = getCenters();

    for (int i = 0; i < num_clusters; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            avg += centers[i][j];
        }
    }

    avg = avg / (num_clusters * num_dimensions);

    return avg;
}

void FCM::saveClusters(const char* prefix)
{
    int i, j, cluster;
    char fname[100];
    double highest;
    FILE * f[MAX_CLUSTER];

    for (j = 0; j < num_clusters; j++) {
        sprintf(fname, "cluster.%d", j);
        if ((f[j] = fopen(fname, "w")) == NULL) {
            printf("Could not create %s\n", fname);
            for (i = 0; i < j; i++) {
                fclose(f[i]);
                sprintf(fname, "cluster.%d", i);
                remove(fname);
            }
            return;
        }
        fprintf(f[j], "#Data points for cluster: %d\n", j);
    }
    for (i = 0; i < num_data_points; i++) {
        cluster = 0;
        highest = 0.0;
        for (j = 0; j < num_clusters; j++) {
            if (degree_of_memb[i][j] > highest) {
                highest = degree_of_memb[i][j];
                cluster = j;
            }
        }
//        fprintf(f[cluster], "%lf %lf\n", data_point[i][0], data_point[i][1]);
        fprintf(f[cluster], "%d\n", i); // save index of point cluster
    }

    for (j = 0; j < num_clusters; j++) {
           fclose(f[j]);
       }

}

void FCM::calculate_centre_vectors() {
    int i, j, k;
    double numerator, denominator;
    double t[MAX_DATA_POINTS][MAX_CLUSTER];
    for (i = 0; i < num_data_points; i++) {
        for (j = 0; j < num_clusters; j++) {
            t[i][j] = pow(degree_of_memb[i][j], fuzziness);
        }
    }
    for (j = 0; j < num_clusters; j++) {
        for (k = 0; k < num_dimensions; k++) {
            numerator = 0.0;
            denominator = 0.0;
            for (i = 0; i < num_data_points; i++) {
                numerator += t[i][j] * data_point[i][k];
                denominator += t[i][j];
            }
            cluster_centre[j][k] = numerator / denominator;
//            std::cout << cluster_centre[j][k] << endl;
        }
    }
}


double FCM::update_degree_of_membership() {
    int i, j;
    double new_uij;
    double max_diff = 0.0, diff;
    for (j = 0; j < num_clusters; j++) {
        for (i = 0; i < num_data_points; i++) {
            new_uij = get_new_value(i, j);
            diff = new_uij - degree_of_memb[i][j];
            if (diff > max_diff)
                max_diff = diff;
            degree_of_memb[i][j] = new_uij;
        }
    }
    return max_diff;
}


double FCM::get_new_value(int i, int j) {
    int k;
    double t, p, sum;
    sum = 0.0;
    p = 2 / (fuzziness - 1);
    for (k = 0; k < num_clusters; k++) {
        t = get_norm(i, j) / get_norm(i, k);
        t = pow(t, p);
        sum += t;
    }
    return 1.0 / sum;
}

double FCM::get_norm(int i, int j) {
    int k;
    double sum = 0.0;
    for (k = 0; k < num_dimensions; k++) {
        sum += pow(data_point[i][k] - cluster_centre[j][k], 2);
    }
    return sqrt(sum);
}

