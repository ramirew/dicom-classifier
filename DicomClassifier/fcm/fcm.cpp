#include "fcm.h"

FCM::FCM(double fuzziness, double epsilon) {
    this->fuzziness = fuzziness;
    this->epsilon = epsilon;

    // Initialize pointers to nullptr to avoid issues
    data_point = nullptr;
    degree_of_memb = nullptr;
    cluster_centre = nullptr;
    low_high = nullptr;
}

FCM::~FCM() {
    // Destructor to free allocated memory
    if (data_point) {
        for (int i = 0; i < num_data_points; ++i) {
            delete[] data_point[i];
        }
        delete[] data_point;
    }

    if (degree_of_memb) {
        for (int i = 0; i < num_data_points; ++i) {
            delete[] degree_of_memb[i];
        }
        delete[] degree_of_memb;
    }

    if (cluster_centre) {
        for (int i = 0; i < num_clusters; ++i) {
            delete[] cluster_centre[i];
        }
        delete[] cluster_centre;
    }

    if (low_high) {
        delete[] low_high[0];
        delete[] low_high[1];
        delete[] low_high;
    }
}

void FCM::init(double **data, int clusters, int num_points, int num_dimensions) {
    // Validate parameters
    if (clusters > MAX_CLUSTER || num_points > MAX_DATA_POINTS || num_dimensions > MAX_DATA_DIMENSION) {
        std::cerr << "Error: Invalid parameters for FCM initialization.\n";
        exit(1);
    }

    // Free previous memory if already initialized
    if (data_point) {
        for (int i = 0; i < num_data_points; ++i) {
            delete[] data_point[i];
        }
        delete[] data_point;
    }

    if (degree_of_memb) {
        for (int i = 0; i < num_data_points; ++i) {
            delete[] degree_of_memb[i];
        }
        delete[] degree_of_memb;
    }

    if (cluster_centre) {
        for (int i = 0; i < num_clusters; ++i) {
            delete[] cluster_centre[i];
        }
        delete[] cluster_centre;
    }

    if (low_high) {
        delete[] low_high[0];
        delete[] low_high[1];
        delete[] low_high;
    }

    // Assign new dimensions
    this->num_clusters = clusters;
    this->num_data_points = num_points;
    this->num_dimensions = num_dimensions;

    // Allocate memory for data and membership matrices
    data_point = new double*[num_points];
    degree_of_memb = new double*[num_points];
    cluster_centre = new double*[clusters];
    low_high = new double*[2];

    low_high[0] = new double[num_dimensions];
    low_high[1] = new double[num_dimensions];

    for (int i = 0; i < num_points; ++i) {
        data_point[i] = new double[num_dimensions];
        degree_of_memb[i] = new double[clusters];
    }

    for (int i = 0; i < clusters; ++i) {
        cluster_centre[i] = new double[num_dimensions];
    }

    // Inicializacion de datos y matrices
    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            data_point[i][j] = data[i][j];
            if (data[i][j] < low_high[j][0])
                low_high[j][0] = data[i][j];
            if (data[i][j] > low_high[j][1])
                low_high[j][1] = data[i][j];
        }
    }

    double s;
    int r, rval;

    for (int i = 0; i < num_points; i++) {
        s = 0.0;
        r = 100;
        for (int j = 1; j < clusters; j++) {
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

double** FCM::getCenters() {
    return cluster_centre;
}

double** FCM::getMembershipMatrix() {
    return degree_of_memb;
}

void FCM::saveMembershipMatrixU(const char* name) {
    std::ofstream output_file(name);
    for (int i = 0; i < num_data_points; ++i) {
        for (int j = 0; j < num_clusters; ++j) {
            output_file << degree_of_memb[i][j] << ",";
        }
        output_file << "\n";
    }
    output_file.close();
}

void FCM::saveCenters(const char* name) {
    std::ofstream output_file(name);
    for (int i = 0; i < num_clusters; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            output_file << cluster_centre[i][j] << ",";
        }
        output_file << "\n";
    }
    output_file.close();
}

double FCM::getCenterAVG() {
    double avg = 0;
    for (int i = 0; i < num_clusters; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            avg += cluster_centre[i][j];
        }
    }
    avg /= (num_clusters * num_dimensions);
    return avg;
}

void FCM::saveClusters(const char* prefix) {
    char fname[100];
    FILE * f[MAX_CLUSTER];

    for (int j = 0; j < num_clusters; j++) {
        sprintf(fname, "%s.cluster.%d", prefix, j);
        if ((f[j] = fopen(fname, "w")) == NULL) {
            std::cerr << "Could not create " << fname << "\n";
            for (int i = 0; i < j; i++) {
                fclose(f[i]);
                sprintf(fname, "%s.cluster.%d", prefix, i);
                remove(fname);
            }
            return;
        }
        fprintf(f[j], "#Data points for cluster: %d\n", j);
    }

    for (int i = 0; i < num_data_points; i++) {
        int cluster = 0;
        double highest = 0.0;
        for (int j = 0; j < num_clusters; j++) {
            if (degree_of_memb[i][j] > highest) {
                highest = degree_of_memb[i][j];
                cluster = j;
            }
        }
        fprintf(f[cluster], "%d\n", i);
    }

    for (int j = 0; j < num_clusters; j++) {
        fclose(f[j]);
    }
}

void FCM::calculate_centre_vectors() {
    double t[MAX_DATA_POINTS][MAX_CLUSTER];
    for (int i = 0; i < num_data_points; i++) {
        for (int j = 0; j < num_clusters; j++) {
            t[i][j] = std::pow(degree_of_memb[i][j], fuzziness);
        }
    }

    for (int j = 0; j < num_clusters; j++) {
        for (int k = 0; k < num_dimensions; k++) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (int i = 0; i < num_data_points; i++) {
                numerator += t[i][j] * data_point[i][k];
                denominator += t[i][j];
            }
            cluster_centre[j][k] = numerator / denominator;
        }
    }
}

double FCM::update_degree_of_membership() {
    double max_diff = 0.0;
    for (int j = 0; j < num_clusters; j++) {
        for (int i = 0; i < num_data_points; i++) {
            double new_uij = get_new_value(i, j);
            double diff = new_uij - degree_of_memb[i][j];
            if (diff > max_diff)
                max_diff = diff;
            degree_of_memb[i][j] = new_uij;
        }
    }
    return max_diff;
}

double FCM::get_new_value(int i, int j) {
    double p = 2 / (fuzziness - 1);
    double sum = 0.0;
    for (int k = 0; k < num_clusters; k++) {
        double t = get_norm(i, j) / get_norm(i, k);
        t = std::pow(t, p);
        sum += t;
    }
    return 1.0 / sum;
}

double FCM::get_norm(int i, int j) {
    double sum = 0.0;
    for (int k = 0; k < num_dimensions; k++) {
        sum += std::pow(data_point[i][k] - cluster_centre[j][k], 2);
    }
    return std::sqrt(sum);
}
