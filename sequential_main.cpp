#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "Dot.h"
#include "Cluster.h"
#include "fstream"
using namespace std;


int number_of_points = 500;
int  number_of_clusters = 10;

vector<Dot> create_point(int number_of_points);
vector<Cluster> create_cluster(int number_of_clusters);


int main() {

    srand(time(NULL));
    printf("Number of Points %d\n",number_of_points);
    printf("Number of Clusters %d\n",number_of_clusters);

    double start_time = omp_get_wtime();
    printf("Initialization \n");

    printf("Creation of the Points \n");
    vector<Dot> pts = create_point(number_of_points);
    printf("Points Created \n");

    printf("Creations of the Clusters \n");
    vector<Cluster> cls = create_cluster(number_of_clusters);
    printf("Clusters Created \n");

    double end_time1 = omp_get_wtime();

    auto duration = end_time1 - start_time;
    printf("Initialization made in: %f seconds\n",duration);

    int iteration_num = 0;
    bool iterate = true;
    printf("-STARTING ITERATE-\n");



    return 0;
}
