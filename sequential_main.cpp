#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "Dot.h"
#include "Cluster.h"
#include "fstream"
using namespace std;


int number_of_dots = 500;
int  number_of_clusters = 10;
int iterations = 20;

vector<Dot> create_point(int number_of_dots);
vector<Cluster> create_cluster(int number_of_clusters);
void plot(vector<Dot> &dots);

int main() {
    srand(time(NULL));
    printf("Number of dots %d\n",number_of_dots);
    printf("Number of Clusters %d\n",number_of_clusters);

    double start_time = omp_get_wtime();
    printf("Initialization \n");

    printf("Creation of the dots \n");
    vector<Dot> pts = create_point(number_of_dots);
    printf("dots Created \n");

    printf("Creations of the Clusters \n");
    vector<Cluster> cls = create_cluster(number_of_clusters);
    printf("Clusters Created \n");

    double end_time1 = omp_get_wtime();
    auto duration = end_time1 - start_time;
    printf("Initialization made in: %f seconds\n",duration);

    int iteration_num = 0;
    bool iterate = true;
    printf("-STARTING ITERATE-\n");

    while(iteration_num < iterations && iterate){
        iteration_num ++;
        double cluster_start_time = omp_get_wtime();
        // Updating of clusters centroids
        printf("Iteration %d \n",iteration_num);
        double cluster_end_time = omp_get_wtime();
        auto cluster_duration = cluster_end_time - cluster_start_time;
        printf("Clusters Update made in: %f seconds\n",cluster_duration);
    }
    double end_time2 = omp_get_wtime();
    duration = end_time2 - end_time1;
    printf("Number of iterarions %d, total time %f seconds, iteration time avg %f seconds \n", iteration_num,duration, duration/iteration_num);
    printf("Storing the dots coordinates and cluster-id...\n");
    plot(pts);
    return 0;
}

void plot(vector<Dot> &dots){
    ofstream Myfile("data.txt");
    Myfile << "x,y,cluster_id"<< endl ;
    for(int i = 0; i < dots.size(); i++){
        Dot point = dots[i];
        Myfile << point.get_x() << "," << point.get_y() << "," << point.get_cluster_id() << endl;
    }
    Myfile.close();
}