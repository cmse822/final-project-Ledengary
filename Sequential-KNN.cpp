#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "Dot.h"
#include "knn-functions.h"
#include "fstream"
using namespace std;

int number_of_dots = 50000;
int  number_of_clusters = 100;
int iterations = 40;
double max_value = 100000;

vector<Dot> create_dot(int number_of_dots, int max_value);
vector<Cluster> create_cluster(int number_of_clusters, int max_value);
void plot(vector<Dot> &dots);
void find_distance(vector<Dot> &pts, vector<Cluster> &cls);
bool update_clusters(vector<Cluster> &cls);
double euclidean_dist(Dot pt, Cluster cl);

int main() {
    srand(time(NULL));
    printf("Number of dots %d\n",number_of_dots);
    printf("Number of Clusters %d\n",number_of_clusters);

    double start_time = omp_get_wtime();
    printf("Initialization \n");

    printf("Creation of the dots \n");
    vector<Dot> pts = create_dot(number_of_dots, max_value);
    printf("dots Created \n");

    printf("Creations of the Clusters \n");
    vector<Cluster> cls = create_cluster(number_of_clusters, max_value);
    printf("Clusters Created \n");

    double end_time1 = omp_get_wtime();
    auto duration = end_time1 - start_time;
    printf("Initialization made in: %f seconds\n",duration);

    int iteration_num = 0;
    bool iterate = true;
    printf("-STARTING ITERATE-\n");

    while(iteration_num < iterations && iterate){
        iteration_num ++;
        find_distance(pts,cls);
        double cluster_start_time = omp_get_wtime();
        iterate = update_clusters(cls);
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


void find_distance(vector<Dot>&pts,vector<Cluster>&cls){
        unsigned long pts_size = pts.size();
        unsigned long cls_size = cls.size();

        double min_dist;
        int min_index;

    for (int i = 0; i <pts_size ; i++) {
        Dot &current_point = pts[i];
        min_dist = euclidean_dist(current_point,cls[0]);
        min_index = 0;
        for (int j = 0; j < cls_size; j++) {
            Cluster &current_cluster = cls[j];
            double dist = euclidean_dist(current_point, current_cluster);
            if (dist<min_dist){
                min_dist = dist;
                min_index = j;
            }
        }
        pts[i].set_id(min_index);
        cls[min_index].add_point(pts[i]);
    }
 }