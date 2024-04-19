#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <fstream>
#include "knn-functions.h"
#include "Dot.h"
using namespace std;

int number_of_dots = 50000;
int  number_of_clusters = 100;
int iterations = 40;
double max_value = 100000;
int number_of_threads = 10;
string filename = "parallel_clusters.txt";

vector<Dot> create_dot(int number_of_dots, int max_value);
vector<Cluster> create_cluster(int number_of_clusters, int max_value);
void find_distance(vector<Dot> &pts, vector<Cluster> &cls);
double euclidean_dist(Dot pt, Cluster cl);
bool update_clusters(vector<Cluster> &cls);
void plot(vector<Dot> &points, string filename);


int main() {
    srand(time(NULL));
    printf("Number of Points %d\n",number_of_dots);
    printf("Number of Clusters %d\n",number_of_clusters);

    double start_time = omp_get_wtime();

    printf("Initialization \n");

    printf("Creation of the Points \n");
    vector<Dot> pts = create_dot(number_of_dots, max_value);
    printf("Points Created \n");

    printf("Creations of the Clusters \n");
    vector<Cluster> cls = create_cluster(number_of_clusters, max_value);
    printf("Clusters Created \n");

    double end_time1 = omp_get_wtime();

    auto duration = end_time1 - start_time;
    printf("Initialization made in: %f seconds\n",duration);

    int iteration_num=0;
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
    printf("Number of iterarions %d, total time %f seconds, iteration time avg %f seconds \n",
           iteration_num,duration, duration/iteration_num);
    printf("Storing the points coordinates and cluster-id...\n");
    plot(pts, filename);
    return 0;
}


void find_distance(vector<Dot>&pts,vector<Cluster>&cls){
    unsigned long pts_size = pts.size();
    unsigned long cls_size = cls.size();

    double min_dist;
    int min_index;
    int Thread_ID;
    #pragma omp parallel default(none) num_threads(4) private(min_dist, min_index,Thread_ID) firstprivate(pts_size, cls_size) shared(pts,cls)
        {
    #pragma omp for schedule(static,1000)
        for (int i = 0; i <pts_size ; ++i) {
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
}

// RUN WITH THE FOLLOWING
// clang++ -fopenmp -o parallel_knn Parallel-KNN.cpp -L/opt/homebrew/opt/llvm/lib -I/opt/homebrew/opt/llvm/include -Wl,-rpath,/opt/homebrew/opt/llvm/lib
// ./parallel_knn