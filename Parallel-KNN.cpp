#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <fstream>
#include "Dot.h"
#include "Cluster.h"
using namespace std;

int number_of_dots = 500;
int  number_of_clusters = 10;
int iterations = 20;
double max_value = 100000;
int number_of_threads = 10;

vector<Dot> create_point(int num_pt);
vector<Cluster> create_cluster(int num_cl);
void find_distance(vector<Dot> &pts, vector<Cluster> &cls);
double euclidean_dist(Dot pt, Cluster cl);
bool update_clusters(vector<Cluster> &cls);
void plot(vector<Dot> &points);

