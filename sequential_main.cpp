#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "Dot.h"
#include "Cluster.h"
#include "fstream"
using namespace std;


int number_of_points = 500000;
int  number_of_clusters = 1000;

vector<Dot> create_point(int number_of_points);
vector<Cluster> create_cluster(int number_of_clusters);