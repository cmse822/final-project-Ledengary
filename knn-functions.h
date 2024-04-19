#ifndef KNN_FUNCTIONS_H
#define KNN_FUNCTIONS_H

#include "Dot.h"
#include "Cluster.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "Dot.h"
#include "fstream"
using namespace std;


void plot(vector<Dot> &dots, string filename){
    ofstream Myfile(filename);
    Myfile << "x,y,cluster_id"<< endl ;
    for(int i = 0; i < dots.size(); i++){
        Dot point = dots[i];
        Myfile << point.get_x() << "," << point.get_y() << "," << point.get_cluster_id() << endl;
    }
    Myfile.close();
}

double euclidean_dist(Dot pt, Cluster cl){
    double dist =sqrt(pow(pt.get_x() - cl.get_x(),2) +
                      pow(pt.get_y() - cl.get_y(), 2));
    return dist;
}

bool update_clusters(vector<Cluster>&cls) {
    bool  iterate = false;
    for (int i = 0; i < cls.size(); i++) {
        iterate = cls[i].update_values();
        cls[i].delete_values();
    }
    return iterate;
}

vector<Dot> create_dot(int num_pt, int max_value){
    vector<Dot>pts(num_pt);
    Dot *ptr = &pts[0];
    for (int i = 0; i <num_pt; i++) {
        Dot* point = new  Dot(rand() % (int) max_value, rand() % (int) max_value);
        ptr[i]= *point;
    }
    return pts;
}

vector<Cluster> create_cluster(int num_cl, int max_value){
    vector<Cluster>cls(num_cl);
     Cluster* ptr = &cls[0];
    for (int i = 0; i <num_cl; i++) {
        Cluster* cluster = new  Cluster(rand() % (int) max_value, rand() % (int) max_value);
        ptr[i] = *cluster;
    }
    return cls;
}

#endif // KNN_FUNCTIONS_H