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
#include <sstream>
using namespace std;


void plot(vector<Dot> &dots, string filename, int num_pts, int num_cls, int iterations, int num_threads){
    // add .txt to the end of the filename
    string general_filename = filename + ".txt";
    ofstream Myfile1(general_filename); // Use a different variable name
    Myfile1 << "x,y,cluster_id"<< endl ;
    for(int i = 0; i < dots.size(); i++){
        Dot point = dots[i];
        Myfile1 << point.get_x() << "," << point.get_y() << "," << point.get_cluster_id() << endl;
    }
    string specific_filename = "outputs/sanity_checks/" + filename + "_" + to_string(num_pts) + "_" + to_string(num_cls) + "_" + to_string(iterations) + "_" + to_string(num_threads) + ".txt";
    ofstream Myfile2(specific_filename); // Use a different variable name
    Myfile2 << "x,y,cluster_id"<< endl ;
    for(int i = 0; i < dots.size(); i++){
        Dot point = dots[i];
        Myfile2 << point.get_x() << "," << point.get_y() << "," << point.get_cluster_id() << endl;
    }
    Myfile1.close();
    Myfile2.close();
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

vector<Dot> read_dots_from_file(const string& filename) {
    vector<Dot> dots;
    ifstream infile(filename);
    string line;
    int threshold = 1;
    getline(infile, line); // Skip the header

    while (getline(infile, line)) {
        stringstream ss(line);
        double x, y;
        int cluster_id;
        char delim;

        ss >> x >> delim >> y >> delim >> cluster_id;
        int xx = (int)(x * threshold);
        int yy = (int)(y * threshold);
        Dot dot(xx, yy);
        dots.push_back(dot);
    }

    return dots;
}

#endif // KNN_FUNCTIONS_H