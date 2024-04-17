#include "Dot.h"
#include <omp.h>
#include <queue>

class Cluster{
    private:
        double x_coordinate;
        double y_coordinate;
        int number_of_points; 
        double x_coordinate_total;
        double y_coordinate_total;
    public:
        Cluster(double x_coordinate,double y_coordinate){
            number_of_points = 0;
            this->x_coordinate = x_coordinate;
            this->y_coordinate = y_coordinate;
            x_coordinate_total = 0;
            y_coordinate_total = 0;
        }
        Cluster(){
            number_of_points = 0;
            this->x_coordinate = 0;
            this->y_coordinate = 0;
            x_coordinate_total = 0;
            y_coordinate_total = 0;
        }
        double get_x(){ return this->x_coordinate; }
        double get_y(){ return this->y_coordinate; }
        void add_point(Point pt) {
            #pragma omp atomic update
                    number_of_points++;
            #pragma atomic update
                    x_coordinate_total += pt.get_x();
            #pragma omp atomic update
                    y_coordinate_total += pt.get_y();
        }
        void delete_values(){
            this->number_of_points = 0;
            this->x_coordinate_total = 0;
            this->y_coordinate_total = 0;
        }
        bool update_values(){
            if(this->x_coordinate == x_coordinate_total / this->number_of_points && this->y_coordinate == y_coordinate_total / this->number_of_points){
                return false;
            }
            this->x_coordinate = x_coordinate_total/this->number_of_points;
            this->y_coordinate = y_coordinate_total/this->number_of_points;
            return true;
        }
};