class Dot{
    private:
        double x_coordinate; 
        double y_coordinate;
        int cluster_id; 
    public:
        Dot(double x_coordinate, double y_coordinate){
            this->x_coordinate = x_coordinate;
            this->y_coordinate = y_coordinate;
            cluster_id = 0;
        }
        Dot(){ x_coordinate = 0; y_coordinate = 0; cluster_id = 0; }
        double get_x(){ return this->x_coordinate; }
        double get_y(){ return this->y_coordinate; }
        void set_id(int id){ this->cluster_id = id; }
        int get_cluster_id(){ return cluster_id; }
};