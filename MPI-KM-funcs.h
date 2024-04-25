typedef struct {
  double *dataset;
  unsigned int *members;
  int leading_dim;
  int secondary_dim; 
} data_struct;

void kmeans_distancer(data_struct *data_in, data_struct *clusters, double *newCentroids, double* SumOfDist);