#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "MPI-KM-funcs.h"
#include "MPI-cluster.h"
#include <sys/time.h>
#include <math.h>
#include <mpi.h>


int number_of_dots = 5000; // go to 5000
int  number_of_clusters = 100;
int max_iterations = 100;
int number_of_processes = 2;

void clean(data_struct* data1);
void print(data_struct* data2print);
void initialize_clusters(data_struct *data_in,data_struct *cluster_in);
void random_initialization(data_struct *data_in);
void mpi_plot(data_struct* data, int numAttributes, int numObjects, const char* filename);


int main(int argc, char **argv){
	int i, processors, rank; 
	struct timeval first, second, lapsed;
	struct timezone tzp;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processors);

	int numObjects = number_of_dots;
	int numAttributes = 2;
	int numClusters = number_of_clusters;

	data_struct data_in;
	data_struct clusters;

	data_in.leading_dim = numAttributes;
	data_in.secondary_dim = numObjects;
	data_in.dataset = (double*)malloc(numObjects * numAttributes * sizeof(double));
	data_in.members = (unsigned int*)malloc(numObjects * sizeof(unsigned int));

	clusters.leading_dim = numAttributes;
	clusters.secondary_dim = numClusters;
	clusters.dataset = (double*)malloc(numClusters * numAttributes * sizeof(double));
	clusters.members = (unsigned int*)malloc(numClusters * sizeof(unsigned int)); 

    FILE *logFile = NULL;
    char filename[256];

    if (rank == 0) {
        sprintf(filename, "outputs/method_comparison/mpi_%d_%d_%d_%d.txt", number_of_dots, number_of_clusters, max_iterations, number_of_processes);
        logFile = fopen(filename, "w");
        if (!logFile) {
            fprintf(stderr, "Failed to open log file for writing.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

	double start_time, init_end_time;
	if (rank == 0) {
		printf("Number of Points %d\n", numObjects);
		printf("Number of Clusters %d\n", numClusters);
		printf("Number of Processes %d\n", processors);
		printf("Initialization \n");
	}
	random_initialization(&data_in);
	if (rank == 0){
		start_time = MPI_Wtime();
		printf("Creation of the Points and Clusters\n");
		initialize_clusters(&data_in, &clusters);
		init_end_time = MPI_Wtime();
		printf("Points and Clusters Created \n");
	    printf("Initialization made in: %f seconds\n", init_end_time - start_time);
		fprintf(logFile, "Initialization time: %f seconds\n", init_end_time - start_time);
 	 }
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(clusters.dataset, numClusters*numAttributes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	data_struct p_data;
	p_data.leading_dim = numAttributes; 
	double n_split =  numObjects / processors;
	double p_objects = ceil(n_split);
	int n_temp = p_objects *  processors;
	if (rank != 0){
		p_data.secondary_dim = p_objects;
		p_data.dataset = (double*)malloc(p_objects * numAttributes * sizeof(double));
		p_data.members = (unsigned int*)malloc(p_objects * sizeof(unsigned int)); 
	}
	else{
		p_data.secondary_dim = p_objects + (numObjects - n_temp);
		p_data.dataset = (double*)malloc(p_data.secondary_dim * numAttributes * sizeof(double));
		p_data.members = (unsigned int*)malloc(p_data.secondary_dim * sizeof(unsigned int)); 	
	}
	for (i = 0; i < p_data.secondary_dim * p_data.leading_dim;i++){ 
		p_data.dataset[i] = data_in.dataset[rank * p_data.secondary_dim * p_data.leading_dim+i]; 
	}
	if (rank == 0){
		gettimeofday(&first, &tzp);
	}
	int iter, j, k;
	double SumOfDist = 0, new_SumOfDist=0, temp_SumOfDist=0;
	double* newCentroids;
	int* temp_clusterSize;
	unsigned int*temp_dataMembers;

	temp_clusterSize = (int*)malloc(numClusters * sizeof(int));
	temp_dataMembers = (unsigned int*)malloc(numObjects * sizeof(unsigned int));
	newCentroids = (double*)malloc(numAttributes * numClusters * sizeof(double));
	for (i = 0; i < numClusters; i++){
		temp_clusterSize[i] = 0;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double iter_start_time, iter_end_time;
	if (rank == 0) {
		printf("-STARTING ITERATE-\n");
	}
	for (iter = 0; iter < max_iterations; iter++){
		if (rank == 0) {
			iter_start_time = MPI_Wtime();
		}
		new_SumOfDist = 0;
		temp_SumOfDist = 0;

		for (i = 0; i < numClusters * numAttributes; i++){
			newCentroids[i] = 0;
		}
		kmeans_process(&p_data, &clusters, newCentroids, &new_SumOfDist);
		MPI_Allreduce(newCentroids, clusters.dataset, numClusters*numAttributes, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(clusters.members, temp_clusterSize, numClusters, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);  
		for (i = 0; i < numClusters; i++){
       		clusters.members[i] = temp_clusterSize[i];
		}
		for (i = 0; i < numClusters; i++){
			for (j = 0; j < numAttributes; j++){
				clusters.dataset[i * numAttributes + j] /= (double) clusters.members[i];
			}
		}
		MPI_Allreduce(&new_SumOfDist, &temp_SumOfDist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		SumOfDist = temp_SumOfDist;
		if (rank == 0) {
			iter_end_time = MPI_Wtime();
			printf("Iteration %d \n", iter + 1);
			printf("Clusters Update made in: %f seconds\n", iter_end_time - iter_start_time);
			fprintf(logFile, "Iteration %d: %f seconds\n", iter + 1, iter_end_time - iter_start_time);
		}
	}
	free(newCentroids);
	free(temp_clusterSize);
	MPI_Barrier(MPI_COMM_WORLD);
	for (i = 0; i < p_data.secondary_dim; i++){ 
		temp_dataMembers[rank * p_data.secondary_dim + i] = p_data.members[i]; 
	} 
	MPI_Allreduce(temp_dataMembers, data_in.members, numObjects, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
	free(temp_dataMembers);
	MPI_Barrier(MPI_COMM_WORLD);

	double total_time;
	if (rank == 0) {
		total_time = MPI_Wtime() - start_time;
		printf("Number of iterations %d, total time %f seconds, iteration time avg %f seconds\n", 
			iter, total_time, total_time / iter);
		printf("Storing the points coordinates and cluster-id...\n");
        fprintf(logFile, "Total time: %f seconds\n", total_time);
        fprintf(logFile, "Average iteration time: %f seconds\n", total_time / max_iterations);
        fclose(logFile);
	}
	mpi_plot(&data_in, numAttributes, numObjects, "MPI_clusters.txt");
	clean(&p_data);	
	clean(&data_in);
	clean(&clusters);	
	MPI_Finalize();
}

void mpi_plot(data_struct* data, int numAttributes, int numObjects, const char* filename) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double* all_data = NULL;
    if (rank == 0) {
        all_data = (double*)malloc(numObjects * numAttributes * sizeof(double));
    }
    MPI_Gather(data->dataset, data->secondary_dim * numAttributes, MPI_DOUBLE,
               all_data, data->secondary_dim * numAttributes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* file = fopen(filename, "w");
        if (file == NULL) {
            printf("Error opening file\n");
            return;
        }
        fprintf(file, "x,y,cluster_id\n");
        for (int i = 0; i < numObjects; i++) {
            fprintf(file, "%f,%f,%d\n", all_data[i * numAttributes], all_data[i * numAttributes + 1], data->members[i]);
        }
        fclose(file);
        free(all_data);
    }
	
}

void random_initialization(data_struct *data_in){
	int i, j = 0;
	int n = data_in->leading_dim;
	int m = data_in->secondary_dim;
	double *tmp_dataset = data_in->dataset;
	unsigned int *tmp_Index = data_in->members;

	srand(0); 
	for (i = 0; i < m; i++){
		tmp_Index[i] = 0;
		for (j = 0; j < n; j++){
			tmp_dataset[i * n + j] = (int)(((double) rand() / RAND_MAX) * 1000000); 
    	}
  	}
}


void initialize_clusters(data_struct *data_in,data_struct *cluster_in){
	int i, j, pick = 0;
	int n = cluster_in->leading_dim;
	int m = cluster_in->secondary_dim;
	int Objects = data_in->secondary_dim;
	double *tmp_Centroids = cluster_in->dataset;
	double *tmp_dataset = data_in->dataset;
	unsigned int *tmp_Sizes = data_in->members;
	srand(0);
	int step = Objects / m;
	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++){
      		tmp_Centroids[i * n + j] = tmp_dataset[pick * n + j];
    	}
		pick += step; 
	}	
}

void print(data_struct* data2print){
	int i, j = 0;
	int n = data2print->leading_dim;
	int m = data2print->secondary_dim;
	double *tmp_dataset = data2print->dataset;
  
	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++){
      		printf("%f ", tmp_dataset[i * n + j]);
    	}
    	printf("\n");
  	}
}

void clean(data_struct* data1){
  free(data1->dataset);
  free(data1->members);
}


// RUN WITH THE FOLLOWING
// mpicc -c MPI-KM.c -o MPI-KM.o
// THEN RUN THE COMMAND FOR MPI-KM-funcs.c
// mpicc MPI-KM-funcs.o MPI-KM.o -o MPI-KM 
// mpirun -np 10 ./MPI-KM