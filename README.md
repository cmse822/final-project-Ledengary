# CMSE 822 Final Project: K-Means Clustering

# 1. Introduction

In the modern era of computational science, efficient processing of large datasets is crucial. This project focuses on the efficiency of parallel computing paradigms in handling intensive computational tasks, specifically K-means clustering. K-means is a fundamental algorithm widely used in data mining and machine learning to group a set of objects based on attributes into K distinct groups (clusters).

K-means clustering involves partitioning n observations into k clusters in which each observation belongs to the cluster with the nearest mean. This mean serves as a prototype of the cluster. Given its computationally intensive nature, especially with large datasets or high dimensionality, parallel computing can play a pivotal role. Parallel computing uses multiple processing elements simultaneously to solve a problem, which can significantly improve the computation speed and efficiency.

This project investigates the application of two parallel computing paradigms: Message Passing Interface (MPI) and Open Multi-Processing (OpenMP). Their effectiveness is evaluated based on performance metrics such as computation time and efficiency in implementing the K-means clustering algorithm.

## 1.1 Background

![K-Means Clustering Algorithm in Machine Learning](outputs/figures/k-means-clustering-algorithm-in-machine-learning.png)

The K-means algorithm operates through an iterative refinement technique involving:

1. **Initialization**: Defining K initial "means" (centroids).
2. **Assignment**: Assigning each data point to the nearest centroid.
3. **Update**: Recalculating centroids as the mean of all data points assigned to that centroid.
4. **Iteration**: Repeating assignment and update steps until the centroids stabilize.

Parallelizing K-means is challenging due to its iterative nature and dependencies between data points and centroids. Various strategies can be employed to distribute the workload across different processors.

## 1.2 Computational Methods

This project utilizes MPI and OpenMP for parallelization:
- **MPI** is used to distribute data points across different nodes in a computing cluster, allowing parallel processing in a distributed system.
- **OpenMP** provides thread-based parallelism suitable for shared-memory architectures, enabling multiple threads to operate on shared data structures and minimize data movement.

## 1.3 Project Statement

The project posits that applying parallel processing paradigms can significantly enhance the performance of the K-means clustering algorithm in terms of computational speed and scalability, especially as dataset sizes increase.

## 1.4 Writeup Structure

The structure of this writeup is organized as follows:
- **2. Methods**: Describes the parallel implementation of the K-means clustering algorithm using MPI and OpenMP, including the environment setup, data distribution strategies, and synchronization mechanisms employed.
- **3. Results**: Presents the performance results of the implementations, analyzed through various metrics such as runtime, scalability, and efficiency.
- **4. Conclusions**: Discusses the findings in the context of parallel computing effectiveness and suggests directions for future research.

# 2. Methods

This section describes the methods used to implement and evaluate the K-means clustering algorithm under different computational strategies, starting with the sequential approach.

## 2.1 Sequential Implementation

### 2.1.1 Overview

The sequential version of the K-means algorithm serves as the baseline for comparison with parallel implementations. It was executed on a single-threaded environment to establish a control performance metric. This method helps in understanding the performance gains achieved through parallel processing techniques.

### 2.1.2 Implementation Details

**Environment Setup:**
- **Compiler**: The code was compiled using `clang++` with support for OpenMP to enable easy toggling of multithreading for experimental purposes.
- **Libraries**: Standard C++ libraries along with `<chrono>` for timing, `<cmath>` for mathematical functions, and `<fstream>` for file operations were used.

**Main Components:**
- **Data Initialization**: The dataset, initially read from `"MPI_clusters.txt"`, represents the starting points for clustering, ensuring consistency across different implementations.
- **Cluster Initialization**: Randomly generates initial centroids within the range defined by `max_value`.
- **Distance Calculation**: Computes Euclidean distance between points and centroids to determine the nearest cluster.
- **Cluster Update**: Adjusts centroids based on the mean of points assigned to each cluster.

**Execution Flow:**
1. **Start Timing**: Marks the beginning of the initialization phase.
2. **Read Input**: Points are read from a pre-generated file to ensure consistency.
3. **Initialize Clusters**: Random centroids are generated.
4. **End Initialization Timing**: Concludes the timing of the initialization phase.
5. **Iterative Optimization**:
   - For each iteration, distances are calculated, and clusters are updated.
   - The time for each iteration is logged for later analysis.
6. **Conclude**: Finalizes the clustering process and records total execution time and average iteration time.

**Logging**:
- Outputs are logged into a file named `"sequential_clusters.txt"`, capturing initialization time, per-iteration time, total time, and average time per iteration. 

### 2.1.3 Compilation and Execution

Compiled and executed with the following commands:
```bash
clang++ -fopenmp -o sequential_km Sequential-KM.cpp -L/opt/homebrew/opt/llvm/lib -I/opt/homebrew/opt/llvm/include -Wl,-rpath,/opt/homebrew/opt/llvm/lib
./sequential_km
```

## 2.2 Parallel Implementation 

### 2.2.1 Overview

The OpenMP implementation of the K-means algorithm aims to leverage multi-threading capabilities to reduce the computational time required for clustering large datasets. This parallel approach focuses on distributing the computation of distances and cluster assignments across multiple threads.

### 2.2.2 Implementation Details

**Environment Setup:**
- Utilizes the same compiler and libraries as the sequential implementation to maintain consistency.

**Enhancements in OpenMP:**
- **Parallel Loops**: Key loops in the `find_distance()` and `update_clusters()` functions are parallelized.
- **Data Sharing**: Points and clusters are shared among threads, whereas minimum distances and indices are kept private to prevent data races.
- **Synchronization**: Critical sections are used to safely update cluster centroids when multiple threads attempt to modify the same data.

**Execution Flow:**
1. Initialization remains identical to the sequential version to ensure that any performance differences are due solely to the computation phases.
2. **Parallel Distance Calculation**:
   - Each thread computes distances for a subset of points to all centroids.
   - Threads independently determine the closest centroids for their assigned points.
3. **Concurrent Cluster Updates**:
   - Once points are assigned, threads collaboratively update centroid positions using atomic operations to avoid inconsistencies.

### 2.2.3 Compilation and Execution

The parallel version can be compiled and run using the same commands as the sequential version but requires an environment supporting OpenMP.

### 2.2.4 Performance Metrics

Performance is measured in terms of:
- **Initialization Time**: Time taken to set up clusters and read data.
- **Iteration Time**: Average time per iteration during the clustering process.
- **Total Execution Time**: Time from start to finish of the clustering algorithm.

Performance results are documented for various thread counts to analyze scalability and efficiency gains from parallel processing.

### 2.2.5 Key Functions in Parallel Implementation

#### Find Distance Function

The `find_distance` function is responsible for calculating the Euclidean distance between each point and all centroids, assigning each point to the nearest cluster. This function is highly parallelizable as each point's calculation is independent of others.

**Parallelization Strategy**:
- **Outer Loop Parallelization**: The loop over points is parallelized, allowing each thread to handle a subset of points.
- **Private Variables**: Minimum distance and index variables are private to each thread to prevent read-write conflicts.
- **Shared Data**: Points and clusters are shared across all threads since they are read concurrently without modification.

**OpenMP Pragma**:
```cpp
#pragma omp parallel for private(min_dist, min_index) shared(pts, cls) schedule(static, 1000)
for (int i = 0; i < pts_size; ++i) {
    Point &current_point = pts[i];
    min_dist = euclidean_dist(current_point, cls[0]);
    min_index = 0;
    for (int j = 0; j < cls_size; ++j) {
        double dist = euclidean_dist(current_point, cls[j]);
        if (dist < min_dist) {
            min_dist = dist;
            min_index = j;
        }
    }
    pts[i].set_id(min_index);
}
```

This code snippet demonstrates the use of OpenMP to distribute the workload of computing distances across multiple threads.

#### Update Clusters Function

The `update_clusters` function adjusts the centroids based on the newly assigned points. This function involves modifying shared data, which requires careful synchronization.

**Parallelization Strategy**:
- **Reduction of Cluster Properties**: To efficiently compute the new centroid positions, properties like the total coordinates are combined at the end of each iteration using a reduction clause.
- **Critical Section**: Updating centroid positions is placed within a critical section to ensure that updates from different threads do not interfere with each other.

**OpenMP Pragma**:
```cpp
#pragma omp for schedule(static)
for (int i = 0; i < cls_size; ++i) {
    if (cls[i].update_values()) {
        #pragma omp critical
        {
            cls[i].compute_new_centroid();
        }
    }
}
```

This section ensures that while the update of centroids requires synchronization, the check for movement (`update_values`) does not, optimizing concurrency while maintaining data integrity. These enhancements are crucial for achieving significant performance improvements in the parallel version of the K-means algorithm by effectively utilizing multi-threading capabilities provided by OpenMP.

**Logging**:
- Outputs are logged into a file named `"parallel_clusters.txt"`, capturing initialization time, per-iteration time, total time, and average time per iteration. 

### 2.2.6 Compilation and Execution
Compiled and executed with the following commands:
```bash
lang++ -fopenmp -o parallel_km Parallel-KM.cpp -L/opt/homebrew/opt/llvm/lib -I/opt/homebrew/opt/llvm/include -Wl,-rpath,/opt/homebrew/opt/llvm/lib
./parallel_km 
```

## 2.3 MPI Implementation

### 2.3.1 Overview

The MPI implementation of the K-means algorithm is designed to leverage the distributed computing capabilities across multiple nodes in a cluster. This approach aims to handle even larger datasets by distributing the workload effectively across multiple processing units.

### 2.3.2 Implementation Details

**Environment Setup:**
- **Compiler**: The MPI version was developed using the C programming language due to compatibility and performance considerations on the available computing cluster.
- **Libraries**: Uses MPI library for handling data distribution and aggregation across different nodes.

**Main Components:**
- **Data Distribution**: Initial data points and centroids are distributed across different nodes to ensure parallel processing without interference.
- **Concurrent Computation**: Each node computes distances and assigns points to clusters independently.
- **Reduction Operations**: MPI reduce operations are used to aggregate data and update global centroids accurately.

**Execution Flow:**
1. **Initialization**: Similar to the sequential and OpenMP implementations, with additional steps to distribute data across nodes.
2. **Parallel Distance Calculation and Assignment**:
   - Each process calculates distances for a subset of points to all centroids.
   - Points are then assigned to the nearest centroid locally.
3. **Global Update**:
   - Global centroid updates are performed using MPI's reduction functions to ensure all nodes have updated and consistent centroid values.

**Communication Patterns**:
- Uses `MPI_Bcast` for broadcasting initial centroids.
- Employs `MPI_Allreduce` for gathering and reducing centroid updates from all nodes.


### 2.3.3 Key Functions in MPI Implementation
**K-means Distancer Function**
This function, executed on each node, calculates the distances from points to centroids, assigns points to the nearest cluster, and partially updates centroid coordinates locally.
```c
void kmeans_distancer(data_struct *data_in, data_struct *clusters, double *newCentroids, double* SumOfDist)
{
	int i, j, k;
	double tmp_dist = 0;
	int tmp_index = 0;
	double min_dist = 0;
	double *dataset = data_in->dataset;
	double *centroids = clusters->dataset;
	unsigned int *Index = data_in->members;
	unsigned int *cluster_size = clusters->members;

	for (i = 0; i < clusters->secondary_dim; i++){
		cluster_size[i] = 0;
	}
	for (i = 0; i < data_in->secondary_dim; i++){
		tmp_dist = 0;
		tmp_index = 0;
		min_dist = FLT_MAX;
		for (k = 0; k < clusters->secondary_dim; k++){
			tmp_dist = euclidean_distance(dataset + i * data_in->leading_dim, centroids + k * clusters->leading_dim, data_in->leading_dim);
			if (tmp_dist<min_dist){
				min_dist = tmp_dist;
				tmp_index = k;
			}
		}
		Index[i] = tmp_index;
		SumOfDist[0] += min_dist;
		cluster_size[tmp_index]++;
		for (j = 0; j < data_in->leading_dim; j++){
			newCentroids[tmp_index * clusters->leading_dim + j] += dataset[i * data_in->leading_dim + j]; 
		}
	}
}
```

**Parallel and Distributed Strategies:**
- Each node operates on its local data independently.
- Uses collective communication to ensure all nodes synchronize their results effectively.

### 2.3.4 Logging and Output
Similar to other implementations, logs include detailed timing for initialization, each iteration, and total computation time are stored in `"MPI_clusters.txt"`. 

### 2.3.6 Compilation and Execution

Compiled and executed with the following commands:
```bash
mpicc -c MPI-KM.c -o MPI-KM.o
mpicc -c MPI-KM-funcs.c -o MPI-KM-funcs.o
mpicc MPI-KM-funcs.o MPI-KM.o -o MPI-KM
mpirun -np <NUMBER_OF_PROCESSORS> ./MPI-KM
```

### 3. Results
#### 3.1 Performance Analysis

The performance of the K-means clustering algorithm was measured across different computational approaches and dataset sizes. This analysis presents the results of the total computation time and average iteration time for sequential, MPI, and OpenMP implementations.

#### 3.2 Experimental Setup

- Dataset Sizes: 5,000 to 5,000,000 points
- Clusters: 100
- Iterations: 100
- Processors/Threads: 1, 2, 4, 6, 8, 10

#### 3.3 Total Time and Average Iteration Time

The following figures compare the total computation time and the average time per iteration across different numbers of processes/threads for MPI and OpenMP, with the sequential approach as a baseline.

##### 3.3.1 Comparison for 5,000 Points

For a small dataset of 5,000 points, the performance gains with increasing processors/threads are evident. MPI and OpenMP show significant improvements over sequential execution, with MPI often outperforming OpenMP as the number of processors increases.

##### 3.3.2 Comparison for 50,000 Points

As the dataset size increases to 50,000 points, the benefit of parallel computing becomes more pronounced. Both MPI and OpenMP maintain a lower computation time than the sequential approach, with MPI showing the best performance at higher processor counts.

##### 3.3.3 Comparison for 500,000 Points

At 500,000 points, the efficiency of MPI in a distributed environment is highlighted, managing to maintain scalability and outperforming OpenMP in total computation time. However, OpenMP shows competitive performance, especially in the average time per iteration.

##### 3.3.4 Comparison for 5,000,000 Points

For the largest tested dataset of 5,000,000 points, MPI's distributed computing capabilities shine, achieving the best scalability and efficiency. OpenMP's performance also scales well, but the advantages of MPI's distributed memory model are apparent in handling large-scale data.

#### 3.4 Discussion

##### 3.4.1 OpenMP Trends

In the OpenMP implementation, increasing the number of threads up to a certain point results in a decrease in both total and average iteration times, demonstrating the expected performance improvement from parallelization. However, this decline in time is not consistent; at higher thread counts, the performance plateaus and, in some cases, even slightly increases. This could be due to a variety of factors including:

- Overhead of managing additional threads: As the number of threads increases, the overhead associated with creating, managing, and synchronizing them may outweigh the benefits of parallel execution.
- Memory bandwidth saturation: On shared-memory architectures, all threads compete for a finite amount of memory bandwidth. As more threads are added, the system may hit a bottleneck where memory cannot be supplied to threads quickly enough, leading to sub-optimal utilization of CPU resources.
- Amdahl's Law: The principle that the speedup of a program from parallelization is limited by the time needed for the sequential fraction of the task, which means there is an upper limit to the benefit gained from adding more parallel execution units.

##### 3.4.2 MPI Trends

The MPI implementation, conversely, displays a more consistent decline in computation times as the number of processes increases. This consistent performance improvement can be attributed to the distributed nature of the MPI paradigm where each process operates independently on a separate data subset, reducing memory contention and communication overhead between threads. The distributed memory model of MPI allows for more effective scaling as it is not constrained by the shared memory bandwidth.

##### 3.4.3 Comparison Between MPI and OpenMP

MPI consistently outperforms OpenMP across all scenarios for several reasons:

- Distributed vs. Shared Memory: MPI's distributed memory model allows it to better handle larger datasets as it is less likely to be bottlenecked by shared resources.
- Scalability: MPI scales more efficiently across multiple nodes in a cluster, leveraging increased computational resources more effectively than OpenMP, which is confined to a single node.
- Network Communication: MPI is designed for network communication between distributed processes and is optimized for passing messages efficiently, which is crucial for the reduce and broadcast operations in the K-means algorithm.

##### 3.4.4 Sequential Performance

Sequential execution remains consistently slower, approximately 5 times, than both MPI and OpenMP implementations. This substantial performance gap is a clear indicator of the limitations inherent in single-threaded processing for computationally intensive tasks. The inability to parallelize the workload in the sequential approach means it cannot utilize the additional computational resources that significantly boost the performance of MPI and OpenMP.

In conclusion, while both parallel paradigms significantly improve performance over sequential processing, MPI exhibits superior scalability and consistency in decreasing computation times. This makes MPI particularly well-suited for high-performance computing environments where large datasets are processed across multiple nodes. The choice between MPI and OpenMP will thus depend on the specific requirements and constraints of the computing environment and the size of the dataset being processed.
