// adaptive integration 
// compile: gcc -O2 quad.c -fopenmp -lm -o quad

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mtx_sparse.h"
#include "mtx_sparse.c"
#include "sort.c"

#define TOL 1e-8
#define THREADS 32

#define eps 0.01

int distance (float *p, float *p1, int n)
{
    float sum = 0.0;
    float distance;
    for (int i = 0; i<n; i++)
    sum += pow(p[i]-p1[i],2);
    distance = sqrt(sum);
    if (distance > eps)
    return 1;
    else
    return 0;
}

// sorts arrays col and row based on values in col
/*void sort (int col[], int row[], int len){
	int t;
	int t2;
	for (int i = 0; i < len; i++) {     
        for (int j = i+1; j < len; j++) {     
           if(col[i] > col[j]) {    
               t = col[i];
               t2 = row[i];    
               col[i] = col[j];   
               row[i] = row[j];  
               col[j] = t;
               row[j] = t2;    
           }     
        }     
    }    
}*/

//from [0 0 0 1 1 2 2 2 2 3 5 5] to [0 3 5 9 10 10]
void compress (int col[], int colCOO[], int N, int num_nonzeros){
	int COOidx = 0;
	int COOStart = 0;
	for(int i = 0; i < N; i++){
		col[i] = COOidx;
		for(;COOidx < num_nonzeros && colCOO[COOidx] == i; COOidx++);
	}
}

void count(int outbound_links[], int row[], int num_nonzeros){
	#pragma omp parallel for
	for(int i = 0; i < num_nonzeros; i++){
		#pragma omp atomic
		outbound_links[row[i]]++;
	}
}

int main(int argc, char* argv[])
{
    #pragma omp parallel
	#pragma omp master
    printf ("Number of threads: %d\n", omp_get_num_threads ());
    FILE *f;
	struct mtx_COO mCOO;

	float total_time = omp_get_wtime();
	
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		return(1);
	}
	else    
	{ 
		if ((f = fopen(argv[1], "r")) == NULL) 
			return(1);
	}
	
	printf("Reading matrix ... \n");
	float readingTime = omp_get_wtime();
	// create sparse matrices
	if (mtx_COO_create_from_file(&mCOO, f) != 0)
		return(1);

	printf("Finished creating matrix from file - time taken: %.3lf \n", omp_get_wtime() - readingTime);

	int N = mCOO.num_rows;
	printf("Sorting col ...\n");
	float time = omp_get_wtime();
	quicksort(mCOO.col, mCOO.row, 0, mCOO.num_nonzeros);
	printf("Sorted in time: %.3lf\n", omp_get_wtime() - time);

	printf("Compressing col ... \n");
	time = omp_get_wtime();
	int *col = (int *)malloc((N+1) * sizeof(int));
	col[N] = mCOO.num_nonzeros;
	compress(col, mCOO.col, N, mCOO.num_nonzeros);
	printf("Compressed in: %.3lf\n", omp_get_wtime() - time);

	printf("Counting outbund links per page ... \n");
	time = omp_get_wtime();
	int *outbound_links = (int *)calloc(N, sizeof(int));
	int *row = mCOO.row;
	count(outbound_links, row, mCOO.num_nonzeros);
	printf("Links counted in: %.3lf\n", omp_get_wtime() - time);
	

	float *PageRank = (float *)malloc(N * sizeof(float));
    float *PageRank_New = (float *)malloc(N * sizeof(float));
	float *PageRank_New_pom = (float *)malloc(N * sizeof(float));
	float d = 0.85;

	float dt = omp_get_wtime();
	

	#pragma omp parallel for
	for(int i = 0; i < N; i++){
		PageRank[i] = 1.0;	
		PageRank_New[i] = 0.0;
	}
	

	printf("Starting with pagerank calculations \n");

	printf("N: %d num_nonzeros: %d num_rows: %d\n", N, mCOO.num_nonzeros, mCOO.num_rows);
    for(int iteration = 0; iteration < 45; iteration++){

		#pragma omp parallel for
		for(int i = 0; i < N; i++){
			float sum = 0.0;
			for(int j = col[i]; j < col[i+1]; j++){
				int idx_of_inbound_page = row[j];
				sum += PageRank[idx_of_inbound_page] / outbound_links[idx_of_inbound_page];
			}

			PageRank_New[i] = (1-d) + (d * sum);
		}

		float* pr_pointer = PageRank;
		PageRank = PageRank_New;
		PageRank_New = pr_pointer;
	}
    
	/*for(int i = 0; i < 10;i++){
		printf("Pagerank for page %d: %f\n", i, PageRank[i]);
	}*/
    

    dt = omp_get_wtime() - dt;
    printf("\nTime for calculating pagerank: %.3lf\n\n", dt);
	printf("Total time: %.3lf\n\n", omp_get_wtime() - total_time);

    return 0;
}
