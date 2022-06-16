#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include "mtx_sparse.h"
#include <omp.h>

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


void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}
int partition(int arr[], int arr2[], int start, int end)
{
    // Declaration
    int pivot = arr[end];
    int i = (start - 1);
 
    // Rearranging the array
    for (int j = start; j <= end - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
            swap(&arr2[i], &arr2[j]);
        }
    }
    swap(&arr[i + 1], &arr[end]);
    swap(&arr2[i + 1], &arr2[end]);
 
    // Returning the respective index
    return (i + 1);
}

void quicksort(int arr[], int arr2[], int start, int end)
{
    // Declaration
    int index;
 
    if (start < end) {
 
        // Getting the index of pivot
        // by partitioning
        index = partition(arr, arr2, start, end);
 
// Parallel sections
#pragma omp parallel sections
        {
#pragma omp section
            {
                // Evaluating the left half
                quicksort(arr, arr2, start, index - 1);
            }
#pragma omp section
            {
                // Evaluating the right half
                quicksort(arr, arr2, index + 1, end);
            }
        }
    }
}
 
void compress (int col[], int colCOO[], int N, int num_nonzeros){
	int COOidx = 0;
	int COOStart = 0;
	for(int i = 0; i < N; i++){
		col[i] = COOidx;
		for(;COOidx < num_nonzeros && colCOO[COOidx] == i; COOidx++);
	}
}

void count(int outbound_links[], int row[], int num_nonzeros){
	for(int i = 0; i < num_nonzeros; i++){
		outbound_links[row[i]]++;
	}
}

void count2(int outbound_links[], int row[], int row3[], int num_nonzeros){
	for(int i = 0; i < num_nonzeros; i++){
		{
        outbound_links[i] = row[i+1] - row[i];
        if(i != num_nonzeros -1)
        for(int z=row[i]; z < row[i+1]; z++)
                 row3[z] = i;
    }
}
}




int main(int argc, char *argv[])
{

	FILE *f;
    
	struct mtx_COO mCOO;
    float totalTimeCOO = 0;
    float totalTimeCSR = 0;
    int myid, procs;
	struct mtx_CSR mCSR;
	
    MPI_Init(&argc, &argv);
	MPI_Status status;
  	MPI_Request send_request, recv_request;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // process ID
	MPI_Comm_size(MPI_COMM_WORLD, &procs); // number of processes
    if(myid == 0)
    printf("Working with %d number of processes\n", procs);
	double dtimeCOO = MPI_Wtime(); 
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
	else    
	{ 
		if ((f = fopen(argv[1], "r")) == NULL) 
			exit(1);
	}
	
	// create sparse matrices
	if (mtx_COO_create_from_file(&mCOO, f) != 0)
		exit(1);
    dtimeCOO = MPI_Wtime() - dtimeCOO; 
    if(myid == 0)
    {
        // totalTimeCOO += dtimeCOO;
    printf("Time taken for creating COO representation from file: %fs\n", dtimeCOO);} 

    dtimeCOO = MPI_Wtime();  
	mtx_CSR_create_from_mtx_COO(&mCSR, &mCOO);
    dtimeCOO = MPI_Wtime() - dtimeCOO; 
    if(myid == 0)
    {
        // totalTimeCSR += dtimeCOO;
    printf("Time taken for creating CSR representation from COO: %fs\n", dtimeCOO);} 
	
     dtimeCOO = MPI_Wtime();
	int N = mCOO.num_rows;
    float *PageRank = (float *)malloc(N * sizeof(float));
    float *PageRank_New = (float *)malloc(N * sizeof(float));
	float *PageRank_New_pom = (float *)malloc(N * sizeof(float));
	float d = 0.85;
    int *send_counts = (int *)malloc(procs * sizeof(int));	// how much work does every process get
	int *displacments = (int *)malloc(procs * sizeof(int)); // offsets of work in global table for every process
    for (long i=0; i < procs; i++)
    {
		
        displacments[i] = i*N / procs; 
        send_counts[i] = N * (i + 1) / procs - N * i / procs;
    }
    dtimeCOO = MPI_Wtime() - dtimeCOO; 
    if(myid == 0)
    {totalTimeCOO += dtimeCOO;
    totalTimeCSR += dtimeCOO;
    printf("Allocating, initializing, dividing work taken time: %fs\n", dtimeCOO);  }
    dtimeCOO = MPI_Wtime();
    if (myid == 0)
	for (int i = 0; i < N; i++)
		{
			PageRank[i] = 1.0;
			PageRank_New[i] = 0.0; 
			PageRank_New_pom[i]=0.0;
		}
	MPI_Bcast(PageRank, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(PageRank_New, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    dtimeCOO = MPI_Wtime() - dtimeCOO;  
    if(myid == 0)
    {totalTimeCOO += dtimeCOO;
    printf("PageRank Initial values distribute time for COO: %fs\n", dtimeCOO);  }     
    
	 
    dtimeCOO = MPI_Wtime();
    int c = 0; 
    quicksort(mCOO.col,mCOO.row,0, mCOO.num_nonzeros);
    dtimeCOO = MPI_Wtime() - dtimeCOO; 
    if(myid == 0)
    {totalTimeCOO += dtimeCOO;
    printf("COO Sorting time: %fs\n", dtimeCOO);  }
	
    dtimeCOO = MPI_Wtime(); 
    int *col = (int *)malloc((N+1) * sizeof(int));
    col[N] = mCOO.num_nonzeros;
    compress(col, mCOO.col, N, mCOO.num_nonzeros);
    if(myid == 0)
    {dtimeCOO = MPI_Wtime() - dtimeCOO; 
    totalTimeCOO += dtimeCOO;
    printf("COO Compressing time: %fs\n", dtimeCOO); }
    dtimeCOO = MPI_Wtime(); 
    int *outbound_links = (int *)calloc(N, sizeof(int));
	int *row = mCOO.row;
    count(outbound_links, row, mCOO.num_nonzeros);
    if(myid == 0)
    {dtimeCOO = MPI_Wtime() - dtimeCOO; 
    totalTimeCOO += dtimeCOO;
    printf("COO Counting outbound links time: %fs\n", dtimeCOO);}

// 	// while(c == 0)
    dtimeCOO = MPI_Wtime(); 
	 while(c<45) //if you want in terms of number of iterations
   {
            for (int i = 0; i < mCOO.num_cols; i++)
                PageRank_New[i] = 0;
           	for (int k = displacments[myid]; k < displacments[myid] + send_counts[myid]; k++) 
        {
            float sum = 0.0;           
            for(int j = col[k]; j < col[k+1]; j++)
                {
                    int idx_of_inbound_page = row[j];
                    sum += PageRank[idx_of_inbound_page] / outbound_links[idx_of_inbound_page];
                }                

            
             PageRank_New[k] = (1-d) + d*sum;
        } 
		
		MPI_Allreduce(PageRank_New, PageRank_New_pom, N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		
		// if (distance(PageRank, PageRank_New_pom,N) != 1)
			c++;
			for (int i = 0; i < mCOO.num_rows; i++)	
				{
					PageRank[i] = PageRank_New_pom[i];	
				}		
	// c++;
    }   
	  

if (myid == 0)
 {
    dtimeCOO = MPI_Wtime() - dtimeCOO; 

totalTimeCOO += dtimeCOO;
printf("COO PageRank Execution Time: %fs\n", dtimeCOO);
printf("Total COO Time: %fs\n", totalTimeCOO);
//  for (int i = 0; i < mCOO.num_cols; i++)	
	// printf("COO PageRank for page: %d , with value %f\n",i, PageRank[i]);


 }   

// // -------------------------------------------------------------------------------------------------

    dtimeCOO = MPI_Wtime();
    if (myid == 0)
	for (int i = 0; i < N; i++)
		{
			PageRank[i] = 1.0;
			PageRank_New[i] = 0.0; 
			PageRank_New_pom[i]=0.0;
		}
	MPI_Bcast(PageRank, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(PageRank_New, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    dtimeCOO = MPI_Wtime() - dtimeCOO;  
    if(myid == 0)
    {totalTimeCSR += dtimeCOO;
    printf("PageRank Initial values distribute time: %fs\n", dtimeCOO);  }  

    dtimeCOO = MPI_Wtime(); 
	int *outbound_links2 = (int *)calloc(mCSR.num_rows, sizeof(int));
    int *row3 = (int *)calloc(mCSR.num_nonzeros, sizeof(int));
	int *row2 = mCSR.rowptr;
    count2(outbound_links2, row2,row3, mCOO.num_rows+1);
    if(myid == 0)
    {dtimeCOO = MPI_Wtime() - dtimeCOO; 
    totalTimeCSR += dtimeCOO;
    printf("CSR Counting outbound links time: %fs\n", dtimeCOO);}

    dtimeCOO = MPI_Wtime();
    quicksort(mCSR.col,row3,0, mCSR.num_nonzeros);
    dtimeCOO = MPI_Wtime() - dtimeCOO; 
    if(myid == 0)
    {totalTimeCSR += dtimeCOO;
    printf("CSR Sorting time: %fs\n", dtimeCOO);  }
    dtimeCOO = MPI_Wtime();
    int *col2 = (int *)malloc((N+1) * sizeof(int));
    col2[N] = mCOO.num_nonzeros;
    compress(col2, mCSR.col, N, mCSR.num_nonzeros);
    if(myid == 0)
    {dtimeCOO = MPI_Wtime() - dtimeCOO; 
    totalTimeCSR += dtimeCOO;
    printf("CSR Compressing time: %fs\n", dtimeCOO); }
    
    c = 0;

	dtimeCOO = MPI_Wtime();
    // while(c == 0)
    while(c<45) //if you want in terms of number of iterations
        {
            
            for (int i = 0; i < mCOO.num_cols; i++)
                PageRank_New[i] = 0;
            for (int k = displacments[myid]; k < displacments[myid] + send_counts[myid]; k++) 
            { 
            
            float sum = 0.0;
                    for(int j = col2[k]; j < col2[k+1]; j++)
                {
                    int idx_of_inbound_page = row3[j];
                    sum += PageRank[idx_of_inbound_page] / outbound_links2[idx_of_inbound_page];
                }                

            
             PageRank_New[k] = (1-d) + d*sum;
             } 
                
        
		MPI_Allreduce(PageRank_New, PageRank_New_pom, N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		
		// if (distance(PageRank, PageRank_New_pom,N) != 1)
			c++;
			for (int i = 0; i < mCOO.num_rows; i++)	
				{
					PageRank[i] = PageRank_New_pom[i];	
				}		
	// c++;
        }
        
	if (myid == 0)
	{
        dtimeCOO = MPI_Wtime() - dtimeCOO;
        totalTimeCSR += dtimeCOO;
	// for (int i = 0; i < mCOO.num_cols; i++)	
		// printf("CSR PageRank for page: %d , with value %f\n",i, PageRank[i]);
	printf("CSR PageRank Execution Time: %fs\n", dtimeCOO);
    printf("Total CSR Time: %fs\n", totalTimeCSR);
    
	}   

 	
    mtx_COO_free(&mCOO);
    mtx_CSR_free(&mCSR);

	MPI_Finalize();
	return 0;
}











	






    
    