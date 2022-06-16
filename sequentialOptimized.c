#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include "mtx_sparse.h"
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
    struct mtx_CSR mCSR;
    struct mtx_ELL mELL;
    float d = 0.85;
    

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
    mtx_CSR_create_from_mtx_COO(&mCSR, &mCOO);
    mtx_ELL_create_from_mtx_CSR(&mELL, &mCSR);
    
    int N = mCOO.num_rows;
    // allocate vectors
    float *PageRank = (float *)malloc(mCOO.num_rows * sizeof(float));
    float *PageRank_New = (float *)malloc(mCOO.num_rows * sizeof(float));
    int *Test = (int *)malloc(mCOO.num_nonzeros * sizeof(int));
    double dtimeCOO = omp_get_wtime();
    for (int i = 0; i < mCOO.num_cols; i++)
        {
        PageRank[i] = 1.0;
        PageRank_New[i] = 0.0; 
        }
    
    int c = 0;
    int temp = 0; 
    int temp2 = 0; 
    int fast = 0;
    for (int i = 0; i < mCOO.num_nonzeros; i++) {     
        for (int j = i+1; j < mCOO.num_nonzeros; j++) {     
           if(mCOO.col[i] > mCOO.col[j]) {    
               temp = mCOO.col[i];
               temp2 = mCOO.row[i];    
               mCOO.col[i] = mCOO.col[j];   
               mCOO.row[i] = mCOO.row[j];  
               mCOO.col[j] = temp;
               mCOO.row[j] = temp2;    
           }     
        }     
    }    


    int *col = (int *)malloc((N+1) * sizeof(int));
    col[N] = mCOO.num_nonzeros;
    compress(col, mCOO.col, N, mCOO.num_nonzeros);
    
    int *outbound_links = (int *)calloc(N, sizeof(int));
	int *row = mCOO.row;
    count(outbound_links, row, mCOO.num_nonzeros);




//     // while(distance(PageRank, PageRank_New,N))
    while(c<45) //if you want in terms of number of iterations
    
    {
        if (c!=0)
            for (int i = 0; i < mCOO.num_cols; i++)
                PageRank[i]=PageRank_New[i];
        for (int k = 0; k < mCOO.num_rows; k++) 
        {
            float sum = 0.0;                                
            for(int j = col[k]; j < col[k+1]; j++)
                {
                    int idx_of_inbound_page = row[j];
                    sum += PageRank[idx_of_inbound_page] / outbound_links[idx_of_inbound_page];
                }                

            
             PageRank_New[k] = (1-d) + d*sum;
        } 
        c++;
    }   
    dtimeCOO = omp_get_wtime() - dtimeCOO;     
     for (int i = 0; i < mCOO.num_cols; i++)
         printf("Page rank for page %d: %f\n",i,PageRank[i]);
    printf("Execution Time: %fs\n", dtimeCOO);
// // // --------------------------------------------------------------
    dtimeCOO = omp_get_wtime();
    for (int i = 0; i < N; i++)
        {
        PageRank[i] = 1.0;
        PageRank_New[i] = 0.0; 
        }
    int *outbound_links2 = (int *)calloc(mCSR.num_rows, sizeof(int));
    int *row3 = (int *)calloc(mCSR.num_nonzeros, sizeof(int));
	int *row2 = mCSR.rowptr;
    count2(outbound_links2, row2,row3, mCOO.num_rows+1);
    for (int i = 0; i < mCOO.num_nonzeros; i++) {     
        for (int j = i+1; j < mCOO.num_nonzeros; j++) {     
           if(mCSR.col[i] > mCSR.col[j]) {    
               temp = mCSR.col[i];
               temp2 = row3[i];    
               mCSR.col[i] = mCSR.col[j];   
               row3[i] = row3[j];  
               mCSR.col[j] = temp;
               row3[j] = temp2;    
           }     
        }     
    }  
    int *col2 = (int *)malloc((N+1) * sizeof(int));
    col2[N] = mCSR.num_nonzeros;
    compress(col2, mCSR.col, N, mCSR.num_nonzeros);  


    c = 0;
    // while(distance(PageRank, PageRank_New,N))
    while(c<45) //if you want in terms of number of iterations
    
    {
        if (c!=0)
            for (int i = 0; i < mCOO.num_cols; i++)
                PageRank[i]=PageRank_New[i];
        for (int k = 0; k < mCOO.num_rows; k++) 
        {
            float sum = 0.0;                                
            for(int j = col2[k]; j < col2[k+1]; j++)
                {
                    int idx_of_inbound_page = row3[j];
                    sum += PageRank[idx_of_inbound_page] / outbound_links2[idx_of_inbound_page];
                }                

            
             PageRank_New[k] = (1-d) + d*sum;
        } 
        c++;
    }   
        dtimeCOO = omp_get_wtime() - dtimeCOO;
        for (int i = 0; i < mCOO.num_cols; i++)
          printf("Page rank for page %d: %f\n",i,PageRank[i]);
        printf("Execution Time: %fs\n",dtimeCOO);


    mtx_COO_free(&mCOO);
    mtx_CSR_free(&mCSR);
    mtx_ELL_free(&mELL);

	return 0;
}
