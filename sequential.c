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

int bsearc(int *test, int N, int searched)
{
    int lo = 0;
    int hi = N;
    int out = -1;
while (hi > lo) 
{
    int mid = lo + (hi-lo)/2;
    int val = test[mid];
    if (val < searched)
    {
        lo = mid +1;
    }
    else if (val > searched)
    hi = mid;
    else {
        out = mid;
        hi = mid;
    }
}
return out;
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
    for (int i = 0; i < mCOO.num_cols; i++)
        {
        PageRank[i] = 1.0;
        PageRank_New[i] = 0.0; 
        }
    double dtimeCOO = omp_get_wtime();
    int c = 0;
    int temp = 0; 
    int temp2 = 0; 
    int fast = 0;
    for (int i = 0; i < mCOO.num_nonzeros; i++)
        Test[i] = mCOO.row[i];
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
           if(Test[i] > Test[j])
           {
               fast = Test[i];
               Test[i] = Test[j];
               Test[j] = fast;
           }
        }     
    }    
    
//     // while(distance(PageRank, PageRank_New,N))
    while(c<45) //if you want in terms of number of iterations
    
    {
        if (c!=0)
            for (int i = 0; i < mCOO.num_cols; i++)
                PageRank[i]=PageRank_New[i];
        for (int k = 0; k < mCOO.num_rows; k++) 
        {
            float sum = 0.0;           
            
                    int count;
                    int t = bsearc(mCOO.col, (int)mCOO.num_nonzeros, k); 
                    if(t<0)
                    goto a; 
                    for (int z = t; z < mCOO.num_nonzeros; z++) 
                    {
                        if (mCOO.col[z] > k)
                           {break;}                           
                            int x = bsearc(Test,mCOO.num_nonzeros, mCOO.row[z] );
                            
                           count = 0;      
                            for(int g = x; g < mCOO.num_nonzeros; g++)
                                {
                                if(Test[g] == mCOO.row[z])
                                count++;
                                if(Test[g] > mCOO.row[z])
                                    break;
                                }
                     
                        if (count != 0)                   
                            sum += PageRank[mCOO.row[z]]/count;
                    }                

            
            a: PageRank_New[k] = (1-d) + d*sum;
        }
        c++;
    }   
    dtimeCOO = omp_get_wtime() - dtimeCOO;     
    for (int i = 0; i < mCOO.num_cols; i++)
        printf("Page rank for page %d: %f\n",i,PageRank[i]);
    printf("Execution Time: %fms\n", 1000*dtimeCOO);
// // --------------------------------------------------------------
    for (int i = 0; i < N; i++)
        {
        PageRank[i] = 1.0;
        PageRank_New[i] = 0.0; 
        }
    dtimeCOO = omp_get_wtime();
    c = 0;
    // while(distance(PageRank, PageRank_New,N))
    while(c<45) //if you want in terms of number of iterations
        {
            if (c!=0)
                for (int i = 0; i < mCOO.num_cols; i++)
                    PageRank[i]=PageRank_New[i];
            for (int k = 0; k < N; k++) 
            { 
            
            float sum = 0.0;
                for( int g = 0; g < mCSR.num_nonzeros; g++) 
                {
                    if(mCSR.col[g] == k)   
                    {
                        int count = 0;
                        int min = mCSR.col[g];
                        for (int y = g - 1 ; y >= 0; y--)
                        {
                            if( min > mCSR.col[y])
                            min =  mCSR.col[y];
                            else
                            {count++;
                            min =  mCSR.col[y];}
                        }
                    int output = mCSR.rowptr[count+1]-mCSR.rowptr[count];
                        
                        if (output != 0)                   
                        sum += PageRank[count]/output;                        
                    }   
                }
                PageRank_New[k] = (1-d) + d*sum;
        }
        c++;
        }
        dtimeCOO = omp_get_wtime() - dtimeCOO;
        for (int i = 0; i < mCOO.num_cols; i++)
            printf("Page rank for page %d: %f\n",i,PageRank[i]);
        printf("Execution Time: %fms\n", 1000*dtimeCOO);

// // --------------------------------------------------------

for (int i = 0; i < N; i++)
    {
    PageRank[i] = 1.0;
    PageRank_New[i] = 0.0; 
    }
    dtimeCOO = omp_get_wtime();
    c = 0;
    // while(distance(PageRank, PageRank_New,N))
while(c<45)
    {
    if (c!=0)
        for (int i = 0; i < mCOO.num_cols; i++)
            PageRank[i]=PageRank_New[i];
    for (int k = 0; k < N; k++) 
    { 
        float sum = 0.0;
            for( int g = 0; g < mELL.num_elementsinrow; g++) 
            {
                for( int x = 0; x < mELL.num_rows; x++)
                    {
                        int count = 0;
                        if(mELL.col[g * mELL.num_rows + x] == k)   
                        {
                            for (int p = 0; p < mELL.num_elementsinrow; p++) 
                                if(mELL.col[p * mELL.num_rows + x] != -1)
                                    count ++; 
                        }
                        if (count != 0)                   
                            sum += PageRank[x]/count;
                    }
                
            }
            PageRank_New[k] = (1-d) + d*sum;
    }
    c++;
    }
    dtimeCOO = omp_get_wtime() - dtimeCOO;
    
    for (int i = 0; i < mCOO.num_cols; i++)
        printf("Page rank for page %d: %f\n",i,PageRank[i]);
    printf("Execution Time: %fms\n", 1000*dtimeCOO);

    mtx_COO_free(&mCOO);
    mtx_CSR_free(&mCSR);
    mtx_ELL_free(&mELL);

	return 0;
}
