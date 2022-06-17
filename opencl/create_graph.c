#include <stdio.h>

#include <stdlib.h>

#include <time.h>

#include <CL/cl.h>

#include <omp.h>

#include <math.h>

#include "mtx_sparse.h"

#include <errno.h>

#include <string.h>


#define WORKGROUP_SIZE (256)
#define MAX_SOURCE_SIZE (16384)

void count(int outbound_links[], int num_rows, int col[], int num_elementsinrow) {
  for (int x = 0; x < num_rows; x++) {
    for (int p = 0; p < num_elementsinrow; p++)
      if (col[p * num_rows + x] != -1)
        outbound_links[x]++;
  }
}

int main(int argc, char * argv[]) {

  FILE * fp_matrix;

  FILE * ptrr = fopen("Google_modified1.txt", "w");

  fp_matrix = fopen("Modified Google Graph.txt", "r");

  struct mtx_COO mCOO;
  struct mtx_CSR mCSR;
  struct mtx_ELL mELL;
  mtx_COO_create_from_file( & mCOO, fp_matrix);
  mtx_CSR_create_from_mtx_COO( & mCSR, & mCOO);
  mtx_ELL_create_from_mtx_CSR( & mELL, & mCSR);
  int N = mCOO.num_rows;
  int * outbound_links = (int * ) calloc(N, sizeof(int));
  count(outbound_links, mELL.num_rows, mELL.col, mELL.num_elementsinrow);

  FILE * fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen("Google_modified1.txt", "r");
  if (fp == NULL)
    exit(EXIT_FAILURE);
  int count = 0;
  while ((read = getline( & line, & len, fp)) != -1) {
    if (count == 0) {
      fprintf(ptrr, line);
    } else {
      int frst;
      int sec;
      int thrd;
      sscanf(line, "%d %d %d", & frst, & sec, & thrd);
      fprintf(ptrr, "%d %d %f\n", sec, frst, (float) 1 / outbound_links[frst - 1]);
    }
    count++;
  }
}