#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <omp.h>
#include <math.h>
#include "mtx_sparse.h"

#define WORKGROUP_SIZE(256)
#define MAX_SOURCE_SIZE(16384)
#define eps 0.01

int distance(float * p, float * p1, int n) {
  float sum = 0.0;
  float distance;
  for (int i = 0; i < n; i++)
    sum += pow(p[i] - p1[i], 2);
  distance = sqrt(sum);
  if (distance > eps)
    return 1;
  else
    return 0;
}

int main(int argc, char * argv[]) {
  // Read kernel from file
  FILE * fp;
  char fileName[100];
  char * source_str;
  size_t source_size;
  cl_int clStatus;

  FILE * fp_matrix;

  float d = 0.85;
  int t = 0;

  sprintf(fileName, "%s.cl", argv[1]);
  fp = fopen(fileName, "r");

  if (!fp) {
    fprintf(stderr, ":-(#\n");
    exit(1);
  }
  source_str = (char * ) malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  source_str[source_size] = '\0';
  fclose(fp);

  fp_matrix = fopen("50K.txt", "r");

  struct mtx_COO mCOO;
  struct mtx_CSR mCSR;
  struct mtx_ELL mELL;

  mtx_COO_create_from_file( & mCOO, fp_matrix);
  mtx_CSR_create_from_mtx_COO( & mCSR, & mCOO);
  mtx_ELL_create_from_mtx_CSR( & mELL, & mCSR);
  int N = mCOO.num_rows;

  // Get platforms
  cl_uint num_platforms;
  clStatus = clGetPlatformIDs(0, NULL, & num_platforms);
  cl_platform_id * platforms = (cl_platform_id * ) malloc(sizeof(cl_platform_id) * num_platforms);
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
  //Get platform devices
  cl_uint num_devices;
  clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, & num_devices);
  num_devices = 1; // limit to one device
  cl_device_id * devices = (cl_device_id * ) malloc(sizeof(cl_device_id) * num_devices);
  clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
  // Context
  cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, & clStatus);
  // Command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, & clStatus);
  // Create and build a program
  cl_program program = clCreateProgramWithSource(context, 1, (const char ** ) & source_str, NULL, & clStatus);
  clStatus = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

  // Log
  size_t build_log_len;
  char * build_log;
  clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, & build_log_len);
  if (build_log_len > 2) {
    build_log = (char * ) malloc(sizeof(char) * (build_log_len + 1));
    clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
      build_log_len, build_log, NULL);
    printf("%s", build_log);
    free(build_log);
    return 1;
  }

  float * pagerank = (float * ) malloc(mCOO.num_rows * sizeof(float));
  float * pagerank_next = (float * ) malloc(mCOO.num_rows * sizeof(float));
  for (int i = 0; i < mCOO.num_cols; i++) {
    pagerank[i] = 1.0;
    pagerank_next[i] = 0.0;
  }

  cl_mem mELLcol_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mELL.num_elements * sizeof(cl_int), NULL, & clStatus);
  cl_mem pagerank_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mELL.num_elements * sizeof(cl_int), NULL, & clStatus);
  cl_mem pagerank_next_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
    mELL.num_elements * sizeof(cl_int), NULL, & clStatus);
  clStatus = clEnqueueWriteBuffer(command_queue, mELLcol_d, CL_TRUE, 0,
    mELL.num_elements * sizeof(cl_int), mELL.col, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, pagerank_d, CL_TRUE, 0,
    mELL.num_rows * sizeof(cl_int), pagerank, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, pagerank_next_d, CL_TRUE, 0,
    mELL.num_rows * sizeof(cl_int), pagerank_next, 0, NULL, NULL);

  // create kernel ELL and set arguments
  cl_kernel kernelELL = clCreateKernel(program, "mELLPageRank", & clStatus);
  clStatus = clSetKernelArg(kernelELL, 0, sizeof(cl_mem), NULL);
  clStatus |= clSetKernelArg(kernelELL, 1, sizeof(cl_mem), (void * ) & mELLcol_d);
  clStatus |= clSetKernelArg(kernelELL, 2, sizeof(cl_mem), (void * ) & pagerank_d);
  clStatus |= clSetKernelArg(kernelELL, 3, sizeof(cl_mem), (void * ) & pagerank_next_d);
  clStatus |= clSetKernelArg(kernelELL, 4, sizeof(cl_int), (void * ) & (mELL.num_rows));
  clStatus |= clSetKernelArg(kernelELL, 5, sizeof(cl_int), (void * ) & (mELL.num_elementsinrow));

  size_t local_item_size = WORKGROUP_SIZE;
  // Divide work ELL
  int num_groups = ((mELL.num_rows - 1) / local_item_size + 1);
  size_t global_item_size_ELL = num_groups * local_item_size;
  double dtimeELL_cl = omp_get_wtime();
  for (int i = 0; i < 3; i++) {
    clStatus = clEnqueueNDRangeKernel(command_queue, kernelELL, 1, NULL, &
      global_item_size_ELL, & local_item_size, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, pagerank_next_d, CL_TRUE, 0,
      mCSR.num_rows * sizeof(cl_float), pagerank_next, 0, NULL, NULL);
  }

  dtimeELL_cl = omp_get_wtime() - dtimeELL_cl;
  printf("Execution time: %lf\n", dtimeELL_cl);
  //for (int i = 0; i < N; i++)
  //{   if(pagerank_next[i] > 0)
  //    {
  //        printf("Page rank for page %d: %f\n", i, pagerank_next[i]);
  //    }
  //}
}