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
#define eps 0.01


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

  struct mtx_COO mCOO;
  struct mtx_CSR mCSR;
  struct mtx_ELL mELL;
  FILE * fp_converted;
  fp_converted = fopen("Google_modified1.txt", "r");
  mtx_COO_create_from_file( & mCOO, fp_converted);
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
  num_devices = 2; // limit to one device
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
  ////
  ////
  float * pagerank = (float * ) malloc(mCOO.num_cols * sizeof(float));
  float * pagerank_next = (float * ) malloc(mCOO.num_cols * sizeof(float));
  for (int i = 0; i < mCOO.num_cols; i++) {
    pagerank[i] = 1.0;
    pagerank_next[i] = 0.0;
  }
  ////
  float * pagerank_CSR = (float * ) malloc(mCOO.num_rows * sizeof(float));
  float * pagerank_next_CSR = (float * ) malloc(mCOO.num_rows * sizeof(float));
  for (int i = 0; i < mCOO.num_cols; i++) {
    pagerank_CSR[i] = 1.0;
    pagerank_next_CSR[i] = 0.0;
  }
  ////
  //// Buffers for ELL
  cl_mem mELLcol_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mELL.num_elements * sizeof(cl_int), NULL, & clStatus);
  cl_mem mELLdata_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mELL.num_elements * sizeof(cl_int), NULL, & clStatus);
  cl_mem pagerank_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mELL.num_rows * sizeof(cl_int), NULL, & clStatus);
  cl_mem pagerank_next_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
    mELL.num_rows * sizeof(cl_int), NULL, & clStatus);
  clStatus = clEnqueueWriteBuffer(command_queue, mELLcol_d, CL_TRUE, 0,
    mELL.num_elements * sizeof(cl_int), mELL.col, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, mELLdata_d, CL_TRUE, 0,
    mELL.num_elements * sizeof(cl_int), mELL.data, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, pagerank_d, CL_TRUE, 0,
    mELL.num_rows * sizeof(cl_int), pagerank, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, pagerank_next_d, CL_TRUE, 0,
    mELL.num_rows * sizeof(cl_int), pagerank_next, 0, NULL, NULL);

  cl_mem pagerank_d_CSR = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mCSR.num_rows * sizeof(cl_int), NULL, & clStatus);
  cl_mem pagerank_next_d_CSR = clCreateBuffer(context, CL_MEM_READ_WRITE,
    mCSR.num_rows * sizeof(cl_int), NULL, & clStatus);

  cl_mem mCSRrowptr_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    (mCSR.num_rows + 1) * sizeof(cl_int), NULL, & clStatus);
  cl_mem mCSRcol_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mCSR.num_nonzeros * sizeof(cl_int), NULL, & clStatus);
  cl_mem mCSRdata_d = clCreateBuffer(context, CL_MEM_READ_ONLY,
    mCSR.num_nonzeros * sizeof(cl_float), NULL, & clStatus);
  clStatus = clEnqueueWriteBuffer(command_queue, mCSRrowptr_d, CL_TRUE, 0,
    (mCSR.num_rows + 1) * sizeof(cl_int), mCSR.rowptr, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, mCSRcol_d, CL_TRUE, 0,
    mCSR.num_nonzeros * sizeof(cl_int), mCSR.col, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, mCSRdata_d, CL_TRUE, 0,
    mCSR.num_nonzeros * sizeof(cl_float), mCSR.data, 0, NULL, NULL);
  //
  // // create kernel ELL and set arguments
  cl_kernel kernelELL = clCreateKernel(program, "mELLPageRank", & clStatus);
  clStatus = clSetKernelArg(kernelELL, 0, sizeof(cl_mem), NULL);
  clStatus |= clSetKernelArg(kernelELL, 1, sizeof(cl_mem), (void * ) & mELLcol_d);
  clStatus |= clSetKernelArg(kernelELL, 2, sizeof(cl_mem), (void * ) & mELLdata_d);
  clStatus |= clSetKernelArg(kernelELL, 3, sizeof(cl_mem), (void * ) & pagerank_d);
  clStatus |= clSetKernelArg(kernelELL, 4, sizeof(cl_mem), (void * ) & pagerank_next_d);
  clStatus |= clSetKernelArg(kernelELL, 5, sizeof(cl_int), (void * ) & (mELL.num_rows));
  clStatus |= clSetKernelArg(kernelELL, 6, sizeof(cl_int), (void * ) & (mELL.num_elementsinrow));

  cl_kernel kernelCSR = clCreateKernel(program, "mCSRPageRank", & clStatus);
  clStatus = clSetKernelArg(kernelCSR, 0, sizeof(cl_mem), (void * ) & mCSRrowptr_d);
  clStatus |= clSetKernelArg(kernelCSR, 1, sizeof(cl_mem), (void * ) & mCSRcol_d);
  clStatus |= clSetKernelArg(kernelCSR, 2, sizeof(cl_mem), (void * ) & mCSRdata_d);
  clStatus |= clSetKernelArg(kernelCSR, 3, sizeof(cl_mem), (void * ) & pagerank_d_CSR);
  clStatus |= clSetKernelArg(kernelCSR, 4, sizeof(cl_mem), (void * ) & pagerank_next_d_CSR);
  clStatus |= clSetKernelArg(kernelCSR, 5, sizeof(cl_int), (void * ) & (mCSR.num_rows));
  //
  // Divide work CSR
  size_t local_item_size = WORKGROUP_SIZE;
  int num_groups = (mCSR.num_rows - 1) / local_item_size + 1;
  size_t global_item_size_CSR = num_groups * local_item_size;
  // Divide work ELL
  num_groups = ((mELL.num_rows - 1) / local_item_size + 1);
  size_t global_item_size_ELL = num_groups * local_item_size;
  double dtimeELL = omp_get_wtime();
  //
  ////ELL
  for (int i = 0; i < 52; i++) {
    if (i > 0) {
      clStatus = clEnqueueWriteBuffer(command_queue, pagerank_d, CL_TRUE, 0,
        mELL.num_rows * sizeof(cl_int), pagerank_next, 0, NULL, NULL);
    }
    clStatus = clEnqueueNDRangeKernel(command_queue, kernelELL, 1, NULL, &
      global_item_size_ELL, & local_item_size, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, pagerank_next_d, CL_TRUE, 0,
      mCSR.num_rows * sizeof(cl_float), pagerank_next, 0, NULL, NULL);
  }
  dtimeELL = omp_get_wtime() - dtimeELL;
  ////CSR
  double dtimeCSR = omp_get_wtime();
  for (int i = 0; i < 52; i++) {
    printf("?\n");
    if (i > 0) {
      clStatus = clEnqueueWriteBuffer(command_queue, pagerank_d_CSR, CL_TRUE, 0,
        N * sizeof(cl_int), pagerank_next_CSR, 0, NULL, NULL);
    }
    clStatus = clEnqueueNDRangeKernel(command_queue, kernelCSR, 1, NULL, &
      global_item_size_CSR, & local_item_size, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, pagerank_next_d_CSR, CL_TRUE, 0,
      N * sizeof(cl_float), pagerank_next_CSR, 0, NULL, NULL);
  }

  dtimeCSR = omp_get_wtime() - dtimeCSR;

  printf("Execution time CSR: %lf\n", dtimeCSR);
  printf("Execution time ELL: %lf\n", dtimeELL);
}