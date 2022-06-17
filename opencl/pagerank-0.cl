__kernel void mELLPageRank(__global const int *rowptr ,
                            __global const int *col,
                            __global const float *data,
                            __global const float *pagerank,
                            __global float *pagerank_new,
                            int num_row,
                            int elem_in_row)
{
    int gid = get_global_id(0);

	if(gid < num_row)
	{
		float sum = 0.0;

		for (int j = 0; j < elem_in_row; j++)
		{
		    int idx = j * num_row + gid;
            sum += 0.85 * data[idx] * pagerank[col[idx]];
		}
		pagerank_new[gid] = (1-0.85) + sum;
	}
}



__kernel void mCSRPageRank(__global const int *rowptr,
                           __global const int *col,
                           __global const float *data,
                           __global const float *pagerank,
                           __global float *pagerank_new,
                           int rows)
{
    int gid = get_global_id(0);

    if(gid < rows)
	{
		float sum = 0.0f;
        for (int j = rowptr[gid]; j < rowptr[gid + 1]; j++)
            sum += 0.85 * data[j] * pagerank[col[j]];
		pagerank_new[gid] = (1-0.85) + sum;
	}
}

