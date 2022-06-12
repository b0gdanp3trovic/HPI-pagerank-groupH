__kernel void mELLPageRank(__global const int *rowptr ,
                            __global const int *col,
                            __global const float *pagerank,
                            __global float *pagerank_new,
                            int num_row,
                            int elem_in_row)
{
    int gid = get_global_id(0);

    if(gid < num_row)
    {
        float sum = 0.0;
        for(int i = 0; i < elem_in_row; i++)
        {
            for(int j = 0; j < num_row; j++)
            {
                int count = 0;
                if(col[i * num_row + j] == gid)
                {
                    for(int p = 0; p < elem_in_row; p++)
                    {
                        if(col[p*num_row + j] != -1)
                            count++;
                    }
                }
                if(count != 0)
                {
                    sum += pagerank[j]/count;
                }
            }
        }
        pagerank_new[gid]=(1-0.85)+0.85*sum;
    }
}