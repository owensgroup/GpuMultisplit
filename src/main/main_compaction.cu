/*
GpuMultisplit is the proprietary property of The Regents of the University of California ("The Regents") and is copyright © 2016 The Regents of the University of California, Davis campus. All Rights Reserved. 

Redistribution and use in source and binary forms, with or without modification, are permitted by nonprofit educational or research institutions for noncommercial use only, provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
* The name or other trademarks of The Regents may not be used to endorse or promote products derived from this software without specific prior written permission.

The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.

THE SOFTWARE PROVIDED IS ON AN "AS IS" BASIS, AND THE REGENTS HAVE NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. THE REGENTS SPECIFICALLY DISCLAIM ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES, INCLUDING BUT NOT LIMITED TO  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA OR PROFITS, OR BUSINESS INTERRUPTION, HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

If you do not agree to these terms, do not download or use the software.  This license may be modified only in a writing signed by authorized signatory of both parties.

For license information please contact copyright@ucdavis.edu re T11-005.
*/

#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <cstring>

#include "cpu_functions.h"
#include "kernels/compaction/multisplit2_compaction.cuh"

template<uint32_t flag, typename key_type>
struct test_bucket : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return (a & ((1<<flag) - 1));
  }
};

int main()
{
	//=========
	int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if(devCount){
    cudaSetDevice(DEVICE_ID__); // be changed later
    cudaGetDeviceProperties(&devProp, DEVICE_ID__);
  }
	printf("=====================================\n");
  printf("Device: %s\n", devProp.name);
	printf("=====================================\n");
	printf("Compaction test \n");
	printf("=====================================\n");

	const uint32_t n_start = (1<<25);
	const uint32_t n_end = (1<<25);
	const uint32_t n_buckets = 2;
	const uint32_t kIter = 1;
	bool 	validate = true;

	uint32_t *h_key_in = new uint32_t[n_end];
	uint32_t *h_cpu_results_key = new uint32_t[n_end];
	uint32_t *h_value_in = new uint32_t[n_end];
	uint32_t *h_cpu_results_value = new uint32_t[n_end];
	uint32_t *h_gpu_results_key = new uint32_t[n_end];
	uint32_t *h_gpu_results_value = new uint32_t[n_end];
	bool total_correctness = true;

	uint32_t* d_key_in;
	uint32_t* d_key_out;
	uint32_t* d_value_in;
	uint32_t* d_value_out;
	cudaMalloc((void**)&d_key_in, sizeof(uint32_t) * n_end);
	cudaMalloc((void**)&d_key_out, sizeof(uint32_t) * n_end);
	cudaMalloc((void**)&d_value_in, sizeof(uint32_t) * n_end);
	cudaMalloc((void**)&d_value_out, sizeof(uint32_t) * n_end);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	test_bucket<1,uint32_t> myBucket_2;

	for(uint32_t n_elements = n_start; n_elements <= n_end; n_elements <<= 1)
	{
		// ==== key-only scenario:
		float avg_time = 0.0f;
		compaction_context ms_context;
		compaction_allocate_key_only(n_elements, ms_context);

		for(int kk = 0; kk < kIter; kk++)
		{
			float temp_time = 0.0f;
			// generate random keys:
			randomPermute(h_key_in, n_elements);
			cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
			cudaMemset(d_key_out, 0, sizeof(uint32_t) * n_elements);

			cudaEventRecord(start, 0);
			uint32_t n_bucket_zero = compaction_key_only(d_key_in, d_key_out, n_elements, ms_context, myBucket_2);

			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&temp_time, start, stop);
			avg_time += temp_time;

			if(validate)
			{
				cpu_multisplit_general(h_key_in, h_cpu_results_key, n_elements, myBucket_2, 0, n_buckets);
				cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
				bool correct = true;
				for(int i = 0; i<n_elements && correct;i++)
				{
					if(h_cpu_results_key[i] != h_gpu_results_key[i]){
						printf(" ### Iteration %d: Wrong results at index %d: cpu = %d, gpu = %d\n", kk, i, h_cpu_results_key[i], h_gpu_results_key[i]);
						correct = false;
					}
				}
				total_correctness &= correct;
			}
			// if(validate)
			// {
			// 	cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
			// 	cpu_multisplit_general(h_key_in, h_key_out, n_elements, myBucket_2, 0, n_buckets);
			// 	for(int i = 0; i<n_elements; i++)
			// 	{
			// 		if(h_gpu_results_key[i] != h_key_out[i])
			// 		{
			// 			printf("Multisplit(k) with %d elements and %d bucket at iteration %d was incorrect\n", n_elements, n_buckets, kk);
			// 			break;
			// 		}
			// 	}
			// }
		}
		avg_time /= kIter;

		printf("%10u elements and %3u buckets:\n", n_elements, n_buckets);
		printf("\t (key_only ): avg time = %.3f ms \t rate = %.3f Mkeys/s\n", avg_time, float(n_elements)/avg_time/1000.0f);

		compaction_release_memory(ms_context);

		if(validate)
		{
			if(total_correctness) printf("Validation was done successfully for %d elements!\n", n_elements);
			else printf("Validation failed for %d elements!\n", n_elements);			
		}
		total_correctness = true;

		// ==== key-value case:
		float avg_time_kv = 0.0f;
		compaction_context ms_context_kv;
		compaction_allocate_key_value(n_elements, ms_context_kv);

		for(int kk = 0; kk < kIter; kk++)
		{
			float temp_time = 0.0f;
			// generate random keys:
			randomPermute(h_key_in, n_elements);
			std::memcpy(h_value_in, h_key_in, sizeof(uint32_t) * n_elements);
			cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
			cudaMemcpy(d_value_in, h_value_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
			cudaMemset(d_key_out, 0, sizeof(uint32_t) * n_elements);

			cudaEventRecord(start, 0);
			uint32_t n_bucket_zero = compaction_key_value(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, ms_context_kv, myBucket_2);

			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&temp_time, start, stop);
			avg_time_kv += temp_time;

			if(validate)
			{
				cpu_multisplit_pairs_general(h_key_in, h_cpu_results_key, h_value_in, h_cpu_results_value, n_elements, myBucket_2, 0, n_buckets);
				cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
				cudaMemcpy(h_gpu_results_value, d_value_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
				bool correct = true;
				for(int i = 0; i<n_elements && correct;i++)
				{
					if(h_cpu_results_key[i] != h_gpu_results_key[i] || h_cpu_results_value[i] != h_gpu_results_value[i]){
						printf(" ### Wrong results at index %d: cpu = (%d, %d), gpu = (%d,%d)\n", i, h_cpu_results_key[i], h_cpu_results_value[i], h_gpu_results_key[i], h_gpu_results_value[i]);
						correct = false;
					}
				}
				total_correctness &= correct;
			}

			// if(validate)
			// {
			// 	cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
			// 	cudaMemcpy(h_gpu_results_value, d_value_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
			// 	cpu_multisplit_pairs_general(h_key_in, h_key_out, h_value_in, h_value_out, n_elements, myBucket_2, 0, n_buckets);
			// 	for(int i = 0; i<n_elements; i++)
			// 	{
			// 		if(h_gpu_results_key[i] != h_key_out[i] || h_gpu_results_value[i] != h_value_out[i])
			// 		{
			// 			printf("Multisplit(kv) with %d elements and %d bucket at iteration %d was incorrect\n", n_elements, n_buckets, kk);
			// 			break;
			// 		}
			// 	}
			// }
		}
		avg_time_kv /= kIter;
		printf("\t (key_value): avg time = %.3f ms \t rate = %.3f Mkeys/s\n", avg_time_kv, float(n_elements)/avg_time_kv/1000.0f);

		compaction_release_memory(ms_context_kv);

		if(validate)
		{
			if(total_correctness) printf("Validation was done successfully for %d elements!\n", n_elements);
			else printf("Validation failed for %d elements!\n", n_elements);			
		}
		total_correctness = true;		
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	delete[] h_key_in;
	delete[] h_value_in;
	delete[] h_cpu_results_key;
	delete[] h_cpu_results_value;
	delete[] h_gpu_results_key;
	delete[] h_gpu_results_value;

	cudaFree(d_key_in);
	cudaFree(d_key_out);
	cudaFree(d_value_in);
	cudaFree(d_value_out);
}