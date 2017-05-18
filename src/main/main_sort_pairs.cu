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
#include <functional>
#include <time.h>
#define CUB_STDERR
#include <cub/cub.cuh>
#include "cuda_profiler_api.h"

#include "cpu_functions.h"
#include "gpu_functions.cuh"
#include "api/multisplit_sort.cuh"

//=========================================================================
// Defined parameters:
//=========================================================================
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
//=========================================================================
int main(int argc, char** argv)
{
	//=========
	int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if(devCount){
    cudaSetDevice(DEVICE_ID__); // be changed later
    cudaGetDeviceProperties(&devProp, DEVICE_ID__);
  }
  printf("Device: %s\n", devProp.name);

  // ===============================
  srand(time(NULL));
  
  // number of input elements:
	uint32_t n_elements = (1<<25);

	const uint32_t kIter = 10;

	float temp_time = 0.0f;
	float cub_sort_time = 0.0f;
	float ms_sort_time = 0.0f;
	float ms_sort_5bit_time = 0.0f;
	float ms_sort_6bit_time = 0.0f;
	float ms_sort_7bit_time = 0.0f;
	float ms_sort_8bit_time = 0.0f;
	// ===============================
	// allocating memory:
	// ===============================
	uint32_t 	*h_key_in = new uint32_t[n_elements];
	uint32_t	*h_value_in = new uint32_t[n_elements];

	uint32_t* d_key_in;
	uint32_t* d_value_in;
	uint32_t* d_key_out_cub;
	uint32_t* d_value_out_cub;

	// for multisplit
	uint32_t* d_key_out_ms;
	uint32_t* d_value_out_ms;

	cudaMalloc((void**)&d_key_in, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_value_in, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_key_out_cub, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_value_out_cub, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_key_out_ms, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_value_out_ms, sizeof(uint32_t) * n_elements);

//=== for validation
	uint32_t* d_num_diffs_key;
	uint32_t* d_num_diffs_values;
	cudaMalloc((void**)&d_num_diffs_key, sizeof(uint32_t) * kIter);
	cudaMalloc((void**)&d_num_diffs_values, sizeof(uint32_t) * kIter);
	cudaMemset(d_num_diffs_key, 0, sizeof(uint32_t) * kIter);
	cudaMemset(d_num_diffs_values, 0, sizeof(uint32_t) * kIter);
//===================
	cudaEvent_t start_cub, stop_cub;
	cudaEventCreate(&start_cub);
	cudaEventCreate(&stop_cub);

	cudaEvent_t start_ms, stop_ms;
	cudaEventCreate(&start_ms);
	cudaEventCreate(&stop_ms);

	cudaEvent_t start_ms_5bit, stop_ms_5bit;
	cudaEventCreate(&start_ms_5bit);
	cudaEventCreate(&stop_ms_5bit);

	cudaEvent_t start_ms_6bit, stop_ms_6bit;
	cudaEventCreate(&start_ms_6bit);
	cudaEventCreate(&stop_ms_6bit);

	cudaEvent_t start_ms_7bit, stop_ms_7bit;
	cudaEventCreate(&start_ms_7bit);
	cudaEventCreate(&stop_ms_7bit);

	cudaEvent_t start_ms_8bit, stop_ms_8bit;
	cudaEventCreate(&start_ms_8bit);
	cudaEventCreate(&stop_ms_8bit);

	//======================
	// key-value sort:
	void 		*d_temp_storage_sort = NULL;
	size_t 	temp_storage_bytes_sort = 0;

	CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out_cub, d_value_in, d_value_out_cub, n_elements));	
	CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_sort, temp_storage_bytes_sort));

	// allocating memory for multisplit-sort
	ms_sort_context sort_context_8bit;
	multisplit_sort_8bit_allocate_pairs(n_elements, sort_context_8bit);

	ms_sort_context sort_context_7bit;
	multisplit_sort_7bit_allocate_pairs(n_elements, sort_context_7bit);

	ms_sort_context sort_context_6bit;
	multisplit_sort_6bit_allocate_pairs(n_elements, sort_context_6bit);

	ms_sort_context sort_context_5bit;
	multisplit_sort_5bit_allocate_pairs(n_elements, sort_context_5bit);

	ms_sort_context sort_context_4bit;
	multisplit_sort_4bit_allocate_pairs(n_elements, sort_context_4bit);

	for(int kk = 0; kk<kIter; kk++)
	{
		// generating key-values:
		for(int j = 0; j<n_elements; j++)
		{
			h_key_in[j] = rand();
			h_value_in[j] = h_key_in[j];
		}

		cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(d_value_in, h_value_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);

		// CUB sort:
		cudaEventRecord(start_cub, 0);
		cub::DeviceRadixSort::SortPairs(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out_cub, d_value_in, d_value_out_cub, n_elements);
		cudaEventRecord(stop_cub, 0);
		cudaEventSynchronize(stop_cub);
		cudaEventElapsedTime(&temp_time, start_cub, stop_cub);	
		cub_sort_time += temp_time;

		// Multisplit sort:
		cudaEventRecord(start_ms, 0);
		multisplit_sort_4bit_pairs(d_key_in, d_key_out_ms, d_value_in, d_value_out_ms, n_elements, sort_context_4bit);
		cudaEventRecord(stop_ms, 0);
		cudaEventSynchronize(stop_ms);
		cudaEventElapsedTime(&temp_time, start_ms, stop_ms);	
		ms_sort_time += temp_time;

		// validation:
		uint32_t n_blocks_val = (n_elements+255)/256;
		compare_vectors<<<n_blocks_val, 256>>>(d_key_out_cub, d_key_out_ms, n_elements, d_num_diffs_key + kk);
		compare_vectors<<<n_blocks_val, 256>>>(d_value_out_cub, d_value_out_ms, n_elements, d_num_diffs_values + kk);

		// Multisplit sort:
		cudaEventRecord(start_ms_5bit, 0);
		multisplit_sort_5bit_pairs(d_key_in, d_key_out_ms, d_value_in, d_value_out_ms, n_elements, sort_context_5bit);		
		cudaEventRecord(stop_ms_5bit, 0);
		cudaEventSynchronize(stop_ms_5bit);
		cudaEventElapsedTime(&temp_time, start_ms_5bit, stop_ms_5bit);	
		ms_sort_5bit_time += temp_time;

		compare_vectors<<<n_blocks_val, 256>>>(d_key_out_cub, d_key_out_ms, n_elements, d_num_diffs_key + kk);
		compare_vectors<<<n_blocks_val, 256>>>(d_value_out_cub, d_value_out_ms, n_elements, d_num_diffs_values + kk);

		// Multisplit sort:
		cudaEventRecord(start_ms_6bit, 0);
		multisplit_sort_6bit_pairs(d_key_in, d_key_out_ms, d_value_in, d_value_out_ms, n_elements, sort_context_6bit);		
		cudaEventRecord(stop_ms_6bit, 0);
		cudaEventSynchronize(stop_ms_6bit);
		cudaEventElapsedTime(&temp_time, start_ms_6bit, stop_ms_6bit);	
		ms_sort_6bit_time += temp_time;

		compare_vectors<<<n_blocks_val, 256>>>(d_key_out_cub, d_key_out_ms, n_elements, d_num_diffs_key + kk);
		compare_vectors<<<n_blocks_val, 256>>>(d_value_out_cub, d_value_out_ms, n_elements, d_num_diffs_values + kk);

		// Multisplit sort:
		cudaEventRecord(start_ms_7bit, 0);
		multisplit_sort_7bit_pairs(d_key_in, d_key_out_ms, d_value_in, d_value_out_ms, n_elements, sort_context_7bit);
		cudaEventRecord(stop_ms_7bit, 0);
		cudaEventSynchronize(stop_ms_7bit);
		cudaEventElapsedTime(&temp_time, start_ms_7bit, stop_ms_7bit);	
		ms_sort_7bit_time += temp_time;

		compare_vectors<<<n_blocks_val, 256>>>(d_key_out_cub, d_key_out_ms, n_elements, d_num_diffs_key + kk);
		compare_vectors<<<n_blocks_val, 256>>>(d_value_out_cub, d_value_out_ms, n_elements, d_num_diffs_values + kk);
		
		cudaEventRecord(start_ms_8bit, 0);
		multisplit_sort_8bit_pairs(d_key_in, d_key_out_ms, d_value_in, d_value_out_ms, n_elements, sort_context_8bit);		
		cudaEventRecord(stop_ms_8bit, 0);
		cudaEventSynchronize(stop_ms_8bit);
		cudaEventElapsedTime(&temp_time, start_ms_8bit, stop_ms_8bit);	
		ms_sort_8bit_time += temp_time;

		compare_vectors<<<n_blocks_val, 256>>>(d_key_out_cub, d_key_out_ms, n_elements, d_num_diffs_key + kk);
		compare_vectors<<<n_blocks_val, 256>>>(d_value_out_cub, d_value_out_ms, n_elements, d_num_diffs_values + kk);						
	}

	uint32_t* h_num_diffs_key = new uint32_t[kIter];
	uint32_t* h_num_diffs_values = new uint32_t[kIter];

	cudaMemcpy(h_num_diffs_key, d_num_diffs_key, sizeof(uint32_t)*kIter, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_num_diffs_values, d_num_diffs_values, sizeof(uint32_t)*kIter, cudaMemcpyDeviceToHost);

	uint32_t total_num_diffs_keys = 0;
	uint32_t total_num_diffs_values = 0;
	for(int i = 0; i<kIter; i++)
	{
		total_num_diffs_keys += h_num_diffs_key[i];
		total_num_diffs_values += h_num_diffs_values[i];
	}

	cub_sort_time /= kIter;
	ms_sort_time /= kIter;
	ms_sort_5bit_time /= kIter;
	ms_sort_6bit_time /= kIter;
	ms_sort_7bit_time /= kIter;
	ms_sort_8bit_time /= kIter;
	//======================
	printf("For %d random key-value pairs:\n", n_elements);
	printf("\tCUB radix sort finished in %.3f ms (%.3f Gpairs/s)\n", cub_sort_time, float(n_elements)/cub_sort_time/1000.0f);
	printf("\tMS 4bit radix sort finished in %.3f ms (%.3f Gpairs/s)\n", ms_sort_time, float(n_elements)/ms_sort_time/1000.0f);
	printf("\tMS 5bit radix sort finished in %.3f ms (%.3f Gpairs/s)\n", ms_sort_5bit_time, float(n_elements)/ms_sort_5bit_time/1000.0f);
	printf("\tMS 6bit radix sort finished in %.3f ms (%.3f Gpairs/s)\n", ms_sort_6bit_time, float(n_elements)/ms_sort_6bit_time/1000.0f);
	printf("\tMS 7bit radix sort finished in %.3f ms (%.3f Gpairs/s)\n", ms_sort_7bit_time, float(n_elements)/ms_sort_7bit_time/1000.0f);
	printf("\tMS 8bit radix sort finished in %.3f ms (%.3f Gpairs/s)\n", ms_sort_8bit_time, float(n_elements)/ms_sort_8bit_time/1000.0f);				
	printf("\tTotal number of differences: %d keys, %d values.\n", total_num_diffs_keys, total_num_diffs_values);
	//======================
	cudaEventDestroy(start_cub);
	cudaEventDestroy(stop_cub);
	cudaEventDestroy(start_ms);
	cudaEventDestroy(stop_ms);
	cudaEventDestroy(start_ms_5bit);
	cudaEventDestroy(stop_ms_5bit);
	cudaEventDestroy(start_ms_6bit);
	cudaEventDestroy(stop_ms_6bit);
	cudaEventDestroy(start_ms_7bit);
	cudaEventDestroy(stop_ms_7bit);
	cudaEventDestroy(start_ms_8bit);
	cudaEventDestroy(stop_ms_8bit);
	
	multisplit_sort_release_memory(sort_context_8bit);
	multisplit_sort_release_memory(sort_context_7bit);
	multisplit_sort_release_memory(sort_context_6bit);
	multisplit_sort_release_memory(sort_context_5bit);
	multisplit_sort_release_memory(sort_context_4bit);

	if(d_temp_storage_sort)CubDebugExit(g_allocator.DeviceFree(d_temp_storage_sort));

	if(h_key_in) delete[] h_key_in;
	if(h_value_in) delete[] h_value_in;
	delete[] h_num_diffs_key;
	delete[] h_num_diffs_values;
	if(d_key_in) 		cudaFree(d_key_in);
	if(d_value_in) 	cudaFree(d_value_in);
	if(d_key_out_cub) 		cudaFree(d_key_out_cub);
	if(d_value_out_cub) 	cudaFree(d_value_out_cub);
	if(d_key_out_ms) 		cudaFree(d_key_out_ms);
	if(d_value_out_ms) 	cudaFree(d_value_out_ms);
	if(d_num_diffs_key) cudaFree(d_num_diffs_key);
	if(d_num_diffs_values) cudaFree(d_num_diffs_values);
}