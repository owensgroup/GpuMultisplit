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
#include <cassert>
#include <functional>
#define CUB_STDERR
#include <cub/cub.cuh>
#include "cuda_profiler_api.h"

#include "config/config_bms.h"
#include "api/bms_api.h"

#include "kernels/bms/bms_prescan.cuh"
#include "kernels/bms/bms_postscan.cuh"
#include "kernels/bms/bms_postscan_pairs.cuh"

#include "cpu_functions.h"
// #define DEVICE_ID 0
// #define MY_ARCH_35__ // for K40c 
// #define MY_ARCH_61__ // for GTX 1080

cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
//===========================================================
template<typename key_type>
struct identity_bucket : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return uint32_t(a);	
  }
};

struct delta_func : public std::unary_function<uint32_t, uint32_t> {
  delta_func(uint32_t delta) : delta_(delta) {}
  uint32_t delta_;
  __forceinline__ __device__ __host__ uint32_t operator()(uint32_t a) const {
    return (a/delta_);
  }
};

void random_input_generator(uint32_t* input, uint32_t n, uint32_t num_buckets, uint32_t log_buckets, uint32_t random_mode, uint32_t bucket_mode, uint32_t delta = 1, double alpha = 1.0);
//===================
// main funciton:
//===================
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
	printf("=====================================\n");
  printf("Device: %s\n", devProp.name);

  // ===============================
  srand(time(NULL));
  // srand(0);

  // number of input elements
	uint32_t n_elements =  (1<<25);
	// number of buckets: 
	const uint32_t kNumBuckets = 32;
	const uint32_t kLogNumBuckets = int(ceil(log2(float(kNumBuckets))));
	// number of iterations (random trials)
	uint32_t kIter = 1;
  if(cmdOptionExists(argv, argc+argv, "-iter"))
  	kIter = atoi(getCmdOption(argv, argv+argc, "-iter")); 

  // ==== simulation mode:
  // 0: 	CUB radix sort (key-only)
  // 10: 	CUB radix sort (key-value)
  // 1:		BMS key-only
  // 12 	BMS key-value

	uint32_t 	mode = 1;
  if(cmdOptionExists(argv, argc+argv, "-mode"))
    mode = atoi(getCmdOption(argv, argv+argc, "-mode"));

	printf("=====================================\n");
	printf("Mode %d \n", mode);
	printf("=====================================\n");
	switch(mode){
		case 0:
		printf("\t CUB radix sort: key-only\n");
		break;
		case 10:
		printf("\t CUB radix sort: key-value\n");
		break;
		case 1:
		printf("\t BMS: key-only\n");
		break;
		case 12:
		printf("\t BMS: key-value\n");
		break;
	}
	printf("=====================================\n");

	bool 			validate = true;
	bool 			debug_print = false;
  if(cmdOptionExists(argv, argc+argv, "-debug"))
	  debug_print = true;	

	// random input generator parameters:
	const uint32_t random_mode = 1;
	double alpha_hockey = 0.25;
	enum bucket_distribution{UNIFORM = 0, BINOMIAL = 1, HOCKEY = 2, UNIFORM_BUCKET = 3};

	// 1: random key generation within same width buckets (delta_bucket)
	uint32_t delta_buckets = (n_elements + kNumBuckets - 1)/kNumBuckets;
	delta_func bucket_identifier(delta_buckets); // bucket identifier
	bucket_distribution bucket_d = UNIFORM;

	// 2: Identity buckets
	// identity_bucket<uint32_t> bucket_identifier;
	// bucket_distribution bucket_d = UNIFORM_BUCKET;	
	printf("\t Number of buckets: %d\n", kNumBuckets);
	printf("\t Input distribution mode: %d\n", bucket_d);
	printf("\t UNIFORM = 0, BINOMIAL = 1, HOCKEY = 2, UNIFORM_BUCKET = 3\n");
	printf("=====================================\n");
	// ===== per algorithm parameters:

	uint32_t size_sub_prob = 1;
	if(mode == 1)
		size_sub_prob = subproblem_size_bms_key_only(kNumBuckets);
	else if (mode == 12)
		size_sub_prob = subproblem_size_bms_key_value(kNumBuckets);

	//=========
	float temp_time = 0.0f;
	float sort_time = 0.0f;
	float pre_scan_time = 0.0f;
	float scan_time = 0.0f;
	float post_scan_time = 0.0f;
	// ===============================
	// allocating memory:
	// ===============================
	uint32_t 	*h_key_in = new uint32_t[n_elements];
	uint32_t	*h_value_in = new uint32_t[n_elements];
	uint32_t	*h_value_out = new uint32_t[n_elements]; 	
	uint32_t 	*h_key_out = new uint32_t[n_elements];
	uint32_t 	*h_gpu_results_key = new uint32_t[n_elements];
	uint32_t 	*h_gpu_results_value = new uint32_t[n_elements];
	uint32_t	*h_cpu_results_key = NULL; // for validation
	uint32_t	*h_cpu_results_value = NULL; // for validation

	uint32_t* d_key_in;
	uint32_t* d_key_out;
	uint32_t* d_value_in;
	uint32_t* d_value_out;

	cudaMalloc((void**)&d_key_in, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_key_out, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_value_in, sizeof(uint32_t) * n_elements);
	cudaMalloc((void**)&d_value_out, sizeof(uint32_t) * n_elements);

	cudaEvent_t start_pre, stop_pre, start_post, stop_post, start_scan, stop_scan;
	cudaEvent_t start_sort, stop_sort; // for CUB
	cudaEventCreate(&start_pre);
	cudaEventCreate(&start_post);
	cudaEventCreate(&start_scan);
	cudaEventCreate(&start_sort);
	cudaEventCreate(&stop_pre);
	cudaEventCreate(&stop_post);
	cudaEventCreate(&stop_scan);
	cudaEventCreate(&stop_sort);

	// ===============================
	// CUB's radix sort
	// ===============================
	if(mode == 0){
		random_input_generator(h_key_in, n_elements, kNumBuckets, kLogNumBuckets, bucket_d, random_mode, delta_buckets, alpha_hockey);
		cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		// key-only sort:
		void 		*d_temp_storage_sort = NULL;
		size_t 	temp_storage_bytes_sort = 0;

		CubDebugExit(cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out, n_elements, 0, int(ceil(log2(float(kNumBuckets))))));	
		CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_sort, temp_storage_bytes_sort));

		cudaEventRecord(start_sort, 0);
		cub::DeviceRadixSort::SortKeys(d_temp_storage_sort, temp_storage_bytes_sort, d_key_in, d_key_out, n_elements, 0, int(ceil(log2(float(kNumBuckets)))));
		cudaEventRecord(stop_sort, 0);
		cudaEventSynchronize(stop_sort);
		cudaEventElapsedTime(&temp_time, start_sort, stop_sort);	
		sort_time += temp_time;

		printf("CUB's radix sort finished in %.3f ms, %.3f Mkey/s\n", sort_time, float(n_elements)/sort_time/1000.0f);

		if(validate)
		{
			h_cpu_results_key = new uint32_t[n_elements];
			cpu_multisplit_general(h_key_in, h_cpu_results_key, n_elements, bucket_identifier, 0, kNumBuckets);
			cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
			bool correct = true;
			for(int i = 0; i<n_elements && correct;i++)
			{
				if(h_cpu_results_key[i] != h_gpu_results_key[i]){
					printf("### Wrong results at index %d: cpu = %d, gpu = %d\n", i, h_cpu_results_key[i], h_gpu_results_key[i]);
					correct = false;
				}
			}
			printf("Validation was done successfully!\n");
		}
		if(d_temp_storage_sort)CubDebugExit(g_allocator.DeviceFree(d_temp_storage_sort));
	}
	else if(mode == 10)
	{
		random_input_generator(h_key_in, n_elements, kNumBuckets, kLogNumBuckets, bucket_d, random_mode, delta_buckets, alpha_hockey);
		for(int k = 0; k<n_elements;k++)
			h_value_in[k] = h_key_in[k];
		cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
		cudaMemcpy(d_value_in, h_value_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		// key-value sort:
		void 		*d_temp_storage_sort_pairs = NULL;
		size_t 	temp_storage_bytes_sort_pairs = 0;

		CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage_sort_pairs, temp_storage_bytes_sort_pairs, d_key_in, d_key_out, d_value_in, d_value_out, n_elements, 0, int(ceil(log2(float(kNumBuckets))))));	
		CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_sort_pairs, temp_storage_bytes_sort_pairs));

		cudaEventRecord(start_sort, 0);
		cub::DeviceRadixSort::SortPairs(d_temp_storage_sort_pairs, temp_storage_bytes_sort_pairs, d_key_in, d_key_out, d_value_in, d_value_out, n_elements, 0, int(ceil(log2(float(kNumBuckets)))));
		cudaEventRecord(stop_sort, 0);
		cudaEventSynchronize(stop_sort);
		cudaEventElapsedTime(&temp_time, start_sort, stop_sort);	
		sort_time += temp_time;

		printf("CUB's key-value radix sort finished in %.3f ms, %.3f Mkey/s\n", sort_time, float(n_elements)/sort_time/1000.0f);

		if(validate)
		{
			h_cpu_results_key = new uint32_t[n_elements];
			h_cpu_results_value = new uint32_t[n_elements];
			cpu_multisplit_pairs_general(h_key_in, h_cpu_results_key, h_value_in, h_cpu_results_value, n_elements, bucket_identifier, 0, kNumBuckets);
			cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_gpu_results_value, d_value_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
			bool correct = true;
			for(int i = 0; i<n_elements && correct;i++)
			{
				if((h_cpu_results_key[i] != h_gpu_results_key[i]) || (h_cpu_results_value[i] != h_gpu_results_value[i])){
					printf("### Wrong results at index %d: cpu = (%d,%d), gpu = (%d,%d)\n", i, h_cpu_results_key[i], h_cpu_results_value[i], h_gpu_results_key[i], h_gpu_results_value[i]);
					correct = false;
				}
			}
			printf("Validation was done successfully!\n");
		}
		if(d_temp_storage_sort_pairs)CubDebugExit(g_allocator.DeviceFree(d_temp_storage_sort_pairs));		
	}
	// ======================================
	// Multisplit:
	// ======================================
	if(mode == 1)
	{
		// ===============================
		// CUDA options:
		//================================
		uint32_t num_sub_prob = (n_elements + size_sub_prob - 1)/(size_sub_prob);
		uint32_t num_blocks = (n_elements + size_sub_prob - 1)/(size_sub_prob);
		printf("n = %d, num_blocks = %d, size_sub_prob = %d, num_sub_prob = %d\n", n_elements, num_blocks, size_sub_prob, num_sub_prob);		
		uint32_t* d_histogram;
		cudaMalloc((void**)&d_histogram, sizeof(uint32_t)*kNumBuckets*num_sub_prob);

		void 		*d_temp_storage = NULL;
		size_t 	temp_storage_bytes = 0;

		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histogram, d_histogram, kNumBuckets * num_sub_prob);
		cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);

		if(validate){
			h_cpu_results_key = new uint32_t[n_elements];			
		}
		bool total_correctness = true;
		for(int kk = 0; kk<kIter; kk++)
		{
			random_input_generator(h_key_in, n_elements, kNumBuckets, kLogNumBuckets, bucket_d, random_mode, delta_buckets, alpha_hockey);
			cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);

			cudaMemset(d_key_out, 0, sizeof(uint32_t)*n_elements);
			cudaMemset(d_value_out, 0, sizeof(uint32_t)*n_elements);
			cudaDeviceSynchronize();

			cudaEventRecord(start_pre, 0);
			switch(kNumBuckets){
				case 256:
					if(NUM_WARPS_K_8 == 8)
						BMS_prescan_256bucket_256<NUM_ROLLS_K_8><<<num_blocks, 32*NUM_WARPS_K_8>>>(d_key_in, n_elements, d_histogram, bucket_identifier);				
				break;
				case 128:
					if(NUM_WARPS_K_7 == 8){
						BMS_prescan_128bucket_256<NUM_ROLLS_K_7><<<num_blocks, 32*NUM_WARPS_K_7>>>(d_key_in, n_elements, d_histogram, bucket_identifier);									
					}
				break;
				case 64:
					if(NUM_WARPS_K_6 == 8)
						BMS_prescan_64bucket_256<NUM_ROLLS_K_6><<<num_blocks, 32*NUM_WARPS_K_6>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
				break;
				case 32:
					if(NUM_WARPS_K_5 == 4)
						BMS_prescan_128<NUM_ROLLS_K_5, kNumBuckets , 5><<<num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
					else if(NUM_WARPS_K_5 == 8)
						BMS_prescan_256<NUM_ROLLS_K_5, kNumBuckets, 5><<<num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, n_elements, d_histogram, bucket_identifier);				
					break;
				case 16:
					if(NUM_WARPS_K_4 == 4)
						BMS_prescan_128<NUM_ROLLS_K_4, kNumBuckets , 4><<<num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
					else if(NUM_WARPS_K_4 == 8)
						BMS_prescan_256<NUM_ROLLS_K_4, kNumBuckets, 4><<<num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
				break;
				case 8:
					if(NUM_WARPS_K_3 == 4)
						BMS_prescan_128<NUM_ROLLS_K_3, kNumBuckets , 3><<<num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
					else if(NUM_WARPS_K_3 == 8)
						BMS_prescan_256<NUM_ROLLS_K_3, kNumBuckets, 3><<<num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
				break;				
				case 4:
					if(NUM_WARPS_K_2 == 4)
						BMS_prescan_128<NUM_ROLLS_K_2, kNumBuckets , 2><<<num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
					else if(NUM_WARPS_K_2 == 8)
						BMS_prescan_256<NUM_ROLLS_K_2, kNumBuckets, 2><<<num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
				break;
				case 2:
					if(NUM_WARPS_K_1 == 4)
						BMS_prescan_128<NUM_ROLLS_K_1, kNumBuckets , 1><<<num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, n_elements, d_histogram, bucket_identifier);					
					else if(NUM_WARPS_K_1 == 8)
						BMS_prescan_256<NUM_ROLLS_K_1, kNumBuckets, 1><<<num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, n_elements, d_histogram, bucket_identifier);									
				break;
			}
			cudaEventRecord(stop_pre, 0);
			cudaEventSynchronize(stop_pre);
			cudaEventElapsedTime(&temp_time, start_pre, stop_pre);	
			pre_scan_time += temp_time;

			if(debug_print){
				printf(" ### Input keys:\n");
				printGPUArray(d_key_in, n_elements, 32);
				cudaMemset(d_key_out, 0, sizeof(uint32_t) * n_elements);
				printf(" ### GPU Histogram:\n");
				printGPUArray(d_histogram, num_sub_prob * kNumBuckets, 32);
			}
			// ===== Scan stage
			cudaEventRecord(start_scan, 0);
			cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histogram, d_histogram, kNumBuckets * num_sub_prob);
			cudaEventRecord(stop_scan, 0);
			cudaEventSynchronize(stop_scan);
			cudaEventElapsedTime(&temp_time, start_scan, stop_scan);	
			scan_time += temp_time;

			if(debug_print){
				printf("### GPU scanned histogram\n");
				printGPUArray(d_histogram, num_sub_prob * kNumBuckets, 32);
			}

			// post scan stage:
			cudaEventRecord(start_post, 0);
			switch(kNumBuckets){
				case 256:
					if(NUM_WARPS_K_8 == 8)
						BMS_postscan_256bucket_256<NUM_ROLLS_K_8><<<num_blocks, 32*NUM_WARPS_K_8>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);										
				break;
				case 128:
					if(NUM_WARPS_K_7 == 8){
						BMS_postscan_128bucket_256<NUM_ROLLS_K_7><<<num_blocks, 32*NUM_WARPS_K_7>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);						
					}
				break;
				case 64:
				if(NUM_WARPS_K_6 == 8)
						BMS_postscan_64bucket_256<NUM_ROLLS_K_6><<<num_blocks, 32*NUM_WARPS_K_6>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);									
				break;
				case 32:
					if(NUM_WARPS_K_5 == 4)
						BMS_postscan_128<NUM_ROLLS_K_5, kNumBuckets, 5><<<num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_K_5 == 8)
						BMS_postscan_256<NUM_ROLLS_K_5, kNumBuckets, 5><<<num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);				
				break;
				case 16:
					if(NUM_WARPS_K_4 == 4)
						BMS_postscan_128<NUM_ROLLS_K_4, kNumBuckets, 4><<<num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_K_4 == 8)
						BMS_postscan_256<NUM_ROLLS_K_4, kNumBuckets, 4><<<num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);			
				break;
				case 8:
					if(NUM_WARPS_K_3 == 4)
						BMS_postscan_128<NUM_ROLLS_K_3, kNumBuckets, 3><<<num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_K_3 == 8)
						BMS_postscan_256<NUM_ROLLS_K_3, kNumBuckets, 3><<<num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);			
				break;						
				case 4:
					if(NUM_WARPS_K_2 == 4)
						BMS_postscan_128<NUM_ROLLS_K_2, kNumBuckets, 2><<<num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_K_2 == 8)
						BMS_postscan_256<NUM_ROLLS_K_2, kNumBuckets, 2><<<num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);					
				break;	
				case 2:
					if(NUM_WARPS_K_1 == 4)
						BMS_postscan_128<NUM_ROLLS_K_1, kNumBuckets, 1><<<num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_K_1 == 8)
						BMS_postscan_256<NUM_ROLLS_K_1, kNumBuckets, 1><<<num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, d_key_out, n_elements, d_histogram, bucket_identifier);					
				break;
			}
			cudaEventRecord(stop_post, 0);
			cudaEventSynchronize(stop_post);
			cudaEventElapsedTime(&temp_time, start_post, stop_post);	
			post_scan_time += temp_time;

			if(validate)
			{
				cpu_multisplit_general(h_key_in, h_cpu_results_key, n_elements, bucket_identifier, 0, kNumBuckets);
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
		}

		pre_scan_time /= kIter;
		post_scan_time /= kIter;
		scan_time /= kIter;
		if(debug_print){
			printf(" ### Output keys:\n");
			printGPUArray(d_key_out, n_elements, 32);
		}

		float total_time = pre_scan_time + post_scan_time + scan_time;
		printf("Multisplit key-only with %d buckets finished in %.3f ms, and %.3f Mkey/s\n", kNumBuckets, total_time, float(n_elements)/total_time/1000.0f);
		printf("\t Pre scan %.3f ms (%.2f)\n", pre_scan_time, float(pre_scan_time)/float(total_time));
		printf("\t Scan %.3f ms (%.2f)\n", scan_time, float(scan_time)/float(total_time));
		printf("\t Post scan %.3f ms (%.2f)\n", post_scan_time, float(post_scan_time)/float(total_time));

		if(validate)
		{
			if(total_correctness) printf("Validation was done successfully!\n");
			else printf("Validation failed!\n");			
		}

		//====================================
		//==
		cudaFree(d_histogram);
		cudaFree(d_temp_storage);
	}
	else if(mode == 12)
	{
		uint32_t num_sub_prob = (n_elements + size_sub_prob - 1)/(size_sub_prob);
		uint32_t num_blocks = (n_elements + size_sub_prob - 1)/(size_sub_prob);
		printf("n = %d, num_blocks = %d, size_sub_prob = %d, num_sub_prob = %d\n", n_elements, num_blocks, size_sub_prob, num_sub_prob);		
		uint32_t* d_histogram;
		cudaMalloc((void**)&d_histogram, sizeof(uint32_t)*kNumBuckets*num_sub_prob);

		void 		*d_temp_storage = NULL;
		size_t 	temp_storage_bytes = 0;

		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histogram, d_histogram, kNumBuckets * num_sub_prob);
		cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);

		if(validate){
				h_cpu_results_key = new uint32_t[n_elements];
				h_cpu_results_value = new uint32_t[n_elements];			
		}
		bool total_correctness = true;
		for(int kk = 0; kk<kIter; kk++)
		{
			random_input_generator(h_key_in, n_elements, kNumBuckets, kLogNumBuckets, bucket_d, random_mode, delta_buckets, alpha_hockey);
			for(int k = 0; k<n_elements;k++)
				h_value_in[k] = h_key_in[k];
			cudaMemcpy(d_key_in, h_key_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
			cudaMemcpy(d_value_in, h_value_in, sizeof(uint32_t) * n_elements, cudaMemcpyHostToDevice);
			cudaMemset(d_key_out, 0, sizeof(uint32_t)*n_elements);
			cudaMemset(d_value_out, 0, sizeof(uint32_t)*n_elements);
			cudaDeviceSynchronize();

			cudaEventRecord(start_pre, 0);
			switch(kNumBuckets){
				case 256:
					if(NUM_WARPS_KV_8 == 8)
						BMS_prescan_256bucket_256<NUM_ROLLS_KV_8><<<num_blocks, 32*NUM_WARPS_KV_8>>>(d_key_in, n_elements, d_histogram, bucket_identifier);				
				break;
				case 128:
					if(NUM_WARPS_KV_7 == 8)
						BMS_prescan_128bucket_256<NUM_ROLLS_KV_7><<<num_blocks, 32*NUM_WARPS_KV_7>>>(d_key_in, n_elements, d_histogram, bucket_identifier);				
				break;
				case 64:
					if(NUM_WARPS_KV_6 == 8)
						BMS_prescan_64bucket_256<NUM_ROLLS_KV_6><<<num_blocks, 32*NUM_WARPS_KV_6>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
				break;				
				case 32:
					if(NUM_WARPS_KV_5 == 4)
						BMS_prescan_128<NUM_ROLLS_KV_5, kNumBuckets, 5><<<num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_5 == 8)
						BMS_prescan_256<NUM_ROLLS_KV_5, kNumBuckets, 5><<<num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
				break;
				case 16:
					if(NUM_WARPS_KV_4 == 4)
						BMS_prescan_128<NUM_ROLLS_KV_4, kNumBuckets, 4><<<num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_4 == 8)
						BMS_prescan_256<NUM_ROLLS_KV_4, kNumBuckets, 4><<<num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
				break;
				case 8:
					if(NUM_WARPS_KV_3 == 4)
						BMS_prescan_128<NUM_ROLLS_KV_3, kNumBuckets, 3><<<num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_3 == 8)
						BMS_prescan_256<NUM_ROLLS_KV_3, kNumBuckets, 3><<<num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
				break;				
				case 4:
					if(NUM_WARPS_KV_2 == 4)
						BMS_prescan_128<NUM_ROLLS_KV_2, kNumBuckets, 2><<<num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_2 == 8)
						BMS_prescan_256<NUM_ROLLS_KV_2, kNumBuckets, 2><<<num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, n_elements, d_histogram, bucket_identifier);				
				break;
				case 2:
					if(NUM_WARPS_KV_1 == 4)
						BMS_prescan_128<NUM_ROLLS_KV_1, kNumBuckets, 1><<<num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_1 == 8)
						BMS_prescan_256<NUM_ROLLS_KV_1, kNumBuckets, 1><<<num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, n_elements, d_histogram, bucket_identifier);				
				break;				
			}
	
			cudaEventRecord(stop_pre, 0);
			cudaEventSynchronize(stop_pre);
			cudaEventElapsedTime(&temp_time, start_pre, stop_pre);	
			pre_scan_time += temp_time;

			if(debug_print){
				printf(" ### Input keys:\n");
				printGPUArray(d_key_in, n_elements, 32);

				printf(" ### GPU Histogram:\n");
				printGPUArray(d_histogram, num_sub_prob * kNumBuckets, 32);
			}
			// ===== Scan stage
			cudaEventRecord(start_scan, 0);
			cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_histogram, d_histogram, kNumBuckets * num_sub_prob);
			cudaEventRecord(stop_scan, 0);
			cudaEventSynchronize(stop_scan);
			cudaEventElapsedTime(&temp_time, start_scan, stop_scan);	
			scan_time += temp_time;

			if(debug_print){
				printf("### GPU scanned histogram\n");
				printGPUArray(d_histogram, num_sub_prob * kNumBuckets, 32);
			}
			// post scan stage:
			cudaEventRecord(start_post, 0);
			switch(kNumBuckets){
				case 256:
					if(NUM_WARPS_KV_8 == 8)
						BMS_postscan_256bucket_256_pairs<NUM_ROLLS_KV_8><<<num_blocks, 32*NUM_WARPS_KV_8>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);
				break;
				case 128:
					if(NUM_WARPS_KV_7 == 8)
						BMS_postscan_128bucket_256_pairs<NUM_ROLLS_KV_7><<<num_blocks, 32*NUM_WARPS_KV_7>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);	
				break;
				case 64:
					if(NUM_WARPS_KV_6 == 8)
						BMS_postscan_64bucket_256_pairs<NUM_ROLLS_KV_6><<<num_blocks, 32*NUM_WARPS_KV_6>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);	
				break;				
				case 32:
					if(NUM_WARPS_KV_5 == 4)
						BMS_postscan_128_pairs<NUM_ROLLS_KV_5, kNumBuckets, 5><<<num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_5 == 8 && NUM_ROLLS_KV_5 <= 4)
						BMS_postscan_256_pairs<NUM_ROLLS_KV_5, kNumBuckets, 5><<<num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);			
				break;
				case 16:
					if(NUM_WARPS_KV_4 == 4)
						BMS_postscan_128_pairs<NUM_ROLLS_KV_4, kNumBuckets, 4><<<num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_4 == 8 && NUM_ROLLS_KV_4 <= 4)
						BMS_postscan_256_pairs<NUM_ROLLS_KV_4, kNumBuckets, 4><<<num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);			
				break;
				case 8:
					if(NUM_WARPS_KV_3 == 4)
						BMS_postscan_128_pairs<NUM_ROLLS_KV_3, kNumBuckets, 3><<<num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_3 == 8 && NUM_ROLLS_KV_3 <= 4)
						BMS_postscan_256_pairs<NUM_ROLLS_KV_3, kNumBuckets, 3><<<num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);
				break;				
				case 4:
					if(NUM_WARPS_KV_2 == 4)
						BMS_postscan_128_pairs<NUM_ROLLS_KV_2, kNumBuckets, 2><<<num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_2 == 8 && NUM_ROLLS_KV_2 <= 4)
						BMS_postscan_256_pairs<NUM_ROLLS_KV_2, kNumBuckets, 2><<<num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);			
				break;				
				case 2:
					if(NUM_WARPS_KV_1 == 4)
						BMS_postscan_128_pairs<NUM_ROLLS_KV_1, kNumBuckets, 1><<<num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);
					else if(NUM_WARPS_KV_1 == 8 && NUM_ROLLS_KV_1 <= 4)
						BMS_postscan_256_pairs<NUM_ROLLS_KV_1, kNumBuckets, 1><<<num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, d_histogram, bucket_identifier);			
				break;				

			}			
			cudaEventRecord(stop_post, 0);
			cudaEventSynchronize(stop_post);
			cudaEventElapsedTime(&temp_time, start_post, stop_post);	
			post_scan_time += temp_time;

			if(validate)
			{
				cpu_multisplit_pairs_general(h_key_in, h_cpu_results_key, h_value_in, h_cpu_results_value, n_elements, bucket_identifier, 0, kNumBuckets);
				cudaMemcpy(h_gpu_results_key, d_key_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
				cudaMemcpy(h_gpu_results_value, d_value_out, sizeof(uint32_t) * n_elements, cudaMemcpyDeviceToHost);
				bool correct = true;
				for(int i = 0; i<n_elements && correct;i++)
				{
					if((h_cpu_results_key[i] != h_gpu_results_key[i]) || (h_cpu_results_value[i] != h_gpu_results_value[i])){
						printf(" ### Wrong results at index %d: cpu = (%d, %d), gpu = (%d,%d)\n", i, h_cpu_results_key[i], h_cpu_results_value[i], h_gpu_results_key[i], h_gpu_results_value[i]);
						correct = false;
					}				
				}
				total_correctness &= correct;
			}
		}

		pre_scan_time /= kIter;
		post_scan_time /= kIter;
		scan_time /= kIter;
		if(debug_print){
			printf(" ### Output keys:\n");
			printGPUArray(d_key_out, n_elements, 32);
		}

		float total_time = pre_scan_time + post_scan_time + scan_time;
		printf("Multisplit key-value with %d buckets finished in %.3f ms, and %.3f Mkey/s\n", kNumBuckets, total_time, float(n_elements)/total_time/1000.0f);
		printf("\t Pre scan %.3f ms (%.2f)\n", pre_scan_time, float(pre_scan_time)/float(total_time));
		printf("\t Scan %.3f ms (%.2f)\n", scan_time, float(scan_time)/float(total_time));
		printf("\t Post scan %.3f ms (%.2f)\n", post_scan_time, float(post_scan_time)/float(total_time));

		if(validate)
		{
			if(total_correctness) printf("Validation was done successfully!\n");
			else printf("Validation failed!\n");			
		}

		//====================================
		//==
		cudaFree(d_histogram);
		cudaFree(d_temp_storage);
	}

	// ===============================
	// releasing memory:
	// ===============================
	cudaEventDestroy(start_pre);
	cudaEventDestroy(start_scan);
	cudaEventDestroy(start_post);
	cudaEventDestroy(stop_pre);
	cudaEventDestroy(stop_scan);
	cudaEventDestroy(stop_post);
	cudaEventDestroy(start_sort);
	cudaEventDestroy(stop_sort);

	if(h_key_in) delete[] h_key_in;
	if(h_key_out) delete[] h_key_out;
	if(h_value_in) delete[] h_value_in;
	if(h_value_out) delete[] h_value_out;

	if(h_gpu_results_key) delete[] h_gpu_results_key;
	if(h_cpu_results_key) delete[] h_cpu_results_key;
	if(h_gpu_results_value) delete[] h_gpu_results_value;
	if(h_cpu_results_value) delete[] h_cpu_results_value;

	if(d_key_in) cudaFree(d_key_in);
	if(d_key_out) cudaFree(d_key_out);
	if(d_value_in) cudaFree(d_value_in);
	if(d_value_out) cudaFree(d_value_out);
}
