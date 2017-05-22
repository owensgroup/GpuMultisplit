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
#include "cpu_functions.h"

#include "api/multisplit.cuh"

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
  // 1:		Multisplit key-only
  // 12 	Multisplit key-value

	uint32_t 	mode = 1;
  if(cmdOptionExists(argv, argc+argv, "-mode"))
    mode = atoi(getCmdOption(argv, argv+argc, "-mode"));

	printf("=====================================\n");
	printf("Mode %d \n", mode);
	printf("=====================================\n");
	switch(mode){
		case 1:
		printf("\t Multisplit (BMS): key-only\n");
		break;
		case 12:
		printf("\t Multisplit (BMS): key-value\n");
		break;
	}
	printf("=====================================\n");

	bool 			validate = true;

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

	// ===============================
	// allocating memory:
	// ===============================
	uint32_t 	*h_key_in = new uint32_t[n_elements];
	uint32_t	*h_value_in = new uint32_t[n_elements];
	uint32_t	*h_value_out = new uint32_t[n_elements]; 	
	uint32_t 	*h_key_out = new uint32_t[n_elements];
	uint32_t 	*h_gpu_results_key = new uint32_t[n_elements];
	uint32_t 	*h_gpu_results_value = new uint32_t[n_elements];

	uint32_t 	*h_gpu_multisplit_offset = NULL;
	// h_gpu_multisplit_offset = new uint32_t[kNumBuckets];
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

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float temp_time = 0.0f;
	float total_time = 0.0f;

	if(mode == 1){
		multisplit_context ms_context(kNumBuckets);

		// allocating memory:
		multisplit_allocate_key_only(n_elements, ms_context);

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


			cudaEventRecord(start, 0);
			// ===== Multisplit operation ====================================================
			multisplit_key_only(d_key_in, d_key_out, n_elements, ms_context, bucket_identifier, h_gpu_multisplit_offset);
			// ===============================================================================
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&temp_time, start, stop);	
			total_time += temp_time;

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

			if(h_gpu_multisplit_offset){
				for(int j = 0; j<kNumBuckets; j++){
					printf("%d: %d, ",j, h_gpu_multisplit_offset[j]);
				}
				printf("\n");
			}
		}
		total_time /= kIter;
		printf(" Total number of elements: %d\n", n_elements);
		printf("Multisplit key-only with %d buckets finished in %.3f ms, and %.3f Mkey/s\n", kNumBuckets, total_time, float(n_elements)/total_time/1000.0f);		

		if(validate)
		{
			if(total_correctness) printf("Validation was done successfully!\n");
			else printf("Validation failed!\n");			
		}

		// releasing memory:
		multisplit_release_memory(ms_context);	
	}
	else if(mode == 12){
		multisplit_context ms_context(kNumBuckets);

		// allocating memory:
		multisplit_allocate_key_value(n_elements, ms_context);

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

			cudaEventRecord(start, 0);
			// ===== Multisplit operation ====================================================
			multisplit_key_value(d_key_in, d_value_in, d_key_out, d_value_out, n_elements, ms_context, bucket_identifier, h_gpu_multisplit_offset);
			// ===============================================================================
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&temp_time, start, stop);	
			total_time += temp_time;

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

				if(h_gpu_multisplit_offset){
					for(int j = 0; j<kNumBuckets; j++){
						printf("%d: %d, ",j, h_gpu_multisplit_offset[j]);
					}
					printf("\n");
				}				
			}
		}
		total_time /= kIter;
		printf(" Total number of elements: %d\n", n_elements);
		printf("Multisplit key-value with %d buckets finished in %.3f ms, and %.3f Mkey/s\n", kNumBuckets, total_time, float(n_elements)/total_time/1000.0f);		

		if(validate)
		{
			if(total_correctness) printf("Validation was done successfully!\n");
			else printf("Validation failed!\n");			
		}
		// releasing memory:
		multisplit_release_memory(ms_context);			
	}
	// ===============================
	// releasing memory:
	// ===============================
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if(h_key_in) delete[] h_key_in;
	if(h_key_out) delete[] h_key_out;
	if(h_value_in) delete[] h_value_in;
	if(h_value_out) delete[] h_value_out;

	if(h_gpu_results_key) delete[] h_gpu_results_key;
	if(h_cpu_results_key) delete[] h_cpu_results_key;
	if(h_gpu_results_value) delete[] h_gpu_results_value;
	if(h_cpu_results_value) delete[] h_cpu_results_value;
	if(h_gpu_multisplit_offset) delete[] h_gpu_multisplit_offset;

	if(d_key_in) cudaFree(d_key_in);
	if(d_key_out) cudaFree(d_key_out);
	if(d_value_in) cudaFree(d_value_in);
	if(d_value_out) cudaFree(d_value_out);
}			
