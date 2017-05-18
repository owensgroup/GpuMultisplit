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
#include <algorithm>
#include <functional>
#define CUB_STDERR
#include <cub/cub.cuh>
#include "cuda_profiler_api.h"

#include "kernels/histogram/multisplit2_histograms.cuh"
#include "cpu_functions.h"
#include "gpu_functions.cuh"
//==========================================================
int main(int argc, char** argv)
{
	int devCount;
  cudaGetDeviceCount(&devCount);
  cudaDeviceProp devProp;
  if(devCount){
    cudaSetDevice(DEVICE_ID__); 
    cudaGetDeviceProperties(&devProp, DEVICE_ID__);
  }
  printf("Device: %s\n", devProp.name);

	int n_elements = (1<<25);
	const int histo_size = 32;
	const uint32_t kLogBuckets = 5;
	int n_offsets = histo_size + 1; // for non-atomic HistogramEven
	int n_levels = histo_size + 1; // for HistogramRange
	const float lower_level = 0.0f;
	const float upper_level = 256.0f;
	float inverse_delta = 1.0f/((upper_level - lower_level)/static_cast<float>(histo_size));
	uint32_t kIter = 1;
  if(cmdOptionExists(argv, argc+argv, "-iter"))
  	kIter = atoi(getCmdOption(argv, argv+argc, "-iter")); 

	uint32_t kMode = 0; // if 0: HistogramEven, 1:HistogramRange
  if(cmdOptionExists(argv, argc+argv, "-mode"))
    kMode = atoi(getCmdOption(argv, argv+argc, "-mode"));
	printf("=====================================\n");
	printf("Mode %d \n", kMode);
	printf("=====================================\n");
	if(kMode == 0) printf("HistogramEven\n");
	else if(kMode == 1) printf("HistogramRange\n");
	printf("=====================================\n");
	bool debug_print = false;
	//====== Multisplit parameters:
	const uint32_t num_warps_ms = 4;
	const uint32_t num_rolls_ms = 8;
	uint32_t size_sub_prob = num_rolls_ms * num_warps_ms * 32;
	uint32_t num_sub_prob = (n_elements + size_sub_prob - 1)/(size_sub_prob);
	uint32_t num_blocks = (n_elements + size_sub_prob - 1)/(size_sub_prob);

	//=============================

	n_elements = (n_elements/size_sub_prob)*size_sub_prob;

	float* h_samples = new float[n_elements];
	float* h_levels = new float[n_levels];
	uint32_t* h_num_diffs_key = new uint32_t[kIter];
	int* h_offsets = new int[n_offsets];
	for(int i = 0;i<n_offsets;i++)
		h_offsets[i] = i*num_sub_prob;

	float* d_samples;
	int* d_histogram_cub;
	int* d_histogram_ms;
	int* d_temp_ms;
	int* 			d_offsets;
	float* 		d_levels;
	cudaMalloc((void**)&d_samples, sizeof(float) * n_elements);
	cudaMalloc((void**)&d_histogram_cub, sizeof(int) * (n_levels));
	cudaMalloc((void**)&d_histogram_ms, sizeof(int) * histo_size);
	cudaMalloc((void**)&d_temp_ms, sizeof(int) * num_sub_prob * histo_size);
	cudaMalloc((void**)&d_offsets, sizeof(int) * (n_offsets));
	cudaMalloc((void**)&d_levels, sizeof(int) * (n_levels));
	cudaMemcpy(d_offsets, h_offsets, sizeof(int) * (n_offsets), cudaMemcpyHostToDevice);

	uint32_t* d_num_diffs_key;
	cudaMalloc((void**)&d_num_diffs_key, sizeof(uint32_t) * kIter);
	cudaMemset(d_num_diffs_key, 0, sizeof(uint32_t) * kIter);


	// === For HistogramEven
	void*    d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
	    d_samples, d_histogram_cub, n_levels, lower_level, upper_level, n_elements);
	cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);

	// === For HistogramRange:
	void*    d_temp_storage_histo_range = NULL;
	size_t   temp_storage_bytes_histo_range = 0;
	cub::DeviceHistogram::HistogramRange(d_temp_storage_histo_range, temp_storage_bytes_histo_range,
	    d_samples, d_histogram_cub, n_levels, d_levels, n_elements);
	cudaMalloc((void**)&d_temp_storage_histo_range, temp_storage_bytes_histo_range);

	//==============
	srand(time(NULL));

	float temp_time = 0.0f;
	float time_cub = 0.0f;
	float time_ms_atomic = 0.0f;

	cudaEvent_t start_cub, stop_cub;
	cudaEventCreate(&start_cub);
	cudaEventCreate(&stop_cub);

	cudaEvent_t start_ms_atomic, stop_ms_atomic;
	cudaEventCreate(&start_ms_atomic);
	cudaEventCreate(&stop_ms_atomic);

	cudaEvent_t start_ms, stop_ms;
	cudaEventCreate(&start_ms);
	cudaEventCreate(&stop_ms);

	//=============================================================================
	// for HistogramEven simulations:
	//=============================================================================
	if(kMode == 0){
		for(int kk = 0; kk<kIter; kk++)
		{
			// generating random inputs:
			for(int i = 0; i<n_elements; i++)
			{
				h_samples[i] = lower_level + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(upper_level - lower_level)));
			}

			//==== transferring into GPU:
			cudaMemcpy(d_samples, h_samples, sizeof(float) * n_elements, cudaMemcpyHostToDevice);
			cudaMemset(d_histogram_cub, 0, sizeof(int) * n_levels);
			cudaEventRecord(start_cub, 0);
			// Running CUB: =================
			cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
		    d_samples, d_histogram_cub, n_levels, lower_level, upper_level, n_elements);		
			//===============================
			cudaEventRecord(stop_cub, 0);
			cudaEventSynchronize(stop_cub);
			cudaEventElapsedTime(&temp_time, start_cub, stop_cub);	
			time_cub += temp_time;

			// Atomic based method:
			cudaMemset(d_histogram_ms, 0, sizeof(int) * histo_size);
			cudaEventRecord(start_ms_atomic, 0);
			if(histo_size <= 32){
				if(num_warps_ms == 4)
					multisplit2_histogram_even_128<num_rolls_ms, histo_size, kLogBuckets><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms,  lower_level, upper_level, inverse_delta);		
				else if(num_warps_ms == 8)
					multisplit2_histogram_even_256<num_rolls_ms, histo_size, kLogBuckets><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms,  lower_level, upper_level, inverse_delta);
			}
			else if(histo_size == 64){
				multisplit2_histogram_even_64bin_128<num_rolls_ms><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms,  lower_level, upper_level, inverse_delta);				
			}
			else if(histo_size == 128){
				multisplit2_histogram_even_128bin_128<num_rolls_ms><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms,  lower_level, upper_level, inverse_delta);				
			}
			else if(histo_size == 256){
				multisplit2_histogram_even_256bin_128<num_rolls_ms><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms,  lower_level, upper_level, inverse_delta);
			}
			cudaEventRecord(stop_ms_atomic, 0);
			cudaEventSynchronize(stop_ms_atomic);
			cudaEventElapsedTime(&temp_time, start_ms_atomic, stop_ms_atomic);	
			time_ms_atomic += temp_time;

			uint32_t n_blocks_val = (histo_size+255)/256;
			compare_vectors<<<n_blocks_val, 256>>>(d_histogram_cub, d_histogram_ms, histo_size, d_num_diffs_key + kk);

			// printf("Input:\n");
			// printGPUArray(d_samples, n_elements, 32);
			
			if(debug_print){
				printf("Histogram CUB: \n");
				printGPUArray(d_histogram_cub, histo_size, 32);

				printf("Histogram Multisplit Atomic: \n");
				printGPUArray(d_histogram_ms, histo_size, 32);		
			}
			// cudaMemset(d_histogram_ms, 0, sizeof(int) * histo_size);

			// compare_vectors<<<n_blocks_val, 256>>>(d_histogram_cub, d_histogram_ms, histo_size, d_num_diffs_key + kk);

			// printf("Histogram Multisplit: \n");
			// printGPUArray(d_histogram_ms, histo_size, 32);		
		}

		time_cub /= kIter;
		time_ms_atomic /= kIter;

		printf("HistogramEven CUB: \n\t%d elements, %d bins, %.3f ms (%.3f Gelements/s)\n", n_elements, histo_size, time_cub, float(n_elements)/time_cub/1000.0f);
		printf("HistogramEven Multisplit Atomic: \n\t%d elements, %d bins, %.3f ms (%.3f Gelements/s)\n", n_elements, histo_size, time_ms_atomic, float(n_elements)/time_ms_atomic/1000.0f);

		cudaMemcpy(h_num_diffs_key, d_num_diffs_key, sizeof(uint32_t)*kIter, cudaMemcpyDeviceToHost);

		uint32_t total_num_diffs_keys = 0;
		for(int i = 0; i<kIter; i++)
		{
			total_num_diffs_keys += h_num_diffs_key[i];
		}
		printf("Total number of differences: %d keys.\n", total_num_diffs_keys);
	}
	//=============================================================================
	// for HistogramRange simulations:
	//=============================================================================
	if(kMode == 1){
		cudaMemset(d_histogram_cub, 0, sizeof(int) * n_levels);
		cudaMemset(d_num_diffs_key, 0, sizeof(uint32_t) * kIter);
		float time_cub_range = 0.0f;
		float time_ms_atomic_range = 0.0f;
		// float time_ms_range = 0.0f;
		for(int kk = 0; kk<kIter; kk++)
		{
			// generating sample levels:
			h_levels[0] = lower_level;
			h_levels[n_levels - 1] = upper_level;
			for(int i = 1; i<n_levels-1;i++)
			{
				h_levels[i] = lower_level + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(upper_level - lower_level)));
			}
			std::sort(h_levels, h_levels + n_levels);

			// generating random inputs:
			for(int i = 0; i<n_elements; i++)
			{
				h_samples[i] = lower_level + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(upper_level - lower_level)));
			}

			//==== transferring into GPU:
			cudaMemcpy(d_samples, h_samples, sizeof(float) * n_elements, cudaMemcpyHostToDevice);
			cudaMemcpy(d_levels, h_levels, sizeof(float) * n_levels, cudaMemcpyHostToDevice);
			cudaMemset(d_histogram_cub, 0, sizeof(int) * n_levels);

			cudaEventRecord(start_cub, 0);
			// Running CUB: =================
			cub::DeviceHistogram::HistogramRange(d_temp_storage_histo_range, temp_storage_bytes_histo_range,
			    d_samples, d_histogram_cub, n_levels, d_levels, n_elements);		
	    //===============================
			cudaEventRecord(stop_cub, 0);
			cudaEventSynchronize(stop_cub);
			cudaEventElapsedTime(&temp_time, start_cub, stop_cub);	
			time_cub_range += temp_time;

			// printf("Input:\n");
			// printGPUArray(d_samples, n_elements, 32);

			if(debug_print){
				printf("HistogramRange CUB: \n");
				printGPUArray(d_histogram_cub, histo_size, 32);
			}
			// Atomic based method:
			cudaMemset(d_histogram_ms, 0, sizeof(int) * histo_size);
			cudaEventRecord(start_ms_atomic, 0);
			if(histo_size <= 32)
				multisplit2_histogram_range_128<num_rolls_ms, histo_size, kLogBuckets><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms, d_levels);
			else if(histo_size == 64)
				multisplit2_histogram_range_64bin_128<num_rolls_ms><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms, d_levels);				
			else if(histo_size == 128)
					multisplit2_histogram_range_128bin_128<num_rolls_ms><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms, d_levels);
			else if(histo_size == 256)
				multisplit2_histogram_range_256bin_128<num_rolls_ms><<<num_blocks, 32*num_warps_ms>>>(d_samples, n_elements, d_histogram_ms, d_levels);				
			cudaEventRecord(stop_ms_atomic, 0);
			cudaEventSynchronize(stop_ms_atomic);
			cudaEventElapsedTime(&temp_time, start_ms_atomic, stop_ms_atomic);	
			time_ms_atomic_range += temp_time;

			uint32_t n_blocks_val = (histo_size+255)/256;
			compare_vectors<<<n_blocks_val, 256>>>(d_histogram_cub, d_histogram_ms, histo_size, d_num_diffs_key + kk);

			if(debug_print){
				printf("HistogramRange atomic-Multisplit: \n");
				printGPUArray(d_histogram_ms, histo_size, 32);
			}

			// compare_vectors<<<n_blocks_val, 256>>>(d_histogram_cub, d_histogram_ms, histo_size, d_num_diffs_key + kk);
		}
		time_cub_range /= kIter;
		time_ms_atomic_range /= kIter;
		// time_ms_range /= kIter;

		printf("HistogramRange CUB: %d elements, %d bins, %.3f ms (%.3f Gelements/s)\n", n_elements, histo_size, time_cub_range, float(n_elements)/time_cub_range/1000.0f);
		printf("HistogramRange Atomic MS: %d elements, %d bins, %.3f ms (%.3f Gelements/s)\n", n_elements, histo_size, time_ms_atomic_range, float(n_elements)/time_ms_atomic_range/1000.0f);		

		cudaMemcpy(h_num_diffs_key, d_num_diffs_key, sizeof(uint32_t)*kIter, cudaMemcpyDeviceToHost);

		uint32_t total_num_diffs_keys = 0;
		for(int i = 0; i<kIter; i++)
		{
			total_num_diffs_keys += h_num_diffs_key[i];
		}
		printf("Total number of differences: %d keys.\n", total_num_diffs_keys);
	}

	cudaEventDestroy(start_cub);
	cudaEventDestroy(stop_cub);
	cudaEventDestroy(start_ms_atomic);
	cudaEventDestroy(stop_ms_atomic);
	cudaEventDestroy(start_ms);
	cudaEventDestroy(stop_ms);


	if(d_temp_storage) cudaFree(d_temp_storage);
	if(d_temp_storage_histo_range) cudaFree(d_temp_storage_histo_range);
	if(d_samples) 	cudaFree(d_samples);
	if(d_histogram_cub) cudaFree(d_histogram_cub);
	if(d_histogram_ms) cudaFree(d_histogram_ms);
	if(d_temp_ms) cudaFree(d_temp_ms);
	if(d_num_diffs_key) cudaFree(d_num_diffs_key);
	if(h_samples) delete[] h_samples;
	if(h_num_diffs_key) delete[] h_num_diffs_key;
	if(h_levels) delete[] h_levels;
}