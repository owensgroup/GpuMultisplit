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

#ifndef MULTISPLIT_SORT__
#define MULTISPLIT_SORT__
#include <algorithm>
#include <cub/cub.cuh>
#include "kernels/bms/bms_prescan.cuh"
#include "kernels/bms/bms_postscan.cuh"
#include "kernels/bms/bms_postscan_pairs.cuh"

#define NUM_WARPS_SORT_8BIT 8
#define NUM_ROLLS_SORT_8BIT 4

#define NUM_WARPS_SORT_7BIT 8
#define NUM_ROLLS_SORT_7BIT 4

#define NUM_WARPS_SORT_6BIT 8
#define NUM_ROLLS_SORT_6BIT 4

#define NUM_WARPS_SORT_5BIT 4
#define NUM_ROLLS_SORT_5BIT 7

#define NUM_WARPS_SORT_4BIT 4
#define NUM_ROLLS_SORT_4BIT 7

#define NUM_WARPS_SORT_2BIT 4
#define NUM_ROLLS_SORT_2BIT 7

#define NUM_WARPS_SORT_8BIT_PAIRS 8
#define NUM_ROLLS_SORT_8BIT_PAIRS 4

#define NUM_WARPS_SORT_7BIT_PAIRS 8
#define NUM_ROLLS_SORT_7BIT_PAIRS 4

#define NUM_WARPS_SORT_6BIT_PAIRS 8
#define NUM_ROLLS_SORT_6BIT_PAIRS 4

#define NUM_WARPS_SORT_5BIT_PAIRS 8
#define NUM_ROLLS_SORT_5BIT_PAIRS 4

#define NUM_WARPS_SORT_4BIT_PAIRS 8
#define NUM_ROLLS_SORT_4BIT_PAIRS 4

#define NUM_WARPS_SORT_2BIT_PAIRS 8
#define NUM_ROLLS_SORT_2BIT_PAIRS 4

template<uint32_t x, typename key_type>
struct ms_bucket_sort_4bit : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return uint32_t((a >> x) & 0x0F);	
  }
};
template<uint32_t x, typename key_type>
struct ms_bucket_sort_5bit : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return uint32_t((a >> x) & 0x1F);	
  }
};
template<uint32_t x, typename key_type>
struct ms_bucket_sort_6bit : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return uint32_t((a >> x) & 0x3F);	
  }
};
template<uint32_t x, typename key_type>
struct ms_bucket_sort_7bit : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return uint32_t((a >> x) & 0x7F);	
  }
};
template<uint32_t x, typename key_type>
struct ms_bucket_sort_8bit : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return uint32_t((a >> x) & 0xFF);	
  }
};
template<uint32_t x, typename key_type>
struct ms_bucket_sort_2bit : public std::unary_function<key_type, uint32_t> {
   __forceinline__ __device__ __host__ uint32_t operator()(key_type a) const {
    return uint32_t((a >> x) & 0x03);	
  }
};
//==========================================
class ms_sort_context{
public:
	void*				d_temp_storage;
	size_t   		temp_storage_bytes;
	uint32_t*		d_histogram;
	uint32_t*		d_key_temp_ms;
	uint32_t*		d_value_temp_ms;
	ms_sort_context()
	{
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		d_histogram = NULL;
		d_key_temp_ms = NULL;
		d_value_temp_ms = NULL;
	}
	~ms_sort_context(){}
};
//=========================================
void multisplit_sort_8bit_allocate(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 8-bit BMS
  const uint32_t num_warps_8bit = NUM_WARPS_SORT_8BIT;
  const uint32_t num_roll_8bit = NUM_ROLLS_SORT_8BIT;
  const uint32_t num_sub_prob_8bit = (n_elements + (32 * num_warps_8bit * num_roll_8bit) - 1) / (32 * num_warps_8bit * num_roll_8bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * 256 * num_sub_prob_8bit);
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 256 * num_sub_prob_8bit);

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
}

void multisplit_sort_8bit_allocate_pairs(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 8-bit BMS
  const uint32_t num_warps_8bit = NUM_WARPS_SORT_8BIT_PAIRS;
  const uint32_t num_roll_8bit = NUM_ROLLS_SORT_8BIT_PAIRS;
  const uint32_t num_sub_prob_8bit = (n_elements + (32 * num_warps_8bit * num_roll_8bit) - 1) / (32 * num_warps_8bit * num_roll_8bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * 256 * num_sub_prob_8bit);
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 256 * num_sub_prob_8bit);

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
  cudaMalloc((void**)&context.d_value_temp_ms, sizeof(uint32_t) * n_elements);
}

void multisplit_sort_7bit_allocate(uint32_t n_elements, ms_sort_context& context)
{
	// parameters for 7-bit BMS
  const uint32_t num_warps_7bit = NUM_WARPS_SORT_7BIT;
  const uint32_t num_roll_7bit = NUM_ROLLS_SORT_7BIT;
  const uint32_t num_sub_prob_7bit = (n_elements + (32 * num_warps_7bit * num_roll_7bit) - 1) / (32 * num_warps_7bit * num_roll_7bit);

  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * std::max(128 * num_sub_prob_7bit, 16 * num_sub_prob_4bit));
	
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, std::max(128 * num_sub_prob_7bit, 16 * num_sub_prob_4bit));

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
}
void multisplit_sort_7bit_allocate_pairs(uint32_t n_elements, ms_sort_context& context)
{
	// parameters for 7-bit BMS
  const uint32_t num_warps_7bit = NUM_WARPS_SORT_7BIT_PAIRS;
  const uint32_t num_roll_7bit = NUM_ROLLS_SORT_7BIT_PAIRS;
  const uint32_t num_sub_prob_7bit = (n_elements + (32 * num_warps_7bit * num_roll_7bit) - 1) / (32 * num_warps_7bit * num_roll_7bit);

  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT_PAIRS;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT_PAIRS;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * std::max(128 * num_sub_prob_7bit, 16 * num_sub_prob_4bit));
	
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, std::max(128 * num_sub_prob_7bit, 16 * num_sub_prob_4bit));
	
  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
  cudaMalloc((void**)&context.d_value_temp_ms, sizeof(uint32_t) * n_elements);
}

void multisplit_sort_6bit_allocate(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 6-bit BMS
  const uint32_t num_warps_6bit = NUM_WARPS_SORT_6BIT;
  const uint32_t num_roll_6bit = NUM_ROLLS_SORT_6BIT;
  const uint32_t num_sub_prob_6bit = (n_elements + (32 * num_warps_6bit * num_roll_6bit) - 1) / (32 * num_warps_6bit * num_roll_6bit);

  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * std::max(64 * num_sub_prob_6bit, 16 * num_sub_prob_4bit));
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, std::max(64 * num_sub_prob_6bit, 16 * num_sub_prob_4bit));

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
}

void multisplit_sort_6bit_allocate_pairs(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 6-bit BMS
  const uint32_t num_warps_6bit = NUM_WARPS_SORT_6BIT_PAIRS;
  const uint32_t num_roll_6bit = NUM_ROLLS_SORT_6BIT_PAIRS;
  const uint32_t num_sub_prob_6bit = (n_elements + (32 * num_warps_6bit * num_roll_6bit) - 1) / (32 * num_warps_6bit * num_roll_6bit);

  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * std::max(64 * num_sub_prob_6bit, 16 * num_sub_prob_4bit));
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, std::max(64 * num_sub_prob_6bit, 16 * num_sub_prob_4bit));

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
  cudaMalloc((void**)&context.d_value_temp_ms, sizeof(uint32_t) * n_elements);
}

void multisplit_sort_5bit_allocate(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 5-bit BMS
  const uint32_t num_warps_5bit = NUM_WARPS_SORT_5BIT;
  const uint32_t num_roll_5bit = NUM_ROLLS_SORT_5BIT;
  const uint32_t num_sub_prob_5bit = (n_elements + (32 * num_warps_5bit * num_roll_5bit) - 1) / (32 * num_warps_5bit * num_roll_5bit);

  // parameters for 2-bit BMS 
  const uint32_t num_warps_2bit = NUM_WARPS_SORT_2BIT;
  const uint32_t num_roll_2bit = NUM_ROLLS_SORT_2BIT;
  const uint32_t num_sub_prob_2bit = (n_elements + (32 * num_warps_2bit * num_roll_2bit) - 1) / (32 * num_warps_2bit * num_roll_2bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * std::max(32 * num_sub_prob_5bit, 4 * num_sub_prob_2bit));
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, std::max(32 * num_sub_prob_5bit, 4 * num_sub_prob_2bit));

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
}

void multisplit_sort_5bit_allocate_pairs(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 5-bit BMS
  const uint32_t num_warps_5bit = NUM_WARPS_SORT_5BIT_PAIRS;
  const uint32_t num_roll_5bit = NUM_ROLLS_SORT_5BIT_PAIRS;
  const uint32_t num_sub_prob_5bit = (n_elements + (32 * num_warps_5bit * num_roll_5bit) - 1) / (32 * num_warps_5bit * num_roll_5bit);

  // parameters for 4-bit BMS 
  const uint32_t num_warps_2bit = NUM_WARPS_SORT_2BIT_PAIRS;
  const uint32_t num_roll_2bit = NUM_ROLLS_SORT_2BIT_PAIRS;
  const uint32_t num_sub_prob_2bit = (n_elements + (32 * num_warps_2bit * num_roll_2bit) - 1) / (32 * num_warps_2bit * num_roll_2bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * std::max(32 * num_sub_prob_5bit, 4 * num_sub_prob_2bit));
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, std::max(32 * num_sub_prob_5bit, 4 * num_sub_prob_2bit));

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
  cudaMalloc((void**)&context.d_value_temp_ms, sizeof(uint32_t) * n_elements);
}

void multisplit_sort_4bit_allocate(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * 16 * num_sub_prob_4bit);
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 16 * num_sub_prob_4bit);

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
}
void multisplit_sort_4bit_allocate_pairs(uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT_PAIRS;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT_PAIRS;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);

  cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * 16 * num_sub_prob_4bit);
  
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 16 * num_sub_prob_4bit);

  cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);

  cudaMalloc((void**)&context.d_key_temp_ms, sizeof(uint32_t) * n_elements);
  cudaMalloc((void**)&context.d_value_temp_ms, sizeof(uint32_t) * n_elements);
}


void multisplit_sort_release_memory(ms_sort_context& context)
{
	if(context.d_histogram) cudaFree(context.d_histogram);
	if(context.d_temp_storage) cudaFree(context.d_temp_storage);
	if(context.d_key_temp_ms) cudaFree(context.d_key_temp_ms);
	if(context.d_value_temp_ms) cudaFree(context.d_value_temp_ms);
}
//================================================================================
void multisplit_sort_7bit(uint32_t* d_key_in, uint32_t* d_key_out_ms, uint32_t n_elements, ms_sort_context& context)
{
	// parameters for 7-bit BMS
  const uint32_t num_warps_7bit = NUM_WARPS_SORT_7BIT;
  const uint32_t num_roll_7bit = NUM_ROLLS_SORT_7BIT;
  const uint32_t num_sub_prob_7bit = (n_elements + (32 * num_warps_7bit * num_roll_7bit) - 1) / (32 * num_warps_7bit * num_roll_7bit);

  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);


  ms_bucket_sort_4bit<0,uint32_t> bucket_1;
  ms_bucket_sort_7bit<4,uint32_t> bucket_2;
  ms_bucket_sort_7bit<11,uint32_t> bucket_3;
  ms_bucket_sort_7bit<18,uint32_t> bucket_4;
  ms_bucket_sort_7bit<25,uint32_t> bucket_5;

  // === 1:
  BMS_prescan_128<num_roll_4bit, 16, 4><<<num_sub_prob_4bit, 32*num_warps_4bit>>>(d_key_in, n_elements, context.d_histogram, bucket_1);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 16 * num_sub_prob_4bit);
  BMS_postscan_128<num_roll_4bit, 16, 4><<<num_sub_prob_4bit, 32*num_warps_4bit>>>(d_key_in, d_key_out_ms, n_elements, context.d_histogram, bucket_1);

  // === 2:
  BMS_prescan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_2);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 128 * num_sub_prob_7bit);
  BMS_postscan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_2);


  // === 3:
  BMS_prescan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_3);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 128 * num_sub_prob_7bit);
  BMS_postscan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_3);

  // === 4:
  BMS_prescan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_4);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 128 * num_sub_prob_7bit);
  BMS_postscan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(d_key_out_ms, context.d_key_temp_ms,n_elements, context.d_histogram, bucket_4);

  // === 5:
  BMS_prescan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_5);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 128 * num_sub_prob_7bit);
  BMS_postscan_128bucket_256<num_roll_7bit><<<num_sub_prob_7bit, 32*num_warps_7bit>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_5);		
}

void multisplit_sort_4bit(uint32_t* d_key_in, uint32_t* d_key_out_ms, uint32_t n_elements, ms_sort_context& context)
{
  const uint32_t num_warps = NUM_WARPS_SORT_4BIT;
  const uint32_t num_roll = NUM_ROLLS_SORT_4BIT;
  const uint32_t num_sub_prob = (n_elements + (32 * num_warps * num_roll) - 1) / (32 * num_warps * num_roll);

  ms_bucket_sort_4bit<0,uint32_t> bucket_1;
  ms_bucket_sort_4bit<4,uint32_t> bucket_2;
  ms_bucket_sort_4bit<8,uint32_t> bucket_3;
  ms_bucket_sort_4bit<12,uint32_t> bucket_4;
  ms_bucket_sort_4bit<16,uint32_t> bucket_5;
  ms_bucket_sort_4bit<20,uint32_t> bucket_6;
  ms_bucket_sort_4bit<24,uint32_t> bucket_7;
  ms_bucket_sort_4bit<28,uint32_t> bucket_8;

  const uint32_t kNumBuckets = 16;
  // === 1:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_in, n_elements, context.d_histogram, bucket_1);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_in, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_1); 

  // === 2:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_2);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, d_key_out_ms,n_elements, context.d_histogram, bucket_2);  

  // === 3:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_3);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_3); 

  // === 4:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_4);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_4); 

  // === 5:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_5);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_5); 

  // === 6:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_6);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_6);

  // === 7:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_7);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_7); 

  // === 8:
  BMS_prescan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_8);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, kNumBuckets * num_sub_prob);
  BMS_postscan_128<num_roll, kNumBuckets, 4><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_8);
}

void multisplit_sort_5bit(uint32_t* d_key_in, uint32_t* d_key_out_ms, uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 5-bit BMS
  const uint32_t num_warps_5bit = NUM_WARPS_SORT_5BIT;
  const uint32_t num_roll_5bit = NUM_ROLLS_SORT_5BIT;
  const uint32_t num_sub_prob_5bit = (n_elements + (32 * num_warps_5bit * num_roll_5bit) - 1) / (32 * num_warps_5bit * num_roll_5bit);

  // parameters for 2-bit BMS 
  const uint32_t num_warps_2bit = NUM_WARPS_SORT_2BIT;
  const uint32_t num_roll_2bit = NUM_ROLLS_SORT_2BIT;
  const uint32_t num_sub_prob_2bit = (n_elements + (32 * num_warps_2bit * num_roll_2bit) - 1) / (32 * num_warps_2bit * num_roll_2bit);

  ms_bucket_sort_5bit<0,uint32_t> bucket_1;
  ms_bucket_sort_5bit<5,uint32_t> bucket_2;
  ms_bucket_sort_5bit<10,uint32_t> bucket_3;
  ms_bucket_sort_5bit<15,uint32_t> bucket_4;
  ms_bucket_sort_5bit<20,uint32_t> bucket_5;
  ms_bucket_sort_5bit<25,uint32_t> bucket_6;
  ms_bucket_sort_2bit<30,uint32_t> bucket_7;

  // === 1:
  BMS_prescan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_in, n_elements, context.d_histogram, bucket_1);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 32 * num_sub_prob_5bit);
  BMS_postscan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_in, d_key_out_ms, n_elements, context.d_histogram, bucket_1); 

  // === 2:
  BMS_prescan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_2);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 32 * num_sub_prob_5bit);
  BMS_postscan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_out_ms, context.d_key_temp_ms,n_elements, context.d_histogram, bucket_2); 

  // === 3:
  BMS_prescan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_3);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 32 * num_sub_prob_5bit);
  BMS_postscan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_3);  

  // === 4:
  BMS_prescan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_4);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 32 * num_sub_prob_5bit);
  BMS_postscan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_4);  

  // === 5:
  BMS_prescan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_5);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 32 * num_sub_prob_5bit);
  BMS_postscan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_5);  

  // === 6:
  BMS_prescan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_6);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 32 * num_sub_prob_5bit);
  BMS_postscan_128<num_roll_5bit, 32, 5><<<num_sub_prob_5bit, 32*num_warps_5bit>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_6);

  // === 7:
  BMS_prescan_128<num_roll_2bit, 4, 2><<<num_sub_prob_2bit, 32*num_warps_2bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_7);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 4 * num_sub_prob_2bit);
  BMS_postscan_128<num_roll_2bit, 4, 2><<<num_sub_prob_2bit, 32*num_warps_2bit>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_7); 
}

void multisplit_sort_6bit(uint32_t* d_key_in, uint32_t* d_key_out_ms, uint32_t n_elements, ms_sort_context& context)
{
  // parameters for 6-bit BMS
  const uint32_t num_warps_6bit = NUM_WARPS_SORT_6BIT;
  const uint32_t num_roll_6bit = NUM_ROLLS_SORT_6BIT;
  const uint32_t num_sub_prob_6bit = (n_elements + (32 * num_warps_6bit * num_roll_6bit) - 1) / (32 * num_warps_6bit * num_roll_6bit);

  // parameters for 4-bit BMS 
  const uint32_t num_warps_4bit = NUM_WARPS_SORT_4BIT;
  const uint32_t num_roll_4bit = NUM_ROLLS_SORT_4BIT;
  const uint32_t num_sub_prob_4bit = (n_elements + (32 * num_warps_4bit * num_roll_4bit) - 1) / (32 * num_warps_4bit * num_roll_4bit);

  ms_bucket_sort_4bit<0,uint32_t> bucket_1;
  ms_bucket_sort_4bit<4,uint32_t> bucket_2;
  ms_bucket_sort_6bit<8,uint32_t> bucket_3;
  ms_bucket_sort_6bit<14,uint32_t> bucket_4;
  ms_bucket_sort_6bit<20,uint32_t> bucket_5;
  ms_bucket_sort_6bit<26,uint32_t> bucket_6;

  // 1:
  BMS_prescan_128<num_roll_4bit, 16, 4><<<num_sub_prob_4bit, 32*num_warps_4bit>>>(d_key_in, n_elements, context.d_histogram, bucket_1);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 16 * num_sub_prob_4bit);
  BMS_postscan_128<num_roll_4bit, 16, 4><<<num_sub_prob_4bit, 32*num_warps_4bit>>>(d_key_in, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_1);

  // 2:
  BMS_prescan_128<num_roll_4bit, 16, 4><<<num_sub_prob_4bit, 32*num_warps_4bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_2);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 16 * num_sub_prob_4bit);
  BMS_postscan_128<num_roll_4bit, 16, 4><<<num_sub_prob_4bit, 32*num_warps_4bit>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_2);

  // === 3:
  BMS_prescan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_3);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 64 * num_sub_prob_6bit);
  BMS_postscan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_3);

  // === 4:
  BMS_prescan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_4);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 64 * num_sub_prob_6bit);
  BMS_postscan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(context.d_key_temp_ms, d_key_out_ms,n_elements, context.d_histogram, bucket_4);

  // === 5:
  BMS_prescan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_5);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 64 * num_sub_prob_6bit);
  BMS_postscan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(d_key_out_ms, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_5);

  // === 6:
  BMS_prescan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_6);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 64 * num_sub_prob_6bit);
  BMS_postscan_64bucket_256<num_roll_6bit><<<num_sub_prob_6bit, 32*num_warps_6bit>>>(context.d_key_temp_ms, d_key_out_ms,n_elements, context.d_histogram, bucket_6);   
}

void multisplit_sort_8bit(uint32_t* d_key_in, uint32_t* d_key_out_ms, uint32_t n_elements, ms_sort_context& context)
{
  const uint32_t num_warps = NUM_WARPS_SORT_8BIT;
  const uint32_t num_roll = NUM_ROLLS_SORT_8BIT;
  const uint32_t num_sub_prob = (n_elements + (32 * num_warps * num_roll) - 1) / (32 * num_warps * num_roll);

  ms_bucket_sort_8bit<0,uint32_t> bucket_1;
  ms_bucket_sort_8bit<8,uint32_t> bucket_2;
  ms_bucket_sort_8bit<16,uint32_t> bucket_3;
  ms_bucket_sort_8bit<24,uint32_t> bucket_4;

  // === 1:
  BMS_prescan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(d_key_in, n_elements, context.d_histogram, bucket_1);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 256 * num_sub_prob);
  BMS_postscan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(d_key_in, context.d_key_temp_ms, n_elements, context.d_histogram, bucket_1);

  // === 2:
  BMS_prescan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_2);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 256 * num_sub_prob);
  BMS_postscan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_2);

  // === 3:
  BMS_prescan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, n_elements, context.d_histogram, bucket_3);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 256 * num_sub_prob);
  BMS_postscan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(d_key_out_ms, context.d_key_temp_ms,n_elements, context.d_histogram, bucket_3);

  // === 4:
  BMS_prescan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, n_elements, context.d_histogram, bucket_4);
  cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 256 * num_sub_prob);
  BMS_postscan_256bucket_256<num_roll><<<num_sub_prob, 32*num_warps>>>(context.d_key_temp_ms, d_key_out_ms, n_elements, context.d_histogram, bucket_4);   
}
// adding files related to key-value sorts
#include "api/multisplit_sort_pairs.cuh"
#endif