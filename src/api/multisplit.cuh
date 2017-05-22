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

#ifndef MULTISPLIT_CUH__
#define MULTISPLIT_CUH__
#include <stdint.h>
#include <cub/cub.cuh>
#include "config/config_bms.h"
#include "api/bms_api.h"
#include "kernels/bms/bms_prescan.cuh"
#include "kernels/bms/bms_postscan.cuh"
#include "kernels/bms/bms_postscan_pairs.cuh"

class multisplit_context{
public: 
	uint32_t 		size_sub_prob;
	uint32_t 		num_sub_prob;
	uint32_t 		num_blocks;
	uint32_t 		num_buckets;
	void*				d_temp_storage;
	size_t 			temp_storage_bytes;
	uint32_t* 	d_histogram;


	multisplit_context(uint32_t n_buckets):num_buckets(n_buckets){

		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		d_histogram = NULL;
	}
	~multisplit_context(){}
};

// ==== Memory allocations for multisplit (key-only or key-value)
void multisplit_allocate_key_only(uint32_t num_elements, multisplit_context& context){
	context.size_sub_prob = subproblem_size_bms_key_only(context.num_buckets);
	context.num_sub_prob = (num_elements + context.size_sub_prob - 1)/(context.size_sub_prob);
	context.num_blocks = (num_elements + context.size_sub_prob - 1)/(context.size_sub_prob);	

	// for histogram results per subproblem:
	cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * context.num_buckets * context.num_sub_prob);

	// for CUB's scan:
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, context.num_buckets * context.num_sub_prob);
	cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);
}

void multisplit_allocate_key_value(uint32_t num_elements, multisplit_context& context){
	context.size_sub_prob = subproblem_size_bms_key_value(context.num_buckets);
	context.num_sub_prob = (num_elements + context.size_sub_prob - 1)/(context.size_sub_prob);
	context.num_blocks = (num_elements + context.size_sub_prob - 1)/(context.size_sub_prob);	

	// for histogram results per subproblem:
	cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t) * context.num_buckets * context.num_sub_prob);

	// for CUB's scan:
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, context.num_buckets * context.num_sub_prob);
	cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);
}

void multisplit_release_memory(multisplit_context& context){
	if(context.d_histogram) cudaFree(context.d_histogram);
	if(context.d_temp_storage) cudaFree(context.d_temp_storage);
} 

template<typename KeyT, typename bucket_t>
void multisplit_key_only(
	KeyT* 								d_key_in, 
	KeyT* 								d_key_out, 
	uint32_t 							num_elements, 
	multisplit_context& 	context, 
	bucket_t 							bucket_identifier, 
	uint32_t* 						multisplit_offset = NULL)
{
	// ===== pre-scan stage:
	switch(context.num_buckets){
		case 256:
			if(NUM_WARPS_K_8 == 8)
				BMS_prescan_256bucket_256<NUM_ROLLS_K_8><<<context.num_blocks, 32*NUM_WARPS_K_8>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);				
		break;
		case 128:
			if(NUM_WARPS_K_7 == 8){
				BMS_prescan_128bucket_256<NUM_ROLLS_K_7><<<context.num_blocks, 32*NUM_WARPS_K_7>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);									
			}
		break;
		case 64:
			if(NUM_WARPS_K_6 == 8)
				BMS_prescan_64bucket_256<NUM_ROLLS_K_6><<<context.num_blocks, 32*NUM_WARPS_K_6>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
		break;
		case 32:
			if(NUM_WARPS_K_5 == 4)
				BMS_prescan_128<NUM_ROLLS_K_5, 32 , 5><<<context.num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
			else if(NUM_WARPS_K_5 == 8)
				BMS_prescan_256<NUM_ROLLS_K_5, 32, 5><<<context.num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);				
			break;
		case 16:
			if(NUM_WARPS_K_4 == 4)
				BMS_prescan_128<NUM_ROLLS_K_4, 16 , 4><<<context.num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
			else if(NUM_WARPS_K_4 == 8)
				BMS_prescan_256<NUM_ROLLS_K_4, 16, 4><<<context.num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
		break;
		case 8:
			if(NUM_WARPS_K_3 == 4)
				BMS_prescan_128<NUM_ROLLS_K_3, 8 , 3><<<context.num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
			else if(NUM_WARPS_K_3 == 8)
				BMS_prescan_256<NUM_ROLLS_K_3, 8, 3><<<context.num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
		break;				
		case 4:
			if(NUM_WARPS_K_2 == 4)
				BMS_prescan_128<NUM_ROLLS_K_2, 4 , 2><<<context.num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
			else if(NUM_WARPS_K_2 == 8)
				BMS_prescan_256<NUM_ROLLS_K_2, 4, 2><<<context.num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
		break;
		case 2:
			if(NUM_WARPS_K_1 == 4)
				BMS_prescan_128<NUM_ROLLS_K_1, 2 , 1><<<context.num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);					
			else if(NUM_WARPS_K_1 == 8)
				BMS_prescan_256<NUM_ROLLS_K_1, 2, 1><<<context.num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);									
		break;
		default:
			printf("Error: not a correct number of buckets.\n");
		break;
	}

	// ==== scan stage:
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, context.num_buckets * context.num_sub_prob);	

	// === storing offsets if required:
	if(multisplit_offset){
		multisplit_offset[0] = 0;
		for(int i = 1; i<context.num_buckets; i++){
			cudaMemcpy(&multisplit_offset[i], context.d_histogram + i * context.num_sub_prob, sizeof(uint32_t), cudaMemcpyDeviceToHost);	
		}
	}

	// ==== post-scan stage:
	switch(context.num_buckets){
		case 256:
			if(NUM_WARPS_K_8 == 8)
				BMS_postscan_256bucket_256<NUM_ROLLS_K_8><<<context.num_blocks, 32*NUM_WARPS_K_8>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);										
		break;
		case 128:
			if(NUM_WARPS_K_7 == 8){
				BMS_postscan_128bucket_256<NUM_ROLLS_K_7><<<context.num_blocks, 32*NUM_WARPS_K_7>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);						
			}
		break;
		case 64:
		if(NUM_WARPS_K_6 == 8)
				BMS_postscan_64bucket_256<NUM_ROLLS_K_6><<<context.num_blocks, 32*NUM_WARPS_K_6>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);									
		break;
		case 32:
			if(NUM_WARPS_K_5 == 4)
				BMS_postscan_128<NUM_ROLLS_K_5, 32, 5><<<context.num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_K_5 == 8)
				BMS_postscan_256<NUM_ROLLS_K_5, 32, 5><<<context.num_blocks, 32*NUM_WARPS_K_5>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);				
		break;
		case 16:
			if(NUM_WARPS_K_4 == 4)
				BMS_postscan_128<NUM_ROLLS_K_4, 16, 4><<<context.num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_K_4 == 8)
				BMS_postscan_256<NUM_ROLLS_K_4, 16, 4><<<context.num_blocks, 32*NUM_WARPS_K_4>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);			
		break;
		case 8:
			if(NUM_WARPS_K_3 == 4)
				BMS_postscan_128<NUM_ROLLS_K_3, 8, 3><<<context.num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_K_3 == 8)
				BMS_postscan_256<NUM_ROLLS_K_3, 8, 3><<<context.num_blocks, 32*NUM_WARPS_K_3>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);			
		break;						
		case 4:
			if(NUM_WARPS_K_2 == 4)
				BMS_postscan_128<NUM_ROLLS_K_2, 4, 2><<<context.num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_K_2 == 8)
				BMS_postscan_256<NUM_ROLLS_K_2, 4, 2><<<context.num_blocks, 32*NUM_WARPS_K_2>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);					
		break;	
		case 2:
			if(NUM_WARPS_K_1 == 4)
				BMS_postscan_128<NUM_ROLLS_K_1, 2, 1><<<context.num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_K_1 == 8)
				BMS_postscan_256<NUM_ROLLS_K_1, 2, 1><<<context.num_blocks, 32*NUM_WARPS_K_1>>>(d_key_in, d_key_out, num_elements, context.d_histogram, bucket_identifier);					
		break;
	}
}
//================================================================
template<typename KeyT, typename ValueT, typename bucket_t>
void multisplit_key_value(
	KeyT* 							d_key_in, 
	ValueT* 						d_value_in, 
	KeyT* 							d_key_out, 
	ValueT* 						d_value_out, 
	uint32_t 						num_elements, 
	multisplit_context& context, 
	bucket_t 						bucket_identifier, 
	uint32_t* 					multisplit_offset = NULL)
{

	// pre-scan stage:
	switch(context.num_buckets){
		case 256:
			if(NUM_WARPS_KV_8 == 8)
				BMS_prescan_256bucket_256<NUM_ROLLS_KV_8><<<context.num_blocks, 32*NUM_WARPS_KV_8>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);				
		break;
		case 128:
			if(NUM_WARPS_KV_7 == 8)
				BMS_prescan_128bucket_256<NUM_ROLLS_KV_7><<<context.num_blocks, 32*NUM_WARPS_KV_7>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);				
		break;
		case 64:
			if(NUM_WARPS_KV_6 == 8)
				BMS_prescan_64bucket_256<NUM_ROLLS_KV_6><<<context.num_blocks, 32*NUM_WARPS_KV_6>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
		break;				
		case 32:
			if(NUM_WARPS_KV_5 == 4)
				BMS_prescan_128<NUM_ROLLS_KV_5, 32, 5><<<context.num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_5 == 8)
				BMS_prescan_256<NUM_ROLLS_KV_5, 32, 5><<<context.num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
		break;
		case 16:
			if(NUM_WARPS_KV_4 == 4)
				BMS_prescan_128<NUM_ROLLS_KV_4, 16, 4><<<context.num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_4 == 8)
				BMS_prescan_256<NUM_ROLLS_KV_4, 16, 4><<<context.num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
		break;
		case 8:
			if(NUM_WARPS_KV_3 == 4)
				BMS_prescan_128<NUM_ROLLS_KV_3, 8, 3><<<context.num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_3 == 8)
				BMS_prescan_256<NUM_ROLLS_KV_3, 8, 3><<<context.num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
		break;				
		case 4:
			if(NUM_WARPS_KV_2 == 4)
				BMS_prescan_128<NUM_ROLLS_KV_2, 4, 2><<<context.num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_2 == 8)
				BMS_prescan_256<NUM_ROLLS_KV_2, 4, 2><<<context.num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);				
		break;
		case 2:
			if(NUM_WARPS_KV_1 == 4)
				BMS_prescan_128<NUM_ROLLS_KV_1, 2, 1><<<context.num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_1 == 8)
				BMS_prescan_256<NUM_ROLLS_KV_1, 2, 1><<<context.num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, num_elements, context.d_histogram, bucket_identifier);				
		break;				
	}

	// scan stage:
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, context.num_buckets * context.num_sub_prob);		

	// === storing offsets if required:
	if(multisplit_offset){
		multisplit_offset[0] = 0;
		for(int i = 1; i<context.num_buckets; i++){
			cudaMemcpy(&multisplit_offset[i], context.d_histogram + i * context.num_sub_prob, sizeof(uint32_t), cudaMemcpyDeviceToHost);	
		}
	}

	// post-scan stage:
	switch(context.num_buckets){
		case 256:
			if(NUM_WARPS_KV_8 == 8)
				BMS_postscan_256bucket_256_pairs<NUM_ROLLS_KV_8><<<context.num_blocks, 32*NUM_WARPS_KV_8>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);
		break;
		case 128:
			if(NUM_WARPS_KV_7 == 8)
				BMS_postscan_128bucket_256_pairs<NUM_ROLLS_KV_7><<<context.num_blocks, 32*NUM_WARPS_KV_7>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);	
		break;
		case 64:
			if(NUM_WARPS_KV_6 == 8)
				BMS_postscan_64bucket_256_pairs<NUM_ROLLS_KV_6><<<context.num_blocks, 32*NUM_WARPS_KV_6>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);	
		break;				
		case 32:
			if(NUM_WARPS_KV_5 == 4)
				BMS_postscan_128_pairs<NUM_ROLLS_KV_5, 32, 5><<<context.num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_5 == 8 && NUM_ROLLS_KV_5 <= 4)
				BMS_postscan_256_pairs<NUM_ROLLS_KV_5, 32, 5><<<context.num_blocks, 32*NUM_WARPS_KV_5>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);			
		break;
		case 16:
			if(NUM_WARPS_KV_4 == 4)
				BMS_postscan_128_pairs<NUM_ROLLS_KV_4, 16, 4><<<context.num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_4 == 8 && NUM_ROLLS_KV_4 <= 4)
				BMS_postscan_256_pairs<NUM_ROLLS_KV_4, 16, 4><<<context.num_blocks, 32*NUM_WARPS_KV_4>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);			
		break;
		case 8:
			if(NUM_WARPS_KV_3 == 4)
				BMS_postscan_128_pairs<NUM_ROLLS_KV_3, 8, 3><<<context.num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_3 == 8 && NUM_ROLLS_KV_3 <= 4)
				BMS_postscan_256_pairs<NUM_ROLLS_KV_3, 8, 3><<<context.num_blocks, 32*NUM_WARPS_KV_3>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);
		break;				
		case 4:
			if(NUM_WARPS_KV_2 == 4)
				BMS_postscan_128_pairs<NUM_ROLLS_KV_2, 4, 2><<<context.num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_2 == 8 && NUM_ROLLS_KV_2 <= 4)
				BMS_postscan_256_pairs<NUM_ROLLS_KV_2, 4, 2><<<context.num_blocks, 32*NUM_WARPS_KV_2>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);			
		break;				
		case 2:
			if(NUM_WARPS_KV_1 == 4)
				BMS_postscan_128_pairs<NUM_ROLLS_KV_1, 2, 1><<<context.num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);
			else if(NUM_WARPS_KV_1 == 8 && NUM_ROLLS_KV_1 <= 4)
				BMS_postscan_256_pairs<NUM_ROLLS_KV_1, 2, 1><<<context.num_blocks, 32*NUM_WARPS_KV_1>>>(d_key_in, d_value_in, d_key_out, d_value_out, num_elements, context.d_histogram, bucket_identifier);			
		break;				
	}	
}
#endif 