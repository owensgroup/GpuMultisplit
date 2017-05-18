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

#ifndef __MULTISPLIT2_COMPACTION_CUH_
#define __MULTISPLIT2_COMPACTION_CUH_
#include <cub/cub.cuh>
#include <stdint.h>

#define COMPACTION_MULTISPLIT2_NUM_WARPS 8
#define COMPACTION_MULTISPLIT2_NUM_ROLLS 4
#define COMPACTION_MULTISPLIT2_NUM_TILES 3

#define COMPACTION_MULTISPLIT2_NUM_WARPS_PAIRS 8
#define COMPACTION_MULTISPLIT2_NUM_ROLLS_PAIRS 4
#define COMPACTION_MULTISPLIT2_NUM_TILES_PAIRS 1

template<
	uint32_t 		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	typename 		bucket_t,
	typename 		KeyT>
	__launch_bounds__(256)
__global__ void 
multisplit2_compaction_prescan_protected(
	KeyT* 	d_key_in,
	uint32_t 		num_elements,
	uint32_t* 	d_histogram, 
	bucket_t 		bucket_identifier)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;
	uint32_t binCounter = 0;

	if(blockIdx.x == (gridDim.x - 1)) // last block
	{
		for(int i_tile = 0; i_tile < NUM_TILES_; i_tile++)
		{
			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
					(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
					(i_tile * NUM_ROLLS_) + \
					i_roll) << 5) + \
					laneId;

				KeyT input_key = (temp_address < num_elements)?(d_key_in[temp_address]):0xFFFFFFFF;

				// it's safe if we put the invalid element into the last bucket
				uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key):1u;

				uint32_t 	rx_buffer = __ballot((myBucket) & 0x01);
				uint32_t 	myHisto = (((laneId) & 0x01)?rx_buffer:(~rx_buffer));

				binCounter  += __popc(myHisto);
			}
		}
	}
	else{ // all other blocks
		for(int i_tile = 0; i_tile < NUM_TILES_; i_tile++)
		{
			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				KeyT input_key = (d_key_in[\
					(blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
					(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
					(i_tile * NUM_ROLLS_) + \
					i_roll) << 5) + \
					laneId]);

				uint32_t myBucket = bucket_identifier(input_key);

				uint32_t rx_buffer = __ballot((myBucket) & 0x01);
				uint32_t myHisto = (((laneId) & 0x01)?rx_buffer:(~rx_buffer));

				binCounter  += __popc(myHisto);
			}
		}
	}
	// writing back results per warp into gmem:
	if(laneId < 2)
	{
		d_histogram[(laneId * (blockDim.x >> 5) * gridDim.x) + ((blockDim.x >> 5) * blockIdx.x) + warpId] = binCounter;
	}
}

template<
	uint32_t 		NUM_WARPS_,
	uint32_t		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	typename 		bucket_t,
	typename		KeyT>
__global__ void 
multisplit2_compaction_postscan_4rolls_protected(
	const KeyT* 	__restrict__ d_key_in,
	KeyT* 	__restrict__ d_key_out,
	uint32_t 		num_elements,
	const uint32_t* 	__restrict__ d_histogram, 
	bucket_t 		bucket_identifier)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ KeyT smem[(32 * NUM_WARPS_ * NUM_ROLLS_)];
	KeyT *keys_ms_smem = 						smem;

	KeyT input_key[NUM_ROLLS_]; 				// stores all keys regarding to this thread

	// ==== Loading back histogram results from gmem into smem:
	uint32_t global_offset = 0;
	if(laneId < 2) // warp -> bucket (uncoalesced gmem access)
		global_offset = __ldg(&d_histogram[laneId * gridDim.x * NUM_WARPS_ + blockIdx.x * NUM_WARPS_ + warpId]);

	if(blockIdx.x == (gridDim.x - 1))// last block ===============================================
	{
	for(int i_tile = 0; i_tile < NUM_TILES_; i_tile++)
		{
			uint32_t binCounter = 0;							// total count of elements within its in-charge bucket
			uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
			uint32_t myBucket_list = 0;					// each byte contains bucket index per roll
			uint32_t Histogram_list = 0;									

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
					(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
					(i_tile * NUM_ROLLS_) + \
					i_roll) << 5) + \
					laneId;
				input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;

				uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(1u);

				myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

				uint32_t rx_buffer = __ballot((myBucket) & 0x01);
				uint32_t myHisto = (((laneId) & 0x01)?rx_buffer:(~rx_buffer));
				uint32_t myIndexBmp = (((myBucket) & 0x01)?rx_buffer:(~rx_buffer));

				myHisto = __popc(myHisto);
				binCounter  += myHisto;

				Histogram_list |= (myHisto << (i_roll << 3));
				// each byte contains local index per roll
				myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));			
			}
			// Byte-wide exlusive scan over Histogram_list: (for each roll)
			Histogram_list = (Histogram_list << 8) + (Histogram_list << 16) + (Histogram_list << 24);		

			// inclusive scan on binCounters:
			uint32_t scanned_binCount = binCounter;
			uint32_t temp_scan = __shfl_up(scanned_binCount, 1, 32);
			if(laneId >= 1) scanned_binCount += temp_scan;

			// making it exclusive
			scanned_binCount -= binCounter;

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{	
				uint32_t temp_scanned_roll_histo = __shfl(Histogram_list, (myBucket_list & 0xFF));// tile-wide offset: same bucket
				uint32_t myNewBlockIndex = ((temp_scanned_roll_histo >> (i_roll << 3)) & 0xFF);

				myNewBlockIndex += (myLocalIndex_list & 0xFF); // among my current roll

				myNewBlockIndex += __shfl(scanned_binCount, (myBucket_list & 0xFF)); // tile-wide offset:previous buckets
				myNewBlockIndex += ((warpId * NUM_ROLLS_) << 5);
				myBucket_list >>= 8;
				myLocalIndex_list >>= 8;

				keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
			}

			#pragma unroll
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				uint32_t temp_address_smem = laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5);
				KeyT temp_input = keys_ms_smem[temp_address_smem];
				uint32_t myBucket_temp = ((laneId + (i_roll<<5) + ((i_tile * NUM_ROLLS_) << 5) + ((warpId * NUM_TILES_ * NUM_ROLLS_)<<5) + blockIdx.x * ((NUM_WARPS_*NUM_ROLLS_*NUM_TILES_)<<5)) < num_elements)?bucket_identifier(temp_input):(1u);
				uint32_t myLocalIndex_temp = __shfl(global_offset, myBucket_temp);
				myLocalIndex_temp += (laneId + (i_roll << 5)); 										// local offset
				myLocalIndex_temp	-= __shfl(scanned_binCount, myBucket_temp, 32); 
				if(myLocalIndex_temp < num_elements) // protected writing
					d_key_out[myLocalIndex_temp] = temp_input;
			}
			// updating the global offsets for next tile
			global_offset += binCounter;
		}
	}//====================================================================================
	else{
		for(int i_tile = 0; i_tile < NUM_TILES_; i_tile++)
		{
			uint32_t binCounter = 0;							// total count of elements within its in-charge bucket
			uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
			uint32_t myBucket_list = 0;					// each byte contains bucket index per roll
			uint32_t Histogram_list = 0;									

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				input_key[i_roll] = __ldg(&d_key_in[\
					(blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
					(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
					(i_tile * NUM_ROLLS_) + \
					i_roll) << 5) + \
					laneId]);

				uint32_t myBucket = bucket_identifier(input_key[i_roll]);
				myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

				uint32_t rx_buffer = __ballot((myBucket) & 0x01);
				uint32_t myHisto = (((laneId) & 0x01)?rx_buffer:(~rx_buffer));
				uint32_t myIndexBmp = (((myBucket) & 0x01)?rx_buffer:(~rx_buffer));

				myHisto = __popc(myHisto);
				binCounter  += myHisto;

				Histogram_list |= (myHisto << (i_roll << 3));
				// each byte contains local index per roll
				myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));			
			}
			// Byte-wide exlusive scan over Histogram_list: (for each roll)
			Histogram_list = (Histogram_list << 8) + (Histogram_list << 16) + (Histogram_list << 24);		

			// inclusive scan on binCounters:
			uint32_t scanned_binCount = binCounter;
			uint32_t temp_scan = __shfl_up(scanned_binCount, 1, 32);
			if(laneId >= 1) scanned_binCount += temp_scan;

			// making it exclusive
			scanned_binCount -= binCounter;

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{	
				uint32_t temp_scanned_roll_histo = __shfl(Histogram_list, (myBucket_list & 0xFF));// tile-wide offset: same bucket
				uint32_t myNewBlockIndex = ((temp_scanned_roll_histo >> (i_roll << 3)) & 0xFF);

				myNewBlockIndex += (myLocalIndex_list & 0xFF); // among my current roll

				myNewBlockIndex += __shfl(scanned_binCount, (myBucket_list & 0xFF)); // tile-wide offset:previous buckets
				myNewBlockIndex += ((warpId * NUM_ROLLS_) << 5);
				myBucket_list >>= 8;
				myLocalIndex_list >>= 8;

				keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
			}

			#pragma unroll
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				KeyT temp_input = keys_ms_smem[laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5)];
				uint32_t myBucket_temp = bucket_identifier(temp_input);
				uint32_t myLocalIndex_temp = __shfl(global_offset, myBucket_temp);
				myLocalIndex_temp += (laneId + (i_roll << 5)); 										// local offset
				myLocalIndex_temp	-= __shfl(scanned_binCount, myBucket_temp, 32); 
				d_key_out[myLocalIndex_temp] = temp_input;
			}
			// updating the global offsets for next tile
			global_offset += binCounter;
		}
	}
}

template<
	uint32_t 		NUM_WARPS_,
	uint32_t		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	typename 		bucket_t,
	typename		KeyT,
	typename 		ValueT>
__global__ void 
multisplit2_compaction_postscan_4rolls_pairs_protected(
	const KeyT* 	__restrict__ d_key_in,
	const ValueT* 	__restrict__ d_value_in,
	KeyT* 	__restrict__ d_key_out,
	ValueT* 	__restrict__ d_value_out,
	uint32_t 		num_elements,
	const uint32_t* 	__restrict__ d_histogram, 
	bucket_t 		bucket_identifier)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ KeyT smem[(64 * NUM_WARPS_ * NUM_ROLLS_)];
	KeyT *keys_ms_smem = 						smem;
	ValueT *values_ms_smem = 					&smem[32 * NUM_WARPS_ * NUM_ROLLS_];

	KeyT input_key[NUM_ROLLS_]; 				// stores all keys regarding to this thread
	ValueT input_value[NUM_ROLLS_]; 				// stores all values regarding to this thread
	// ==== Loading back histogram results from gmem into smem:
	uint32_t global_offset = 0;
	if(laneId < 2) // warp -> bucket (uncoalesced gmem access)
		global_offset = __ldg(&d_histogram[laneId * gridDim.x * NUM_WARPS_ + blockIdx.x * NUM_WARPS_ + warpId]);

	if(blockIdx.x == (gridDim.x - 1)) // for last block: =========================================
	{
		for(int i_tile = 0; i_tile < NUM_TILES_; i_tile++)
		{
			uint32_t binCounter = 0;							// total count of elements within its in-charge bucket
			uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
			uint32_t myBucket_list = 0;					// each byte contains bucket index per roll
			uint32_t Histogram_list = 0;									

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
					(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
					(i_tile * NUM_ROLLS_) + \
					i_roll) << 5) + \
					laneId;

				input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
				input_value[i_roll] = (temp_address < num_elements)?__ldg(&d_value_in[temp_address]):0xFFFFFFFF;
				uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(1u);

				myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

				uint32_t rx_buffer = __ballot((myBucket) & 0x01);
				uint32_t myHisto = (((laneId) & 0x01)?rx_buffer:(~rx_buffer));
				uint32_t myIndexBmp = (((myBucket) & 0x01)?rx_buffer:(~rx_buffer));

				myHisto = __popc(myHisto);
				binCounter  += myHisto;

				Histogram_list |= (myHisto << (i_roll << 3));
				// each byte contains local index per roll
				myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));			
			}
			// Byte-wide exlusive scan over Histogram_list: (for each roll)
			Histogram_list = (Histogram_list << 8) + (Histogram_list << 16) + (Histogram_list << 24);		

			// inclusive scan on binCounters:
			uint32_t scanned_binCount = binCounter;
			uint32_t temp_scan = __shfl_up(scanned_binCount, 1, 32);
			if(laneId >= 1) scanned_binCount += temp_scan;

			// making it exclusive
			scanned_binCount -= binCounter;

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{	
				uint32_t temp_scanned_roll_histo = __shfl(Histogram_list, (myBucket_list & 0xFF));// tile-wide offset: same bucket
				uint32_t myNewBlockIndex = ((temp_scanned_roll_histo >> (i_roll << 3)) & 0xFF);

				myNewBlockIndex += (myLocalIndex_list & 0xFF); // among my current roll

				myNewBlockIndex += __shfl(scanned_binCount, (myBucket_list & 0xFF)); // tile-wide offset:previous buckets
				myNewBlockIndex += ((warpId * NUM_ROLLS_) << 5);
				myBucket_list >>= 8;
				myLocalIndex_list >>= 8;

				keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
				values_ms_smem[myNewBlockIndex] = input_value[i_roll];
			}

			#pragma unroll
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				KeyT temp_key = keys_ms_smem[laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5)];
				ValueT temp_value = values_ms_smem[laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5)];
				uint32_t myBucket_temp = ((laneId + (i_roll<<5) + ((i_tile * NUM_ROLLS_) << 5) + ((warpId * NUM_TILES_ * NUM_ROLLS_)<<5) + blockIdx.x * ((NUM_WARPS_*NUM_ROLLS_*NUM_TILES_)<<5)) < num_elements)?bucket_identifier(temp_key):(1u);				
				uint32_t myLocalIndex_temp = __shfl(global_offset, myBucket_temp);
				myLocalIndex_temp += (laneId + (i_roll << 5)); 										// local offset
				myLocalIndex_temp	-= __shfl(scanned_binCount, myBucket_temp, 32); 
				if(myLocalIndex_temp < num_elements){
					d_key_out[myLocalIndex_temp] = temp_key;
					d_value_out[myLocalIndex_temp] = temp_value;
				}
			}
			// updating the global offsets for next tile
			global_offset += binCounter;
		}
	}
	else{ // ==========================================================================
		for(int i_tile = 0; i_tile < NUM_TILES_; i_tile++)
		{
			uint32_t binCounter = 0;							// total count of elements within its in-charge bucket
			uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
			uint32_t myBucket_list = 0;					// each byte contains bucket index per roll
			uint32_t Histogram_list = 0;									

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
					(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
					(i_tile * NUM_ROLLS_) + \
					i_roll) << 5) + \
					laneId;

				input_key[i_roll] = __ldg(&d_key_in[temp_address]);
				input_value[i_roll] = __ldg(&d_value_in[temp_address]);

				uint32_t myBucket = bucket_identifier(input_key[i_roll]);
				myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

				uint32_t rx_buffer = __ballot((myBucket) & 0x01);
				uint32_t myHisto = (((laneId) & 0x01)?rx_buffer:(~rx_buffer));
				uint32_t myIndexBmp = (((myBucket) & 0x01)?rx_buffer:(~rx_buffer));

				myHisto = __popc(myHisto);
				binCounter  += myHisto;

				Histogram_list |= (myHisto << (i_roll << 3));
				// each byte contains local index per roll
				myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));			
			}
			// Byte-wide exlusive scan over Histogram_list: (for each roll)
			Histogram_list = (Histogram_list << 8) + (Histogram_list << 16) + (Histogram_list << 24);		

			// inclusive scan on binCounters:
			uint32_t scanned_binCount = binCounter;
			uint32_t temp_scan = __shfl_up(scanned_binCount, 1, 32);
			if(laneId >= 1) scanned_binCount += temp_scan;

			// making it exclusive
			scanned_binCount -= binCounter;

			#pragma unroll 
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{	
				uint32_t temp_scanned_roll_histo = __shfl(Histogram_list, (myBucket_list & 0xFF));// tile-wide offset: same bucket
				uint32_t myNewBlockIndex = ((temp_scanned_roll_histo >> (i_roll << 3)) & 0xFF);

				myNewBlockIndex += (myLocalIndex_list & 0xFF); // among my current roll

				myNewBlockIndex += __shfl(scanned_binCount, (myBucket_list & 0xFF)); // tile-wide offset:previous buckets
				myNewBlockIndex += ((warpId * NUM_ROLLS_) << 5);
				myBucket_list >>= 8;
				myLocalIndex_list >>= 8;

				keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
				values_ms_smem[myNewBlockIndex] = input_value[i_roll];
			}

			#pragma unroll
			for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
			{
				KeyT temp_key = keys_ms_smem[laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5)];
				ValueT temp_value = values_ms_smem[laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5)];
				uint32_t myBucket_temp = bucket_identifier(temp_key);
				uint32_t myLocalIndex_temp = __shfl(global_offset, myBucket_temp);
				myLocalIndex_temp += (laneId + (i_roll << 5)); 										// local offset
				myLocalIndex_temp	-= __shfl(scanned_binCount, myBucket_temp, 32); 
				d_key_out[myLocalIndex_temp] = temp_key;
				d_value_out[myLocalIndex_temp] = temp_value;
			}
			// updating the global offsets for next tile
			global_offset += binCounter;
		}
	}
}

//=====================================================
// external wrapper for calling our kernels:
//=====================================================
class compaction_context{
public:
	void     	*d_temp_storage;
	size_t   	temp_storage_bytes;
	uint32_t 		*d_histogram;
	compaction_context()
	{
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		d_histogram = NULL;
	}
	~compaction_context(){}
};

void compaction_allocate_key_only(uint32_t numElements, compaction_context& context)
{
	uint32_t size_sub_prob = 32 * COMPACTION_MULTISPLIT2_NUM_ROLLS * COMPACTION_MULTISPLIT2_NUM_TILES;
	uint32_t size_block = size_sub_prob * COMPACTION_MULTISPLIT2_NUM_WARPS;
	uint32_t num_sub_prob_per_block = size_block/size_sub_prob;
	uint32_t num_sub_prob = (numElements + size_sub_prob - 1)/(size_sub_prob);
	num_sub_prob = (num_sub_prob + num_sub_prob_per_block - 1)/num_sub_prob_per_block*num_sub_prob_per_block;

	cudaMalloc((void**)&context.d_histogram, sizeof(uint32_t)*2*num_sub_prob);
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 2 * num_sub_prob);

	cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);
}

void compaction_allocate_key_value(uint32_t numElements, compaction_context& context)
{
	uint32_t size_sub_prob = 32 * COMPACTION_MULTISPLIT2_NUM_ROLLS_PAIRS * COMPACTION_MULTISPLIT2_NUM_TILES_PAIRS;
	uint32_t size_block = size_sub_prob * COMPACTION_MULTISPLIT2_NUM_WARPS_PAIRS;
	uint32_t num_sub_prob_per_block = size_block/size_sub_prob;
	uint32_t num_sub_prob = (numElements + size_sub_prob - 1)/(size_sub_prob);
	num_sub_prob = (num_sub_prob + num_sub_prob_per_block - 1)/num_sub_prob_per_block*num_sub_prob_per_block;

	cudaMalloc((void**)&context.d_histogram, sizeof(unsigned int)*2*num_sub_prob);
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 2 * num_sub_prob);

	cudaMalloc((void**)&context.d_temp_storage, context.temp_storage_bytes);
}

void compaction_release_memory(compaction_context& context)
{
	cudaFree(context.d_histogram);
	cudaFree(context.d_temp_storage);
}

template<typename key_t, typename bucket_t>
unsigned int compaction_key_only(key_t* d_key_in, key_t* d_key_out, uint32_t numElements, compaction_context& context, bucket_t bucket_identifier)
{
	uint32_t size_sub_prob = 32 * COMPACTION_MULTISPLIT2_NUM_ROLLS * COMPACTION_MULTISPLIT2_NUM_TILES;
	uint32_t size_block = size_sub_prob * COMPACTION_MULTISPLIT2_NUM_WARPS;
	uint32_t num_sub_prob_per_block = size_block/size_sub_prob;
	uint32_t num_sub_prob = (numElements + size_sub_prob - 1)/(size_sub_prob);
	num_sub_prob = (num_sub_prob + num_sub_prob_per_block - 1)/num_sub_prob_per_block*num_sub_prob_per_block;
	uint32_t num_blocks = (numElements + size_block - 1)/size_block;
	
	// pre-scan stage:
	multisplit2_compaction_prescan_protected<COMPACTION_MULTISPLIT2_NUM_TILES, COMPACTION_MULTISPLIT2_NUM_ROLLS><<<num_blocks, 32*COMPACTION_MULTISPLIT2_NUM_WARPS>>>(d_key_in, numElements, context.d_histogram, bucket_identifier);	

	// scan stage:
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 2 * num_sub_prob);	

	uint32_t n_bucket_zero = 0;
	cudaMemcpy(&n_bucket_zero, context.d_histogram + num_sub_prob, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// post-scan stage:
	multisplit2_compaction_postscan_4rolls_protected<COMPACTION_MULTISPLIT2_NUM_WARPS, COMPACTION_MULTISPLIT2_NUM_TILES, COMPACTION_MULTISPLIT2_NUM_ROLLS><<<num_blocks, 32*COMPACTION_MULTISPLIT2_NUM_WARPS>>>(d_key_in, d_key_out, numElements, context.d_histogram, bucket_identifier);

	return n_bucket_zero;
}

template<typename key_t, typename value_t, typename bucket_t>
unsigned int compaction_key_value(key_t* d_key_in, value_t* d_value_in, key_t* d_key_out, value_t* d_value_out, uint32_t numElements, compaction_context& context, bucket_t bucket_identifier)
{
	uint32_t size_sub_prob = 32 * COMPACTION_MULTISPLIT2_NUM_ROLLS_PAIRS * COMPACTION_MULTISPLIT2_NUM_TILES_PAIRS;
	uint32_t size_block = size_sub_prob * COMPACTION_MULTISPLIT2_NUM_WARPS_PAIRS;
	uint32_t num_sub_prob_per_block = size_block/size_sub_prob;
	uint32_t num_sub_prob = (numElements + size_sub_prob - 1)/(size_sub_prob);
	num_sub_prob = (num_sub_prob + num_sub_prob_per_block - 1)/num_sub_prob_per_block*num_sub_prob_per_block;
	uint32_t num_blocks = (numElements + size_block - 1)/size_block;
	
	// pre-scan stage:
	multisplit2_compaction_prescan_protected<COMPACTION_MULTISPLIT2_NUM_TILES_PAIRS, COMPACTION_MULTISPLIT2_NUM_ROLLS_PAIRS><<<num_blocks, 32*COMPACTION_MULTISPLIT2_NUM_WARPS_PAIRS>>>(d_key_in, numElements, context.d_histogram, bucket_identifier);	

	// scan stage:
	cub::DeviceScan::ExclusiveSum(context.d_temp_storage, context.temp_storage_bytes, context.d_histogram, context.d_histogram, 2 * num_sub_prob);	

	uint32_t n_bucket_zero = 0;
	cudaMemcpy(&n_bucket_zero, context.d_histogram + num_sub_prob, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// post-scan stage:
	multisplit2_compaction_postscan_4rolls_pairs_protected<COMPACTION_MULTISPLIT2_NUM_WARPS_PAIRS, COMPACTION_MULTISPLIT2_NUM_TILES_PAIRS, COMPACTION_MULTISPLIT2_NUM_ROLLS_PAIRS><<<num_blocks, 32*COMPACTION_MULTISPLIT2_NUM_WARPS_PAIRS>>>(d_key_in, d_value_in, d_key_out, d_value_out, numElements, context.d_histogram, bucket_identifier);

	return n_bucket_zero;
}
#endif
