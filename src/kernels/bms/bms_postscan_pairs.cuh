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

#ifndef MULTISPLIT_BMS_POSTSCAN_PAIRS__
#define MULTISPLIT_BMS_POSTSCAN_PAIRS__
template<
	uint32_t		NUM_ROLLS,
	uint32_t 		NUM_BUCKETS,
	uint32_t		LOG_BUCKETS,
	typename 		bucket_t,
	typename 		KeyT,
	typename 		ValueT> 
__global__ void 
BMS_postscan_256_pairs(
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

	__shared__ uint32_t smem[2*NUM_BUCKETS + (NUM_BUCKETS * NUM_WARPS_8) + (64 * NUM_WARPS_8 * NUM_ROLLS)
		];
	uint32_t *global_histogram_smem = 	smem; // first half: global, second half: within this block
	uint32_t *local_histogram_smem	=		&smem[2 * NUM_BUCKETS];
	uint32_t *keys_ms_smem = 						&smem[2 * NUM_BUCKETS + (NUM_BUCKETS * NUM_WARPS_8)];
	uint32_t *values_ms_smem = 						&keys_ms_smem[32*NUM_WARPS_8*NUM_ROLLS];

	KeyT input_key[NUM_ROLLS]; 				// stores all keys regarding to this thread
	ValueT input_value[NUM_ROLLS]; 				// stores all keys regarding to this thread
	uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
	uint32_t myBucket_list = 0;					// each byte contains bucket index per roll
	uint32_t Histogram_list = 0;									
	uint32_t binCounter = 0;							// total count of elements within its in-charge bucket

	// ==== Loading back histogram results from gmem into smem:
	if(warpId == 0 && laneId < NUM_BUCKETS)
		global_histogram_smem[laneId] = __ldg(&d_histogram[laneId * gridDim.x + blockIdx.x]);

	if(blockIdx.x == (gridDim.x - 1))
	{
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
			input_value[i_roll] = (temp_address < num_elements)?__ldg(&d_value_in[temp_address]):0xFFFFFFFF;
			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(NUM_BUCKETS-1);
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			myHisto = __popc(myHisto);
			binCounter += myHisto;
			Histogram_list |= (myHisto << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}
	else{
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = __ldg(&d_key_in[temp_address]);
			input_value[i_roll] = __ldg(&d_value_in[temp_address]);
			uint32_t myBucket = bucket_identifier(input_key[i_roll]);
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			myHisto = __popc(myHisto);
			binCounter += myHisto;
			Histogram_list |= (myHisto << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}

	// Byte-wide exlusive scan over Histogram_list:
	// we're sure that each byte is less than 32 (histogram of a warp) and hence no possible overflow
	Histogram_list = (Histogram_list << 8) + (Histogram_list << 16) + (Histogram_list << 24);

	// ==== storing the results back into smem:
	if(laneId < NUM_BUCKETS)
		local_histogram_smem[laneId * NUM_WARPS_8 + warpId] = binCounter;

	__syncthreads();

	// ==== Computing segmented scans per bucket + warp-wide scan over histograms
	if(threadIdx.x < (NUM_BUCKETS * NUM_WARPS_8))
	{
		uint32_t bucket_histo = local_histogram_smem[threadIdx.x];
		uint32_t reduction = bucket_histo;
		// == segmented inclusive scan:
		// if(NUM_WARPS_8 == 8)
		// {
			uint32_t temp_sum = __shfl_up(reduction, 1, 32);
			if((laneId & 0x07) >= 1) reduction += temp_sum;
			temp_sum = __shfl_up(reduction, 2, 32);
			if((laneId & 0x07) >= 2) reduction += temp_sum;
			temp_sum = __shfl_up(reduction, 4, 32);
			if((laneId & 0x07) >= 4) reduction += temp_sum;

			// writing back the results (exclusive scan):
			local_histogram_smem[threadIdx.x] = reduction - bucket_histo;
			// writing back the histogram results into smem: 
			if((laneId & 0x07) == 0x07)
				global_histogram_smem[NUM_BUCKETS + (warpId << 2) + (laneId >> 3)] = reduction;
		// }
	}
	__syncthreads();

	// ==== computing the final indices and performing multisplit in smem
	// each warp computing its own warp-wide histogram scan:
	uint32_t bucket_offset = global_histogram_smem[NUM_BUCKETS + laneId];
	uint32_t local_offset = bucket_offset;

	// we decide to run this part for every warp to avoid more __syncthreads
	// inclusive scan over histogram results of each bucket in the block
	#pragma unroll
	for(int i = 0; i<LOG_BUCKETS; i++)
	{
		uint32_t temp_scan = __shfl_up(local_offset, (1<<i), 32);
		if(laneId >= (1<<i)) local_offset += temp_scan;
	}
	local_offset -= bucket_offset; // making it exclusive
	// at this point, for laneId < NUM_BUCKETS: local_offset equals number of elements within smaller buckets 

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{	
		uint32_t temp_scanned_roll_histo = __shfl(Histogram_list, (myBucket_list & 0xFF));
		uint32_t myNewBlockIndex = ((temp_scanned_roll_histo >> (i_roll << 3)) & 0xFF) // among all roles with the same warp
				+ (myLocalIndex_list & 0xFF) // among my current roll
				+ local_histogram_smem[(myBucket_list & 0xFF) * NUM_WARPS_8 + warpId] // block-wide index (my own bucket)
				+ __shfl(local_offset, (myBucket_list & 0xFF), 32); // block-wide index (other buckets)
		
		myBucket_list >>= 8;
		myLocalIndex_list >>= 8;

		keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
		values_ms_smem[myNewBlockIndex] = input_value[i_roll];
	}
	__syncthreads();

	// ==== Final stage: Tranferring elements from smem into gmem:
	#pragma unroll 
	for(int kk = 0; kk<NUM_ROLLS; kk++)
	{
		uint32_t temp_index = threadIdx.x + kk * blockDim.x;
		input_key[0] = keys_ms_smem[temp_index];
		input_value[0] = values_ms_smem[temp_index];
		if(blockIdx.x == (gridDim.x - 1))
			myBucket_list = ((temp_index + NUM_ROLLS * blockDim.x * blockIdx.x) < num_elements)?bucket_identifier(input_key[0]):(NUM_BUCKETS - 1);
		else
			myBucket_list = bucket_identifier(input_key[0]);
		myLocalIndex_list = global_histogram_smem[myBucket_list] // global offset
							+ temp_index 										// local offset
							- __shfl(local_offset, myBucket_list, 32);
		if(myLocalIndex_list < num_elements){
			d_key_out[myLocalIndex_list] = input_key[0];
			d_value_out[myLocalIndex_list] = input_value[0];
		}
	}
}
//==========================================
template<
	uint32_t		NUM_ROLLS,
	uint32_t		NUM_BUCKETS,
	uint32_t		LOG_BUCKETS,
	typename 		bucket_t,
	typename 		KeyT,
	typename 		ValueT>
__global__ void 
BMS_postscan_128_pairs(
	const KeyT* 	__restrict__ d_key_in,
	const ValueT* 	__restrict__ d_value_in,
	KeyT* 	__restrict__ d_key_out,
	ValueT* 	__restrict__ d_value_out,
	uint32_t 		num_elements,
	const uint32_t* 	__restrict__ d_histogram, 
	bucket_t 		bucket_identifier)
{
	// Especially designed for >= 4 and <= 8 NUM_ROLLS 
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem[2*NUM_BUCKETS + (NUM_BUCKETS * NUM_WARPS_4) + (64 * NUM_WARPS_4 * NUM_ROLLS)
		];
	uint32_t *global_histogram_smem = 	smem; // first half: global, second half: within this block
	uint32_t *local_histogram_smem	=		&smem[2 * NUM_BUCKETS];
	uint32_t *keys_ms_smem = 						&smem[2 * NUM_BUCKETS + (NUM_BUCKETS * NUM_WARPS_4)];
	uint32_t *values_ms_smem = 					&keys_ms_smem[32*NUM_WARPS_4*NUM_ROLLS];

	KeyT input_key[NUM_ROLLS]; 				// stores all keys regarding to this thread
	ValueT input_value[NUM_ROLLS]; 				// stores all keys regarding to this thread
	uint32_t myLocalIndex_list_first = 0;
									// each byte contains localIndex per roll
	uint32_t myLocalIndex_list_second = 0;
	uint32_t myBucket_list_first = 0;					// each byte contains bucket index per roll
	uint32_t myBucket_list_second = 0;
	uint32_t Histogram_list_first = 0;									
	uint32_t Histogram_list_second = 0;									
	uint32_t binCounter = 0;							// total count of elements within its in-charge bucket

	// ==== Loading back histogram results from gmem into smem:
	if(warpId == 0 && laneId < NUM_BUCKETS)
		global_histogram_smem[laneId] = __ldg(&d_histogram[laneId * gridDim.x + blockIdx.x]);

	if(blockIdx.x == (gridDim.x - 1))
	{
	// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < 4; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
			input_value[i_roll] = (temp_address < num_elements)?__ldg(&d_value_in[temp_address]):0xFFFFFFFF;

			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(NUM_BUCKETS-1);
			myBucket_list_first |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			myHisto = __popc(myHisto);
			binCounter += myHisto;
			Histogram_list_first |= (myHisto << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list_first |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 4; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = 			(blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
			input_value[i_roll] = (temp_address < num_elements)?__ldg(&d_value_in[temp_address]):0xFFFFFFFF;

			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(NUM_BUCKETS-1);
			myBucket_list_second |= (myBucket << ((i_roll - 4) << 3)); // each byte myBucket per roll

			uint32_t myHisto = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			myHisto = __popc(myHisto);
			binCounter += myHisto;
			Histogram_list_second |= (myHisto << ((i_roll - 4) << 3));
			// each byte contains local index per roll
			myLocalIndex_list_second |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << ((i_roll - 4) << 3));
		}
	}
	else{
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < 4; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = __ldg(&d_key_in[temp_address]);
			input_value[i_roll] = __ldg(&d_value_in[temp_address]);

			uint32_t myBucket = bucket_identifier(input_key[i_roll]);
			myBucket_list_first |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			myHisto = __popc(myHisto);
			binCounter += myHisto;
			Histogram_list_first |= (myHisto << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list_first |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 4; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = 			(blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = __ldg(&d_key_in[temp_address]);
			input_value[i_roll] = __ldg(&d_value_in[temp_address]);

			uint32_t myBucket = bucket_identifier(input_key[i_roll]);
			myBucket_list_second |= (myBucket << ((i_roll - 4) << 3)); // each byte myBucket per roll

			uint32_t myHisto = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			myHisto = __popc(myHisto);
			binCounter += myHisto;
			Histogram_list_second |= (myHisto << ((i_roll - 4) << 3));
			// each byte contains local index per roll
			myLocalIndex_list_second |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << ((i_roll - 4) << 3));
		}
	}

	// Byte-wide exlusive scan over Histogram_list:
	// we're sure that each byte is less than 32 (histogram of a warp) and hence no possible overflow
	// inclusive scan on first 4 bytes:
	Histogram_list_first = (Histogram_list_first) + (Histogram_list_first << 8) + (Histogram_list_first << 16) + (Histogram_list_first << 24);
	// exclusive on second 4 bytes
	Histogram_list_second = (Histogram_list_second << 8) + (Histogram_list_second << 16) + 
		(Histogram_list_second << 24);
	// adding the summation of all first 4 bytes to all second 4 bytes of exclusive scan	
	uint32_t temp_sum = (Histogram_list_first & 0xFF000000) >> 24;
	temp_sum = temp_sum + (temp_sum << 8) + (temp_sum << 16) + (temp_sum << 24);
	Histogram_list_second += temp_sum;
	// making it exclusive scan
	Histogram_list_first <<= 8;

	// ==== storing the results back into smem:
	if(laneId < NUM_BUCKETS)
		local_histogram_smem[laneId * NUM_WARPS_4 + warpId] = binCounter;

	__syncthreads();

	// ==== Computing segmented scans per bucket + warp-wide scan over histograms
	if(threadIdx.x < (NUM_BUCKETS * NUM_WARPS_4))
	{
		uint32_t bucket_histo = local_histogram_smem[threadIdx.x];
		uint32_t reduction = bucket_histo;
		uint32_t temp_sum = __shfl_up(reduction, 1, 32);
		if((laneId & 0x03) >= 1) reduction += temp_sum;
		temp_sum = __shfl_up(reduction, 2, 32);
		if((laneId & 0x03) >= 2) reduction += temp_sum;

		// writing back the results (exclusive scan):
		local_histogram_smem[threadIdx.x] = reduction - bucket_histo;
		// writing back the histogram results into smem: 
		if((laneId & 0x03) == 0x03)
			global_histogram_smem[NUM_BUCKETS + (warpId << 3) + (laneId >> 2)] = reduction;
	}
	__syncthreads();

	// ==== computing the final indices and performing multisplit in smem
	// each warp computing its own warp-wide histogram scan:
	uint32_t bucket_offset = global_histogram_smem[NUM_BUCKETS + laneId];
	uint32_t local_offset = bucket_offset;

	// we decide to run this part for every warp to avoid more __syncthreads
	// inclusive scan over histogram results of each bucket in the block
	#pragma unroll
	for(int i = 0; i<LOG_BUCKETS; i++)
	{
		uint32_t temp_scan = __shfl_up(local_offset, (1<<i), 32);
		if(laneId >= (1<<i)) local_offset += temp_scan;
	}
	local_offset -= bucket_offset; // making it exclusive
	// at this point, for laneId < NUM_BUCKETS: local_offset equals number of elements within smaller buckets 

	#pragma unroll 
	for(int i_roll = 0; i_roll < 4; i_roll++)
	{	
		uint32_t temp_scanned_roll_histo = __shfl(Histogram_list_first, (myBucket_list_first & 0xFF));
		uint32_t myNewBlockIndex = ((temp_scanned_roll_histo >> (i_roll << 3)) & 0xFF) // among all roles with the same warp
				+ (myLocalIndex_list_first & 0xFF) // among my current roll
				+ local_histogram_smem[(myBucket_list_first & 0xFF) * NUM_WARPS_4 + warpId] // block-wide index (my own bucket)
				+ __shfl(local_offset, (myBucket_list_first & 0xFF), 32); // block-wide index (other buckets)
		
		myBucket_list_first >>= 8;
		myLocalIndex_list_first >>= 8;

		keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
		values_ms_smem[myNewBlockIndex] = input_value[i_roll];
	}
	#pragma unroll 
	for(int i_roll = 4; i_roll < NUM_ROLLS; i_roll++)
	{	
		uint32_t temp_scanned_roll_histo = __shfl(Histogram_list_second, (myBucket_list_second & 0xFF));
		uint32_t myNewBlockIndex = ((temp_scanned_roll_histo >> ((i_roll - 4) << 3)) & 0xFF) // among all roles with the same warp
				+ (myLocalIndex_list_second & 0xFF) // among my current roll
				+ local_histogram_smem[(myBucket_list_second & 0xFF) * NUM_WARPS_4 + warpId] // block-wide index (my own bucket)
				+ __shfl(local_offset, (myBucket_list_second & 0xFF), 32); // block-wide index (other buckets)
		
		myBucket_list_second >>= 8;
		myLocalIndex_list_second >>= 8;

		keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
		values_ms_smem[myNewBlockIndex] = input_value[i_roll];
	}	
	__syncthreads();

	// ==== Final stage: Tranferring elements from smem into gmem:
	#pragma unroll 
	for(int kk = 0; kk<NUM_ROLLS; kk++)
	{
		uint32_t temp_index = threadIdx.x + kk * blockDim.x;
		input_key[0] = keys_ms_smem[temp_index];
		input_value[0] = values_ms_smem[temp_index];
		if(blockIdx.x == (gridDim.x - 1))
			myBucket_list_first = ((temp_index + NUM_ROLLS * blockIdx.x * blockDim.x)<num_elements)?bucket_identifier(input_key[0]):(NUM_BUCKETS-1);
		else
			myBucket_list_first = bucket_identifier(input_key[0]);
		myLocalIndex_list_first = global_histogram_smem[myBucket_list_first] // global offset
							+ (threadIdx.x + kk * blockDim.x) 										// local offset
							- __shfl(local_offset, myBucket_list_first, 32);
		if(myLocalIndex_list_first < num_elements)
		{
			d_key_out[myLocalIndex_list_first] = input_key[0];
			d_value_out[myLocalIndex_list_first] = input_value[0];
		}
	}
}
//=====================================
template<
	uint32_t		NUM_ROLLS,
	typename 		bucket_t,
	typename 		KeyT,
	typename 		ValueT>
__global__ void 
BMS_postscan_64bucket_256_pairs(
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

	__shared__ uint32_t smem[64 + 32 + (32 * NUM_WARPS_8) + (64 * NUM_WARPS_8 * NUM_ROLLS)
		];
	uint32_t *global_histogram_smem = 	smem; // first 64: global histogram, next 32: within this block
	uint32_t *local_histogram_smem	=		&smem[96];
	uint32_t *keys_ms_smem = 						&smem[96 + (32 * NUM_WARPS_8)];
	uint32_t *values_ms_smem = 					&smem[96 + (32 * NUM_WARPS_8) + (32 * NUM_WARPS_8 * NUM_ROLLS)];

	KeyT input_key[NUM_ROLLS]; 				// stores all keys regarding to this thread
	ValueT input_value[NUM_ROLLS];
	uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
	uint32_t myBucket_list = 0;					// each byte contains bucket index per roll
	uint32_t Histogram_list_lo = 0; // for 0-31 buckets
	uint32_t Histogram_list_hi = 0;	// for 32-63 buckets								
	uint32_t binCounter = 0;							// total count of elements within its in-charge bucket

	// ==== Loading back histogram results from gmem into smem:
	if(warpId == 0) // for buckets 0-31
		global_histogram_smem[laneId] = __ldg(&d_histogram[laneId * gridDim.x + blockIdx.x]);
	else if(warpId == 1) // for buckets 32-63
		global_histogram_smem[32 + laneId] = __ldg(&d_histogram[(32 + laneId) * gridDim.x + blockIdx.x]);

	if(blockIdx.x == (gridDim.x - 1))
	{
	// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
			input_value[i_roll] = (temp_address < num_elements)?__ldg(&d_value_in[temp_address]):0xFFFFFFFF;

			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(63);
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto_lo = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_lo &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			uint32_t rx_buffer = __ballot(myBucket & 0x20); // the 6th bit
			myIndexBmp &= ((myBucket & 0x20)?rx_buffer:(~rx_buffer));
			uint32_t myHisto_hi = myHisto_lo & rx_buffer;
			myHisto_lo = myHisto_lo & (~rx_buffer);
			myHisto_lo = __popc(myHisto_lo);
			myHisto_hi = __popc(myHisto_hi);
			binCounter  += (myHisto_lo + (myHisto_hi << 16)); // (histo_32_63, histo_0_31)
			Histogram_list_lo |= (myHisto_lo << (i_roll << 3));
			Histogram_list_hi |= (myHisto_hi << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}
	else{
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId;
			input_key[i_roll] = __ldg(&d_key_in[temp_address]);
			input_value[i_roll] = __ldg(&d_value_in[temp_address]);

			uint32_t myBucket = bucket_identifier(input_key[i_roll]);
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto_lo = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_lo &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			uint32_t rx_buffer = __ballot(myBucket & 0x20); // the 6th bit
			myIndexBmp &= ((myBucket & 0x20)?rx_buffer:(~rx_buffer));
			uint32_t myHisto_hi = myHisto_lo & rx_buffer;
			myHisto_lo = myHisto_lo & (~rx_buffer);
			myHisto_lo = __popc(myHisto_lo);
			myHisto_hi = __popc(myHisto_hi);
			binCounter  += (myHisto_lo + (myHisto_hi << 16)); // (histo_32_63, histo_0_31)
			Histogram_list_lo |= (myHisto_lo << (i_roll << 3));
			Histogram_list_hi |= (myHisto_hi << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}

	// Byte-wide exlusive scan over Histogram_list:
	// we're sure that each byte is less than 32 (histogram of a warp) and hence no possible overflow
	Histogram_list_lo = (Histogram_list_lo << 8) + (Histogram_list_lo << 16) + (Histogram_list_lo << 24);
	Histogram_list_hi = (Histogram_list_hi << 8) + (Histogram_list_hi << 16) + (Histogram_list_hi << 24);

	// ==== storing the results back into smem:
	local_histogram_smem[laneId * NUM_WARPS_8 + warpId] = binCounter;

	__syncthreads();

// ==== Computing segmented scans per bucket + warp-wide scan over histograms
	uint32_t bucket_histo = local_histogram_smem[threadIdx.x];
	uint32_t reduction = bucket_histo;
	uint32_t temp = __shfl_up(reduction, 1, 32);
	if((laneId & 0x07) >= 1) reduction += temp;
	temp = __shfl_up(reduction, 2, 32);
	if((laneId & 0x07) >= 2) reduction += temp;
	temp = __shfl_up(reduction, 4, 32);
	if((laneId & 0x07) >= 4) reduction += temp;

	// writing back the results (exclusive scan):
	local_histogram_smem[threadIdx.x] = reduction - bucket_histo;
	// writing back the histogram results into smem: 
	if((laneId & 0x07) == 0x07)
		global_histogram_smem[64 + (warpId << 2) + (laneId >> 3)] = reduction;

	__syncthreads();

	// ==== computing the final indices and performing multisplit in smem
	// each warp computing its own warp-wide histogram scan:
	uint32_t bucket_offset = global_histogram_smem[64 + laneId];
	uint32_t local_offset = bucket_offset;

	// we decide to run this part for every warp to avoid more __syncthreads
	// inclusive scan over histogram results of each bucket in the block
	#pragma unroll
	for(int i = 0; i<5; i++)
	{
		uint32_t temp_scan = __shfl_up(local_offset, 1<<i, 32);
		if(laneId >= (1<<i)) local_offset += temp_scan;
	}
	temp = __shfl(local_offset,31,32);
	local_offset += ((temp & 0x0000FFFF) << 16); // adding results of bucket 32 to all buckets
	local_offset -= bucket_offset; // making it exclusive

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{	
		uint32_t temp_scanned_roll_lo_histo = __shfl(Histogram_list_lo, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_hi_histo = __shfl(Histogram_list_hi, (myBucket_list & 0x1F));
		uint32_t myNewBlockIndex = (myBucket_list & 0x20)?((temp_scanned_roll_hi_histo >> (i_roll << 3)) & 0xFF):((temp_scanned_roll_lo_histo >> (i_roll << 3)) & 0xFF); // among all roles with the same warp
		myNewBlockIndex	+= (myLocalIndex_list & 0xFF); // among my current roll
		uint32_t temp_local_block = local_histogram_smem[(myBucket_list & 0x1F) * NUM_WARPS_8 + warpId]; // block-wide index (my own bucket)
		myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_block >> 16:temp_local_block & 0x0000FFFF);
		uint32_t temp_local_warp =  __shfl(local_offset, (myBucket_list & 0x1F), 32); // block-wide index (other buckets)
		myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_warp >> 16:temp_local_warp & 0x0000FFFF);

		myBucket_list >>= 8;
		myLocalIndex_list >>= 8;
		keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
		values_ms_smem[myNewBlockIndex] = input_value[i_roll];
	}
	__syncthreads();

	// ==== Final stage: Tranferring elements from smem into gmem:
	#pragma unroll 
	for(int kk = 0; kk<NUM_ROLLS; kk++)
	{
		uint32_t temp_index = threadIdx.x + kk * blockDim.x;
		input_key[0] = keys_ms_smem[temp_index];
		input_value[0] = values_ms_smem[temp_index];
		if(blockIdx.x == (gridDim.x - 1))
			myBucket_list = ((temp_index + NUM_ROLLS * blockIdx.x * blockDim.x) < num_elements)?bucket_identifier(input_key[0]):63;
		else
			myBucket_list = bucket_identifier(input_key[0]);
		myLocalIndex_list = global_histogram_smem[myBucket_list] + (threadIdx.x + kk * blockDim.x); 				
		uint32_t temp_local_warp =  __shfl(local_offset, myBucket_list & 0x1F, 32);
		myLocalIndex_list -= ((myBucket_list & 0x20)?temp_local_warp >> 16:temp_local_warp & 0x0000FFFF);
		if(myLocalIndex_list < num_elements){
			d_key_out[myLocalIndex_list] = input_key[0];
			d_value_out[myLocalIndex_list] = input_value[0];
		}
	}
}
//========================================
template<
	uint32_t		NUM_ROLLS,
	typename 		bucket_t,
	typename 		KeyT,
	typename 		ValueT>
__global__ void 
BMS_postscan_128bucket_256_pairs(
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

	__shared__ uint32_t smem[2*(64 + 32) + (64 * NUM_WARPS_8) + (64 * NUM_WARPS_8 * NUM_ROLLS)
		];
	uint32_t *global_histogram_smem = 	smem; // first 64: global histogram, next 32: within this block
	uint32_t *local_histogram_smem	=		&smem[2*96];
	uint32_t *keys_ms_smem = 						&smem[2*96 + (64 * NUM_WARPS_8)];
	uint32_t *values_ms_smem = 					&smem[2*96 + (64 * NUM_WARPS_8) + 32 * NUM_WARPS_8 * NUM_ROLLS];

	KeyT input_key[NUM_ROLLS]; 				// stores all keys regarding to this thread
	ValueT input_value[NUM_ROLLS]; 				// stores all values regarding to this thread
	uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
	uint32_t myBucket_list = 0;					// each byte contains bucket index per roll

	uint32_t Histogram_list_1 = 0; 	// for 0-31 buckets
	uint32_t Histogram_list_2 = 0;	// for 32-63 buckets								
	uint32_t Histogram_list_3 = 0; 	// for 64-95 buckets
	uint32_t Histogram_list_4 = 0;	// for 96-127 buckets								

	uint32_t binCounter_12 = 0;							// total count of elements within its in-charge bucket
	uint32_t binCounter_34 = 0;

	// ==== Loading back histogram results from gmem into smem:
	if(warpId < 4)
	{
		global_histogram_smem[(warpId << 5) + laneId] = __ldg(&d_histogram[((warpId << 5) + laneId) * gridDim.x + blockIdx.x]);
	}

	if(blockIdx.x == (gridDim.x - 1))
	{
	// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + (i_roll << 5) + laneId;

			input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
			input_value[i_roll] = (temp_address < num_elements)?__ldg(&d_value_in[temp_address]):0xFFFFFFFF;

			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(127);
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto_1 = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_1 &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit

			uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2); 	// for 32-63
			uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2; 	// for 64-95
			uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2;			// for 96-127
			myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2);				// for 0-31
			//updating myIndexBmp:
			myIndexBmp &= ((myBucket & 0x20)?rx_buffer_1:(~rx_buffer_1)); // 6th bit
			myIndexBmp &= ((myBucket & 0x40)?rx_buffer_2:(~rx_buffer_2)); // 7th bit

			myHisto_1 = __popc(myHisto_1);
			myHisto_2 = __popc(myHisto_2);
			myHisto_3 = __popc(myHisto_3);
			myHisto_4 = __popc(myHisto_4);

			binCounter_12  += (myHisto_1 + (myHisto_2 << 16)); // (histo_32_63, histo_0_31)
			binCounter_34  += (myHisto_3 + (myHisto_4 << 16)); // (histo_96_127, histo_64_95)

			Histogram_list_1 |= (myHisto_1 << (i_roll << 3));
			Histogram_list_2 |= (myHisto_2 << (i_roll << 3));
			Histogram_list_3 |= (myHisto_3 << (i_roll << 3));
			Histogram_list_4 |= (myHisto_4 << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}
	else{
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + (i_roll << 5) + laneId;

			input_key[i_roll] = __ldg(&d_key_in[temp_address]);
			input_value[i_roll] = __ldg(&d_value_in[temp_address]);

			uint32_t myBucket = bucket_identifier(input_key[i_roll]);
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto_1 = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_1 &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit

			uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2); 	// for 32-63
			uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2; 	// for 64-95
			uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2;			// for 96-127
			myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2);				// for 0-31
			//updating myIndexBmp:
			myIndexBmp &= ((myBucket & 0x20)?rx_buffer_1:(~rx_buffer_1)); // 6th bit
			myIndexBmp &= ((myBucket & 0x40)?rx_buffer_2:(~rx_buffer_2)); // 7th bit

			myHisto_1 = __popc(myHisto_1);
			myHisto_2 = __popc(myHisto_2);
			myHisto_3 = __popc(myHisto_3);
			myHisto_4 = __popc(myHisto_4);

			binCounter_12  += (myHisto_1 + (myHisto_2 << 16)); // (histo_32_63, histo_0_31)
			binCounter_34  += (myHisto_3 + (myHisto_4 << 16)); // (histo_96_127, histo_64_95)

			Histogram_list_1 |= (myHisto_1 << (i_roll << 3));
			Histogram_list_2 |= (myHisto_2 << (i_roll << 3));
			Histogram_list_3 |= (myHisto_3 << (i_roll << 3));
			Histogram_list_4 |= (myHisto_4 << (i_roll << 3));
			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}

	// Byte-wide exlusive scan over Histogram_list:
	// we're sure that each byte is less than 32 (histogram of a warp) and hence no possible overflow
	Histogram_list_1 = (Histogram_list_1 << 8) + (Histogram_list_1 << 16) + (Histogram_list_1 << 24);
	Histogram_list_2 = (Histogram_list_2 << 8) + (Histogram_list_2 << 16) + (Histogram_list_2 << 24);
	Histogram_list_3 = (Histogram_list_3 << 8) + (Histogram_list_3 << 16) + (Histogram_list_3 << 24);
	Histogram_list_4 = (Histogram_list_4 << 8) + (Histogram_list_4 << 16) + (Histogram_list_4 << 24);

	// ==== storing the results back into smem:
	local_histogram_smem[laneId * NUM_WARPS_8 + warpId] = binCounter_12;
	local_histogram_smem[256 + laneId * NUM_WARPS_8 + warpId] = binCounter_34;
	__syncthreads();

// ==== Computing segmented scans per bucket + warp-wide scan over histograms
	uint32_t bucket_histo_12 = local_histogram_smem[threadIdx.x];
	uint32_t bucket_histo_34 = local_histogram_smem[256 + threadIdx.x];
	uint32_t reduction_12 = bucket_histo_12;
	uint32_t reduction_34 = bucket_histo_34;
	uint32_t temp = __shfl_up(reduction_12, 1, 32);
	
	if((laneId & 0x07) >= 1) reduction_12 += temp;
	temp = __shfl_up(reduction_12, 2, 32);
	if((laneId & 0x07) >= 2) reduction_12 += temp;
	temp = __shfl_up(reduction_12, 4, 32);
	if((laneId & 0x07) >= 4) reduction_12 += temp;
	
	temp = __shfl_up(reduction_34, 1, 32);
	if((laneId & 0x07) >= 1) reduction_34 += temp;
	temp = __shfl_up(reduction_34, 2, 32);
	if((laneId & 0x07) >= 2) reduction_34 += temp;
	temp = __shfl_up(reduction_34, 4, 32);
	if((laneId & 0x07) >= 4) reduction_34 += temp;

	// writing back the results (exclusive scan):
	local_histogram_smem[threadIdx.x] = reduction_12 - bucket_histo_12;
	local_histogram_smem[256 + threadIdx.x] = reduction_34 - bucket_histo_34;
	// writing back the histogram results into smem: 
	if((laneId & 0x07) == 0x07){
		global_histogram_smem[128 + (warpId << 2) + (laneId >> 3)] = reduction_12;
		global_histogram_smem[160 + (warpId << 2) + (laneId >> 3)] = reduction_34;
	}

	__syncthreads();

	// ==== computing the final indices and performing multisplit in smem
	// each warp computing its own warp-wide histogram scan:
	uint32_t bucket_offset_12 = global_histogram_smem[128 + laneId];
	uint32_t bucket_offset_34 = global_histogram_smem[160 + laneId];
	uint32_t local_offset_12 = bucket_offset_12;
	uint32_t local_offset_34 = bucket_offset_34;

	// we decide to run this part for every warp to avoid more __syncthreads
	// inclusive scan over histogram results of each bucket in the block
	#pragma unroll
	for(int i = 0; i<5; i++)
	{
		uint32_t temp_scan_12 = __shfl_up(local_offset_12, 1<<i, 32);
		uint32_t temp_scan_34 = __shfl_up(local_offset_34, 1<<i, 32);
		if(laneId >= (1<<i)){ 
			local_offset_12 += temp_scan_12;
			local_offset_34 += temp_scan_34;
		}
	}
	temp = __shfl(local_offset_12,31,32);
	local_offset_12 += ((temp & 0x0000FFFF) << 16); // adding results of bucket 32 to all buckets
	temp = (temp >> 16) + (temp & 0x0000FFFF);

	uint32_t temp2 = __shfl(local_offset_34,31,32);
	local_offset_34 += ((temp2 & 0x0000FFFF) << 16); // adding results of bucket 32 to all buckets
	local_offset_34 += (temp + (temp << 16));
	
	local_offset_12 -= bucket_offset_12; // making it exclusive
	local_offset_34 -= bucket_offset_34; // making it exclusive


	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{	
		uint32_t temp_scanned_roll_1_histo = __shfl(Histogram_list_1, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_2_histo = __shfl(Histogram_list_2, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_3_histo = __shfl(Histogram_list_3, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_4_histo = __shfl(Histogram_list_4, (myBucket_list & 0x1F));

		uint32_t myNewBlockIndex;
		if((myBucket_list & 0x60) == 0x60)
			myNewBlockIndex = (temp_scanned_roll_4_histo >> (i_roll << 3)) 	& 0xFF;
		else if(myBucket_list & 0x20)
			myNewBlockIndex = (temp_scanned_roll_2_histo >> (i_roll << 3)) 	& 0xFF;
		else if(myBucket_list & 0x40)
			myNewBlockIndex = (temp_scanned_roll_3_histo >> (i_roll << 3))	& 0xFF;
		else
			myNewBlockIndex = (temp_scanned_roll_1_histo >> (i_roll << 3))	& 0xFF;

		myNewBlockIndex	+= (myLocalIndex_list & 0xFF); // among my current roll
		uint32_t temp_local_block = local_histogram_smem[((myBucket_list & 0x40) << 2) + (myBucket_list & 0x1F) * NUM_WARPS_8 + warpId]; // block-wide index (my own bucket)		

		if((myBucket_list & 0x20))
			myNewBlockIndex += (temp_local_block >> 16);
		else
			myNewBlockIndex += (temp_local_block & 0x0000FFFF);		
		uint32_t temp_local_warp_12 =  __shfl(local_offset_12, (myBucket_list & 0x1F), 32); // block-wide index (other buckets)
		uint32_t temp_local_warp_34 =  __shfl(local_offset_34, (myBucket_list & 0x1F), 32);

		if(myBucket_list & 0x40)
			myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_warp_34 >> 16:temp_local_warp_34 & 0x0000FFFF);
		else 
			myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_warp_12 >> 16:temp_local_warp_12 & 0x0000FFFF);
		myBucket_list >>= 8;
		myLocalIndex_list >>= 8;

		keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
		values_ms_smem[myNewBlockIndex] = input_value[i_roll];
	}
	__syncthreads();

	// ==== Final stage: Tranferring elements from smem into gmem:
	#pragma unroll 
	for(int kk = 0; kk<NUM_ROLLS; kk++)
	{
		uint32_t temp_index = threadIdx.x + kk * blockDim.x;
		input_key[0] = keys_ms_smem[temp_index];
		input_value[0] = values_ms_smem[temp_index];
		if(blockIdx.x == (gridDim.x - 1))
			myBucket_list = ((temp_index + NUM_ROLLS * blockIdx.x * blockDim.x) < num_elements)?bucket_identifier(input_key[0]):127;
		else
			myBucket_list = bucket_identifier(input_key[0]);
		myLocalIndex_list = global_histogram_smem[myBucket_list] + (threadIdx.x + kk * blockDim.x); 			
		uint32_t temp_local_warp_12 =  __shfl(local_offset_12, myBucket_list & 0x1F, 32);
		uint32_t temp_local_warp_34 =  __shfl(local_offset_34, myBucket_list & 0x1F, 32);
		if(myBucket_list & 0x40)
			myLocalIndex_list -= ((myBucket_list & 0x20)?temp_local_warp_34 >> 16:temp_local_warp_34 & 0x0000FFFF);
		else	
			myLocalIndex_list -= ((myBucket_list & 0x20)?temp_local_warp_12 >> 16:temp_local_warp_12 & 0x0000FFFF);
		if(myLocalIndex_list < num_elements){
			d_key_out[myLocalIndex_list] = input_key[0];
			d_value_out[myLocalIndex_list] = input_value[0];	
		}
	}
}
//====================================
template<
	uint32_t		NUM_ROLLS,
	typename 		bucket_t,
	typename 		KeyT,
	typename 		ValueT>
__global__ void 
BMS_postscan_256bucket_256_pairs(
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

	__shared__ uint32_t smem[4*(64 + 32) + (128 * NUM_WARPS_8) + (64 * NUM_WARPS_8 * NUM_ROLLS)
		];
	uint32_t *global_histogram_smem = 	smem; // first 64: global histogram, next 32: within this block
	uint32_t *local_histogram_smem	=		&smem[4*96];
	uint32_t *keys_ms_smem = 						&smem[4*96 + (128 * NUM_WARPS_8)];
	uint32_t *values_ms_smem = 					&smem[4*96 + (128 * NUM_WARPS_8) + 32 * NUM_WARPS_8 * NUM_ROLLS];

	KeyT input_key[NUM_ROLLS]; 				// stores all keys regarding to this thread
	ValueT input_value[NUM_ROLLS]; 				// stores all values regarding to this thread
	uint32_t myLocalIndex_list = 0;								// each byte contains localIndex per roll
	uint32_t myBucket_list = 0;					// each byte contains bucket index per roll

	uint32_t Histogram_list_1 = 0; 	// for 0-31 buckets
	uint32_t Histogram_list_2 = 0;	// for 32-63 buckets								
	uint32_t Histogram_list_3 = 0; 	// for 64-95 buckets
	uint32_t Histogram_list_4 = 0;	// for 96-127 buckets								
	uint32_t Histogram_list_5 = 0; 	// for 0-31 buckets
	uint32_t Histogram_list_6 = 0;	// for 32-63 buckets								
	uint32_t Histogram_list_7 = 0; 	// for 64-95 buckets
	uint32_t Histogram_list_8 = 0;	// for 96-127 buckets								


	uint32_t binCounter_12 = 0;							// total count of elements within its in-charge bucket
	uint32_t binCounter_34 = 0;
	uint32_t binCounter_56 = 0;
	uint32_t binCounter_78 = 0;

	// ==== Loading back histogram results from gmem into smem:
	global_histogram_smem[(warpId << 5) + laneId] = __ldg(&d_histogram[((warpId << 5) + laneId) * gridDim.x + blockIdx.x]);

	if(blockIdx.x == (gridDim.x - 1))
	{
	// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + (i_roll << 5) + laneId;
			input_key[i_roll] = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
			input_value[i_roll] = (temp_address < num_elements)?__ldg(&d_value_in[temp_address]):0xFFFFFFFF;
			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):255;
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto_1 = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_1 &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit
			uint32_t rx_buffer_3 = __ballot(myBucket & 0x80); // checking the 8th bit

			uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2) & (~rx_buffer_3); 	// for 32-63
			uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2 & (~rx_buffer_3); 	// for 64-95
			uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2 & (~rx_buffer_3);			// for 96-127
			uint32_t myHisto_5 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2) & rx_buffer_3;
			uint32_t myHisto_6 = myHisto_1 & (rx_buffer_1) & (~rx_buffer_2) & rx_buffer_3;
			uint32_t myHisto_7 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2 & rx_buffer_3;
			uint32_t myHisto_8 = myHisto_1 & rx_buffer_1 & rx_buffer_2 & rx_buffer_3;
			myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2) & (~rx_buffer_3);		// for 0-31
			//updating myIndexBmp:
			myIndexBmp &= ((myBucket & 0x20)?rx_buffer_1:(~rx_buffer_1)); // 6th bit
			myIndexBmp &= ((myBucket & 0x40)?rx_buffer_2:(~rx_buffer_2)); // 7th bit
			myIndexBmp &= ((myBucket & 0x80)?rx_buffer_3:(~rx_buffer_3)); // 8th bit

			myHisto_1 = __popc(myHisto_1);
			myHisto_2 = __popc(myHisto_2);
			myHisto_3 = __popc(myHisto_3);
			myHisto_4 = __popc(myHisto_4);
			myHisto_5 = __popc(myHisto_5);
			myHisto_6 = __popc(myHisto_6);
			myHisto_7 = __popc(myHisto_7);
			myHisto_8 = __popc(myHisto_8);

			binCounter_12  += (myHisto_1 + (myHisto_2 << 16)); // (histo_32_63, histo_0_31)
			binCounter_34  += (myHisto_3 + (myHisto_4 << 16)); // (histo_96_127, histo_64_95)
			binCounter_56  += (myHisto_5 + (myHisto_6 << 16));
			binCounter_78  += (myHisto_7 + (myHisto_8 << 16));

			Histogram_list_1 |= (myHisto_1 << (i_roll << 3));
			Histogram_list_2 |= (myHisto_2 << (i_roll << 3));
			Histogram_list_3 |= (myHisto_3 << (i_roll << 3));
			Histogram_list_4 |= (myHisto_4 << (i_roll << 3));
			Histogram_list_5 |= (myHisto_5 << (i_roll << 3));
			Histogram_list_6 |= (myHisto_6 << (i_roll << 3));
			Histogram_list_7 |= (myHisto_7 << (i_roll << 3));
			Histogram_list_8 |= (myHisto_8 << (i_roll << 3));

			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}
	else{
		// ==== recomputing histograms within each roll:
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + (i_roll << 5) + laneId;
			input_key[i_roll] = __ldg(&d_key_in[temp_address]);
			input_value[i_roll] = __ldg(&d_value_in[temp_address]);
			uint32_t myBucket = bucket_identifier(input_key[i_roll]);
			myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

			uint32_t myHisto_1 = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_1 &= (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit
			uint32_t rx_buffer_3 = __ballot(myBucket & 0x80); // checking the 8th bit

			uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2) & (~rx_buffer_3); 	// for 32-63
			uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2 & (~rx_buffer_3); 	// for 64-95
			uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2 & (~rx_buffer_3);			// for 96-127
			uint32_t myHisto_5 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2) & rx_buffer_3;
			uint32_t myHisto_6 = myHisto_1 & (rx_buffer_1) & (~rx_buffer_2) & rx_buffer_3;
			uint32_t myHisto_7 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2 & rx_buffer_3;
			uint32_t myHisto_8 = myHisto_1 & rx_buffer_1 & rx_buffer_2 & rx_buffer_3;
			myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2) & (~rx_buffer_3);		// for 0-31
			//updating myIndexBmp:
			myIndexBmp &= ((myBucket & 0x20)?rx_buffer_1:(~rx_buffer_1)); // 6th bit
			myIndexBmp &= ((myBucket & 0x40)?rx_buffer_2:(~rx_buffer_2)); // 7th bit
			myIndexBmp &= ((myBucket & 0x80)?rx_buffer_3:(~rx_buffer_3)); // 8th bit

			myHisto_1 = __popc(myHisto_1);
			myHisto_2 = __popc(myHisto_2);
			myHisto_3 = __popc(myHisto_3);
			myHisto_4 = __popc(myHisto_4);
			myHisto_5 = __popc(myHisto_5);
			myHisto_6 = __popc(myHisto_6);
			myHisto_7 = __popc(myHisto_7);
			myHisto_8 = __popc(myHisto_8);

			binCounter_12  += (myHisto_1 + (myHisto_2 << 16)); // (histo_32_63, histo_0_31)
			binCounter_34  += (myHisto_3 + (myHisto_4 << 16)); // (histo_96_127, histo_64_95)
			binCounter_56  += (myHisto_5 + (myHisto_6 << 16));
			binCounter_78  += (myHisto_7 + (myHisto_8 << 16));

			Histogram_list_1 |= (myHisto_1 << (i_roll << 3));
			Histogram_list_2 |= (myHisto_2 << (i_roll << 3));
			Histogram_list_3 |= (myHisto_3 << (i_roll << 3));
			Histogram_list_4 |= (myHisto_4 << (i_roll << 3));
			Histogram_list_5 |= (myHisto_5 << (i_roll << 3));
			Histogram_list_6 |= (myHisto_6 << (i_roll << 3));
			Histogram_list_7 |= (myHisto_7 << (i_roll << 3));
			Histogram_list_8 |= (myHisto_8 << (i_roll << 3));

			// each byte contains local index per roll
			myLocalIndex_list |= ((__popc(myIndexBmp & (0xFFFFFFFF >> (31-laneId))) - 1) << (i_roll << 3));
		}
	}

	// Byte-wide exlusive scan over Histogram_list:
	// we're sure that each byte is less than 32 (histogram of a warp) and hence no possible overflow
	Histogram_list_1 = (Histogram_list_1 << 8) + (Histogram_list_1 << 16) + (Histogram_list_1 << 24);
	Histogram_list_2 = (Histogram_list_2 << 8) + (Histogram_list_2 << 16) + (Histogram_list_2 << 24);
	Histogram_list_3 = (Histogram_list_3 << 8) + (Histogram_list_3 << 16) + (Histogram_list_3 << 24);
	Histogram_list_4 = (Histogram_list_4 << 8) + (Histogram_list_4 << 16) + (Histogram_list_4 << 24);
	Histogram_list_5 = (Histogram_list_5 << 8) + (Histogram_list_5 << 16) + (Histogram_list_5 << 24);
	Histogram_list_6 = (Histogram_list_6 << 8) + (Histogram_list_6 << 16) + (Histogram_list_6 << 24);
	Histogram_list_7 = (Histogram_list_7 << 8) + (Histogram_list_7 << 16) + (Histogram_list_7 << 24);
	Histogram_list_8 = (Histogram_list_8 << 8) + (Histogram_list_8 << 16) + (Histogram_list_8 << 24);			

	// ==== storing the results back into smem:
	local_histogram_smem[laneId * NUM_WARPS_8 + warpId] = binCounter_12;
	local_histogram_smem[256 + laneId * NUM_WARPS_8 + warpId] = binCounter_34;
	local_histogram_smem[512 + laneId * NUM_WARPS_8 + warpId] = binCounter_56;
	local_histogram_smem[768 + laneId * NUM_WARPS_8 + warpId] = binCounter_78;	
	__syncthreads();

// ==== Computing segmented scans per bucket + warp-wide scan over histograms
	uint32_t bucket_histo_12 = local_histogram_smem[threadIdx.x];
	uint32_t bucket_histo_34 = local_histogram_smem[256 + threadIdx.x];
	uint32_t bucket_histo_56 = local_histogram_smem[512 + threadIdx.x];
	uint32_t bucket_histo_78 = local_histogram_smem[768 + threadIdx.x];

	uint32_t reduction_12 = bucket_histo_12;
	uint32_t reduction_34 = bucket_histo_34;
	uint32_t reduction_56 = bucket_histo_56;
	uint32_t reduction_78 = bucket_histo_78;

	uint32_t temp = __shfl_up(reduction_12, 1, 32);
	if((laneId & 0x07) >= 1) reduction_12 += temp;
	temp = __shfl_up(reduction_12, 2, 32);
	if((laneId & 0x07) >= 2) reduction_12 += temp;
	temp = __shfl_up(reduction_12, 4, 32);
	if((laneId & 0x07) >= 4) reduction_12 += temp;
	
	temp = __shfl_up(reduction_34, 1, 32);
	if((laneId & 0x07) >= 1) reduction_34 += temp;
	temp = __shfl_up(reduction_34, 2, 32);
	if((laneId & 0x07) >= 2) reduction_34 += temp;
	temp = __shfl_up(reduction_34, 4, 32);
	if((laneId & 0x07) >= 4) reduction_34 += temp;

	temp = __shfl_up(reduction_56, 1, 32);
	if((laneId & 0x07) >= 1) reduction_56 += temp;
	temp = __shfl_up(reduction_56, 2, 32);
	if((laneId & 0x07) >= 2) reduction_56 += temp;
	temp = __shfl_up(reduction_56, 4, 32);
	if((laneId & 0x07) >= 4) reduction_56 += temp;

	temp = __shfl_up(reduction_78, 1, 32);
	if((laneId & 0x07) >= 1) reduction_78 += temp;
	temp = __shfl_up(reduction_78, 2, 32);
	if((laneId & 0x07) >= 2) reduction_78 += temp;
	temp = __shfl_up(reduction_78, 4, 32);
	if((laneId & 0x07) >= 4) reduction_78 += temp;

	// writing back the results (exclusive scan):
	local_histogram_smem[threadIdx.x] = reduction_12 - bucket_histo_12;
	local_histogram_smem[256 + threadIdx.x] = reduction_34 - bucket_histo_34;
	local_histogram_smem[512 + threadIdx.x] = reduction_56 - bucket_histo_56;
	local_histogram_smem[768 + threadIdx.x] = reduction_78 - bucket_histo_78;
	// writing back the histogram results into smem: 
	if((laneId & 0x07) == 0x07){
		global_histogram_smem[256 + (warpId << 2) + (laneId >> 3)] = reduction_12;
		global_histogram_smem[288 + (warpId << 2) + (laneId >> 3)] = reduction_34;
		global_histogram_smem[320 + (warpId << 2) + (laneId >> 3)] = reduction_56;
		global_histogram_smem[352 + (warpId << 2) + (laneId >> 3)] = reduction_78;
	}

	__syncthreads();

	// ==== computing the final indices and performing multisplit in smem
	// each warp computing its own warp-wide histogram scan:
	uint32_t bucket_offset_12 = global_histogram_smem[256 + laneId];
	uint32_t bucket_offset_34 = global_histogram_smem[288 + laneId];
	uint32_t bucket_offset_56 = global_histogram_smem[320 + laneId];
	uint32_t bucket_offset_78 = global_histogram_smem[352 + laneId];

	uint32_t local_offset_12 = bucket_offset_12;
	uint32_t local_offset_34 = bucket_offset_34;
	uint32_t local_offset_56 = bucket_offset_56;
	uint32_t local_offset_78 = bucket_offset_78;

	// we decide to run this part for every warp to avoid more __syncthreads
	// inclusive scan over histogram results of each bucket in the block
	#pragma unroll
	for(int i = 0; i<5; i++)
	{
		uint32_t temp_scan_12 = __shfl_up(local_offset_12, 1<<i, 32);
		uint32_t temp_scan_34 = __shfl_up(local_offset_34, 1<<i, 32);
		uint32_t temp_scan_56 = __shfl_up(local_offset_56, 1<<i, 32);
		uint32_t temp_scan_78 = __shfl_up(local_offset_78, 1<<i, 32);

		if(laneId >= (1<<i)){ 
			local_offset_12 += temp_scan_12;
			local_offset_34 += temp_scan_34;
			local_offset_56 += temp_scan_56;
			local_offset_78 += temp_scan_78;			
		}
	}
	temp = __shfl(local_offset_12,31,32);
	local_offset_12 += ((temp & 0x0000FFFF) << 16); 
	temp = (temp >> 16) + (temp & 0x0000FFFF);

	uint32_t temp2 = __shfl(local_offset_34,31,32);
	local_offset_34 += ((temp2 & 0x0000FFFF) << 16); 
	temp2 = (temp2 >> 16) + (temp2 & 0x0000FFFF);

	uint32_t temp3 = __shfl(local_offset_56,31,32);
	local_offset_56 += ((temp3 & 0x0000FFFF) << 16); 
	temp3 = (temp3 >> 16) + (temp3 & 0x0000FFFF);

	uint32_t temp4 = __shfl(local_offset_78,31,32);
	local_offset_78 += ((temp4 & 0x0000FFFF) << 16); 
	// temp4 = (temp4 >> 16) + (temp4 & 0x0000FFFF);

	local_offset_34 += (temp + (temp << 16));
	local_offset_56 += (temp + temp2 + ((temp + temp2) << 16));
	local_offset_78 += (temp + temp2 + temp3 + ((temp + temp2 + temp3) << 16));		

	local_offset_12 -= bucket_offset_12; // making it exclusive
	local_offset_34 -= bucket_offset_34; // making it exclusive
	local_offset_56 -= bucket_offset_56; // making it exclusive
	local_offset_78 -= bucket_offset_78; // making it exclusive


	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{	
		uint32_t temp_scanned_roll_1_histo = __shfl(Histogram_list_1, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_2_histo = __shfl(Histogram_list_2, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_3_histo = __shfl(Histogram_list_3, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_4_histo = __shfl(Histogram_list_4, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_5_histo = __shfl(Histogram_list_5, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_6_histo = __shfl(Histogram_list_6, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_7_histo = __shfl(Histogram_list_7, (myBucket_list & 0x1F));
		uint32_t temp_scanned_roll_8_histo = __shfl(Histogram_list_8, (myBucket_list & 0x1F));

		uint32_t myNewBlockIndex;
		switch((myBucket_list & 0xE0) >> 5)
		{
			case 0:
			myNewBlockIndex = (temp_scanned_roll_1_histo >> (i_roll << 3)) 	& 0xFF; break;
			case 1:
			myNewBlockIndex = (temp_scanned_roll_2_histo >> (i_roll << 3)) 	& 0xFF; break;
			case 2:
			myNewBlockIndex = (temp_scanned_roll_3_histo >> (i_roll << 3)) 	& 0xFF; break;
			case 3:
			myNewBlockIndex = (temp_scanned_roll_4_histo >> (i_roll << 3)) 	& 0xFF; break;
			case 4:
			myNewBlockIndex = (temp_scanned_roll_5_histo >> (i_roll << 3)) 	& 0xFF; break;
			case 5:
			myNewBlockIndex = (temp_scanned_roll_6_histo >> (i_roll << 3)) 	& 0xFF; break;
			case 6:
			myNewBlockIndex = (temp_scanned_roll_7_histo >> (i_roll << 3)) 	& 0xFF; break;
			case 7:
			myNewBlockIndex = (temp_scanned_roll_8_histo >> (i_roll << 3)) 	& 0xFF; break;
		}

		myNewBlockIndex	+= (myLocalIndex_list & 0xFF); // among my current roll
		uint32_t temp_local_block = local_histogram_smem[((myBucket_list & 0xC0) << 2) + (myBucket_list & 0x1F) * NUM_WARPS_8 + warpId]; // block-wide index (my own bucket)		

		if((myBucket_list & 0x20))
			myNewBlockIndex += (temp_local_block >> 16);
		else
			myNewBlockIndex += (temp_local_block & 0x0000FFFF);	

		uint32_t temp_local_warp_12 =  __shfl(local_offset_12, (myBucket_list & 0x1F), 32); // block-wide index (other buckets)
		uint32_t temp_local_warp_34 =  __shfl(local_offset_34, (myBucket_list & 0x1F), 32);
		uint32_t temp_local_warp_56 =  __shfl(local_offset_56, (myBucket_list & 0x1F), 32);
		uint32_t temp_local_warp_78 =  __shfl(local_offset_78, (myBucket_list & 0x1F), 32);

		if((myBucket_list & 0xC0) == 0xC0)
			myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_warp_78 >> 16:temp_local_warp_78 & 0x0000FFFF);
		else if(myBucket_list & 0x80)
			myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_warp_56 >> 16:temp_local_warp_56 & 0x0000FFFF);
		else if(myBucket_list & 0x40)
			myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_warp_34 >> 16:temp_local_warp_34 & 0x0000FFFF);
		else 
			myNewBlockIndex += ((myBucket_list & 0x20)?temp_local_warp_12 >> 16:temp_local_warp_12 & 0x0000FFFF);

		myBucket_list >>= 8;
		myLocalIndex_list >>= 8;

		keys_ms_smem[myNewBlockIndex] = input_key[i_roll];
		values_ms_smem[myNewBlockIndex] = input_value[i_roll];
	}
	__syncthreads();

	// ==== Final stage: Tranferring elements from smem into gmem:
	#pragma unroll 
	for(int kk = 0; kk<NUM_ROLLS; kk++)
	{
		uint32_t temp_index = threadIdx.x + kk * blockDim.x;
		input_key[0] = keys_ms_smem[temp_index];
		input_value[0] = values_ms_smem[temp_index];
		if(blockIdx.x == (gridDim.x - 1))
			myBucket_list = (temp_index < num_elements)?bucket_identifier(input_key[0]):255;
		else
			myBucket_list = bucket_identifier(input_key[0]);
		myLocalIndex_list = global_histogram_smem[myBucket_list] + (threadIdx.x + kk * blockDim.x); 			
		uint32_t temp_local_warp_12 =  __shfl(local_offset_12, myBucket_list & 0x1F, 32);
		uint32_t temp_local_warp_34 =  __shfl(local_offset_34, myBucket_list & 0x1F, 32);
		uint32_t temp_local_warp_56 =  __shfl(local_offset_56, myBucket_list & 0x1F, 32);
		uint32_t temp_local_warp_78 =  __shfl(local_offset_78, myBucket_list & 0x1F, 32);		
		if((myBucket_list & 0xC0) == 0xC0)
			myLocalIndex_list -= ((myBucket_list & 0x20)?temp_local_warp_78 >> 16:temp_local_warp_78 & 0x0000FFFF);
		else if(myBucket_list & 0x80)
			myLocalIndex_list -= ((myBucket_list & 0x20)?temp_local_warp_56 >> 16:temp_local_warp_56 & 0x0000FFFF);
		else if(myBucket_list & 0x40)
			myLocalIndex_list -= ((myBucket_list & 0x20)?temp_local_warp_34 >> 16:temp_local_warp_34 & 0x0000FFFF);
		else 
			myLocalIndex_list -= ((myBucket_list & 0x20)?temp_local_warp_12 >> 16:temp_local_warp_12 & 0x0000FFFF);
		if(myLocalIndex_list < num_elements){
			d_key_out[myLocalIndex_list] = input_key[0];
			d_value_out[myLocalIndex_list] = input_value[0];
		}
	}
}
#endif