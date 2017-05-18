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

#ifndef MULTISPLIT_HISTOGRAM__
#define MULTISPLIT_HISTOGRAM__
#ifndef NUM_WARPS_4 
#define NUM_WARPS_4 4
#endif
#ifndef NUM_WARPS_8 
#define NUM_WARPS_8 8
#endif

#include "cub/thread/thread_search.cuh" // for binary search used in CUB

template<
	uint32_t		NUM_ROLLS,
	uint32_t		NUM_BUCKETS,
	uint32_t		LOG_BUCKETS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_even_128(
	KeyT* 			d_key_in,
	int 		num_elements,
	int* 		d_histogram, 
	const KeyT 				lower_level,
	const KeyT 				upper_level,
	const float				inverse_delta)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * NUM_BUCKETS)];

	uint32_t binCounter = 0;

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[
			(blockIdx.x * blockDim.x * NUM_ROLLS) + 
			((warpId * NUM_ROLLS) << 5) + 
			(i_roll << 5) + 
			laneId];

		uint32_t myBucket = static_cast<uint32_t>((input_key - lower_level)*inverse_delta);

		uint32_t myHisto = 0xFFFFFFFF;
		#pragma unroll
		for(int i = 0; i<LOG_BUCKETS; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
		}
		binCounter  += __popc(myHisto);
	}
	// aggregating the results from all warps:
	// simple case, for 16 buckets and 8 warps, will have 4-way bank conflict
	// == storing results into smem: bucket -> warp
	if(laneId < NUM_BUCKETS)
	{
		smem_histogram[laneId * NUM_WARPS_4 + warpId] = binCounter;
	}
	__syncthreads();

	// segmented reduction stage:
	if(threadIdx.x < ((NUM_BUCKETS * NUM_WARPS_4)))
	{
		uint32_t reduction = smem_histogram[threadIdx.x];
		reduction += __shfl_xor(reduction, 2, 32);
		reduction += __shfl_xor(reduction, 1, 32);
		// if(ATOMIC_USED){
		if(reduction && ((laneId & 0x03) == 0)) 
			atomicAdd(&d_histogram[(warpId << 3) + (laneId >> 2)], reduction);
		// }
		// else{
		// if((laneId & 0x03) == 0)
		// 	d_histogram[((warpId << 3) + (laneId >> 2)) * gridDim.x + 
		// 		blockIdx.x] = reduction;			
		// }
	}
}
//====================================
template<
	uint32_t		NUM_ROLLS,
	uint32_t 		NUM_BUCKETS,
	uint32_t		LOG_BUCKETS,
	typename		KeyT>
__global__ void 
multisplit2_histogram_even_256(
	KeyT* 			d_key_in,
	int 		num_elements,
	int* 		d_histogram, 
	const KeyT 				lower_level,
	const KeyT 				upper_level,
	const float				inverse_delta)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_8 * NUM_BUCKETS)];

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	
	uint32_t binCounter = 0;

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		uint32_t input_key = d_key_in[
			(blockIdx.x * blockDim.x * NUM_ROLLS) + 
			((warpId * NUM_ROLLS) << 5) + 
			(i_roll << 5) + 
			laneId];

		uint32_t myBucket = static_cast<uint32_t>((input_key - lower_level)*inverse_delta);

		uint32_t myHisto = 0xFFFFFFFF;
		#pragma unroll
		for(int i = 0; i<LOG_BUCKETS; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
		}
		binCounter  += __popc(myHisto);
	}
	// aggregating the results from all warps:
	// simple case, for 16 buckets and 8 warps, will have 4-way bank conflict
	// == storing results into smem: bucket -> warp
	if(laneId < NUM_BUCKETS)
	{
		smem_histogram[laneId * NUM_WARPS_8 + warpId] = binCounter;
	}
	__syncthreads();

	// segmented reduction stage:
	if(threadIdx.x < ((NUM_BUCKETS * NUM_WARPS_8)))
	{
		uint32_t reduction = smem_histogram[threadIdx.x];
		reduction += __shfl_xor(reduction, 4, 32);
		reduction += __shfl_xor(reduction, 2, 32);
		reduction += __shfl_xor(reduction, 1, 32);
		// if(ATOMIC_USED){
			if(reduction && ((laneId & 0x07) == 0)) 
				atomicAdd(&d_histogram[(warpId << 2) + (laneId >> 3)], reduction);
		// }
		// else{
		// if((laneId & 0x07) == 0)
		// 	d_histogram[((warpId << 2) + (laneId >> 3)) * gridDim.x + 
		// 		blockIdx.x] = reduction;			
		// }	
	}
}
//================================================
template<
	uint32_t		NUM_ROLLS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_even_64bin_128(
	KeyT* 			d_key_in,
	int 		num_elements,
	int* 		d_histogram, 
	const KeyT 				lower_level,
	const KeyT 				upper_level,
	const float				inverse_delta)
{
	// NUM_ROLL: number of consecutive windows read by one warp

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * 32)];

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[ \
			(blockIdx.x * blockDim.x * NUM_ROLLS) + \
			((warpId * NUM_ROLLS) << 5) + \
			(i_roll << 5) + \
			laneId];

		uint32_t myBucket = static_cast<uint32_t>((input_key - lower_level)*inverse_delta);

		uint32_t myHisto_1 = 0xFFFFFFFF; // for bucket 0-31
		
		#pragma unroll
		for(int i = 0; i<5; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto_1 = myHisto_1 & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
		}

		uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit

		uint32_t myHisto_2 = myHisto_1 & rx_buffer_1; 	// for 32-63
		myHisto_1 = myHisto_1 & (~rx_buffer_1);		// for 0-31

		binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
	}
	// storing results into smem:
	smem_histogram[laneId * NUM_WARPS_4 + warpId] = binCounter_12;
	__syncthreads();

	// segmented reduction stage:
	uint32_t reduction = smem_histogram[threadIdx.x];
	reduction += __shfl_xor(reduction, 2, 32);
	reduction += __shfl_xor(reduction, 1, 32);
	if(reduction && ((laneId & 0x03) == 0)) 
		atomicAdd(&d_histogram[(warpId << 3) + (laneId >> 2)], reduction & 0x0000FFFF);
	else if (reduction && ((laneId & 0x03) == 1))
	{
		atomicAdd(&d_histogram[32 + (warpId << 3) + (laneId >> 2)], (reduction >> 16));
	}
}
//================================================
template<
	uint32_t		NUM_ROLLS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_even_128bin_128(
	KeyT* 			d_key_in,
	int 		num_elements,
	int* 		d_histogram, 
	const KeyT 				lower_level,
	const KeyT 				upper_level,
	const float				inverse_delta)
{
	// NUM_ROLL: number of consecutive windows read by one warp

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * 64)];

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)
	uint32_t binCounter_34 = 0; // for (16bit: 96-127, 	16bit: 64-95)

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[ \
			(blockIdx.x * blockDim.x * NUM_ROLLS) + \
			((warpId * NUM_ROLLS) << 5) + \
			(i_roll << 5) + \
			laneId];

		uint32_t myBucket = static_cast<uint32_t>((input_key - lower_level)*inverse_delta);

		uint32_t myHisto_1 = 0xFFFFFFFF; // for bucket 0-31
		
		#pragma unroll
		for(int i = 0; i<5; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto_1 = myHisto_1 & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
		}

		uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
		uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit

		uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2); 	// for 32-63
		uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2; 	// for 64-95
		uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2;			// for 96-127
		myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2);		// for 0-31

		binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
		binCounter_34  += (__popc(myHisto_3) + (__popc(myHisto_4) << 16)); // (histo_96_127, histo_64_95)
	}

	// storing in warp -> bucket order
	smem_histogram[threadIdx.x] = binCounter_12;
	smem_histogram[128 + threadIdx.x] = binCounter_34;
	__syncthreads();

	if(warpId < 2){
		uint32_t reduction = smem_histogram[(warpId << 7) + laneId];
		reduction += smem_histogram[(warpId << 7) + laneId + 32];
		reduction += smem_histogram[(warpId << 7) + laneId + 64];
		reduction += smem_histogram[(warpId << 7) + laneId + 96];

		atomicAdd(&d_histogram[(warpId << 6) + laneId], reduction & 0x0000FFFF);
		atomicAdd(&d_histogram[32 + (warpId << 6) + laneId], reduction >> 16);
	}
}
//================================================
template<
	uint32_t		NUM_ROLLS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_even_256bin_128(
	KeyT* 			d_key_in,
	int 		num_elements,
	int* 		d_histogram, 
	const KeyT 				lower_level,
	const KeyT 				upper_level,
	const float				inverse_delta)
{
	// NUM_ROLL: number of consecutive windows read by one warp

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * 128)];

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)
	uint32_t binCounter_34 = 0; // for (16bit: 96-127, 	16bit: 64-95)
	uint32_t binCounter_56 = 0; // for (16bit: 160-191, 	16bit: 128-159)	
	uint32_t binCounter_78 = 0; // for (16bit: 224-255, 	16bit: 192-223)	

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[
			(blockIdx.x * blockDim.x * NUM_ROLLS) + 
			((warpId * NUM_ROLLS) << 5) + 
			(i_roll << 5) + 
			laneId];

		uint32_t myBucket = static_cast<uint32_t>((input_key - lower_level)*inverse_delta);

		uint32_t myHisto_1 = 0xFFFFFFFF; // for bucket 0-31
		
		#pragma unroll
		for(int i = 0; i<5; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto_1 = myHisto_1 & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
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

		binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
		binCounter_34  += (__popc(myHisto_3) + (__popc(myHisto_4) << 16)); // (histo_96_127, histo_64_95)
		binCounter_56  += (__popc(myHisto_5) + (__popc(myHisto_6) << 16)); // (histo_160-191, histo_128_159)
		binCounter_78  += (__popc(myHisto_7) + (__popc(myHisto_8) << 16)); // (histo_224_255, histo_192_223)
	}

	// storing in bucket -> warp order
	smem_histogram[threadIdx.x] = binCounter_12;
	smem_histogram[128 + threadIdx.x] = binCounter_34;
	smem_histogram[256 + threadIdx.x] = binCounter_56;
	smem_histogram[384 + threadIdx.x] = binCounter_78;
	__syncthreads();

	uint32_t reduction = smem_histogram[(warpId << 7) + laneId];
	reduction += smem_histogram[(warpId << 7) + laneId + 32];
	reduction += smem_histogram[(warpId << 7) + laneId + 64];
	reduction += smem_histogram[(warpId << 7) + laneId + 96];

	// warp_0 -> histo_12, warp_1 -> histo_34, warp_2 -> histo_56, warp_3 -> histo_78
	// if(reduction){
		atomicAdd(&d_histogram[(warpId << 6) + laneId], reduction & 0x0000FFFF);
		atomicAdd(&d_histogram[32 + (warpId << 6) + laneId], reduction >> 16);
	// }
}
//================================================
/*	
* Histogram with customized bin ranges 
*/
template<
	uint32_t		NUM_ROLLS,
	uint32_t		NUM_BUCKETS,
	uint32_t		LOG_BUCKETS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_range_128(
	KeyT* 								d_key_in,
	int 									num_elements,
	int*			 						d_histogram, 
	KeyT* 								d_level)
{
	// NUM_ROLL: number of consecutive windows read by one warp
	// currently assume no scaling factor between prescan and post scan. Next: scaling those.

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * NUM_BUCKETS) + NUM_BUCKETS];
	// float* levels_smem = static_cast<float*>(&smem_histogram[NUM_WARPS_4 * NUM_BUCKETS]);
	KeyT* levels_smem = (KeyT*)&smem_histogram[NUM_WARPS_4 * NUM_BUCKETS];

	if(warpId == 0 && laneId < NUM_BUCKETS)
		levels_smem[laneId] = d_level[laneId];
	__syncthreads();
	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter = 0;

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[ \
			(blockIdx.x * blockDim.x * NUM_ROLLS) + \
			((warpId * NUM_ROLLS) << 5) + \
			(i_roll << 5) + \
			laneId];

		uint32_t myBucket = cub::UpperBound(levels_smem, NUM_BUCKETS, input_key) - 1;

		uint32_t myHisto = 0xFFFFFFFF;
		#pragma unroll
		for(int i = 0; i<LOG_BUCKETS; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
		}
		binCounter  += __popc(myHisto);
	}
	// aggregating the results from all warps:
	// simple case, for 16 buckets and 8 warps, will have 4-way bank conflict
	// == storing results into smem: bucket -> warp
	if(laneId < NUM_BUCKETS)
	{
		smem_histogram[laneId * NUM_WARPS_4 + warpId] = binCounter;
	}
	__syncthreads();

	// segmented reduction stage:
	if(threadIdx.x < ((NUM_BUCKETS * NUM_WARPS_4)))
	{
		uint32_t reduction = smem_histogram[threadIdx.x];
		reduction += __shfl_xor(reduction, 2, 32);
		reduction += __shfl_xor(reduction, 1, 32);
		// if(ATOMIC_USED){
		if(reduction && ((laneId & 0x03) == 0)) 
			atomicAdd(&d_histogram[(warpId << 3) + (laneId >> 2)], reduction);
		// }
		// else{
		// if((laneId & 0x03) == 0)
		// 	d_histogram[((warpId << 3) + (laneId >> 2)) * gridDim.x + 
		// 		blockIdx.x] = reduction;			
		// }
	}
}
//======================================
template<
	uint32_t		NUM_ROLLS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_range_64bin_128(
	KeyT* 								d_key_in,
	int 									num_elements,
	int*			 						d_histogram, 
	KeyT* 								d_level)
{
	// NUM_ROLL: number of consecutive windows read by one warp

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * 32) + 64];
	KeyT* levels_smem = (KeyT*)&smem_histogram[NUM_WARPS_4 * 32];

	if(warpId < 2)
		levels_smem[threadIdx.x] = d_level[threadIdx.x];
	__syncthreads();

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[ \
			(blockIdx.x * blockDim.x * NUM_ROLLS) + \
			((warpId * NUM_ROLLS) << 5) + \
			(i_roll << 5) + \
			laneId];

		uint32_t myBucket = cub::UpperBound(levels_smem, 64, input_key) - 1;

		uint32_t myHisto_1 = 0xFFFFFFFF; // for bucket 0-31
		
		#pragma unroll
		for(int i = 0; i<5; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto_1 = myHisto_1 & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
		}

		uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit

		uint32_t myHisto_2 = myHisto_1 & rx_buffer_1; 	// for 32-63
		myHisto_1 = myHisto_1 & (~rx_buffer_1);		// for 0-31

		binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
	}
	// storing results into smem:
	smem_histogram[laneId * NUM_WARPS_4 + warpId] = binCounter_12;
	__syncthreads();

	// segmented reduction stage:
	uint32_t reduction = smem_histogram[threadIdx.x];
	reduction += __shfl_xor(reduction, 2, 32);
	reduction += __shfl_xor(reduction, 1, 32);
	if(reduction && ((laneId & 0x03) == 0)) 
		atomicAdd(&d_histogram[(warpId << 3) + (laneId >> 2)], reduction & 0x0000FFFF);
	else if (reduction && ((laneId & 0x03) == 1))
	{
		atomicAdd(&d_histogram[32 + (warpId << 3) + (laneId >> 2)], (reduction >> 16));
	}
}
//================================================
template<
	uint32_t		NUM_ROLLS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_range_128bin_128(
	KeyT* 								d_key_in,
	int 									num_elements,
	int*			 						d_histogram, 
	KeyT* 								d_level)
{
	// NUM_ROLL: number of consecutive windows read by one warp

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * 64) + 128];
	KeyT* levels_smem = (KeyT*)&smem_histogram[NUM_WARPS_4 * 64];

	levels_smem[threadIdx.x] = d_level[threadIdx.x];
	__syncthreads();

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)
	uint32_t binCounter_34 = 0; // for (16bit: 96-127, 	16bit: 64-95)

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[ \
			(blockIdx.x * blockDim.x * NUM_ROLLS) + \
			((warpId * NUM_ROLLS) << 5) + \
			(i_roll << 5) + \
			laneId];

		uint32_t myBucket = cub::UpperBound(levels_smem, 128, input_key) - 1;

		uint32_t myHisto_1 = 0xFFFFFFFF; // for bucket 0-31
		
		#pragma unroll
		for(int i = 0; i<5; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto_1 = myHisto_1 & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
		}

		uint32_t rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
		uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit

		uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2); 	// for 32-63
		uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2; 	// for 64-95
		uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2;			// for 96-127
		myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2);		// for 0-31

		binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
		binCounter_34  += (__popc(myHisto_3) + (__popc(myHisto_4) << 16)); // (histo_96_127, histo_64_95)
	}

	// storing in warp -> bucket order
	smem_histogram[threadIdx.x] = binCounter_12;
	smem_histogram[128 + threadIdx.x] = binCounter_34;
	__syncthreads();

	if(warpId < 2){
		uint32_t reduction = smem_histogram[(warpId << 7) + laneId];
		reduction += smem_histogram[(warpId << 7) + laneId + 32];
		reduction += smem_histogram[(warpId << 7) + laneId + 64];
		reduction += smem_histogram[(warpId << 7) + laneId + 96];

		atomicAdd(&d_histogram[(warpId << 6) + laneId], reduction & 0x0000FFFF);
		atomicAdd(&d_histogram[32 + (warpId << 6) + laneId], reduction >> 16);
	}
}
//======================================
template<
	uint32_t		NUM_ROLLS,
	typename 		KeyT>
__global__ void 
multisplit2_histogram_range_256bin_128(
	KeyT* 			d_key_in,
	int 				num_elements,
	int* 				d_histogram, 
	KeyT* 			d_level)
{
	// NUM_ROLL: number of consecutive windows read by one warp

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;
	
	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * 128) + 256];
	KeyT* levels_smem = (KeyT*)&smem_histogram[NUM_WARPS_4 * 128];

	levels_smem[threadIdx.x] = d_level[threadIdx.x];
	levels_smem[128 + threadIdx.x] = d_level[128 + threadIdx.x];
	__syncthreads();

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)
	uint32_t binCounter_34 = 0; // for (16bit: 96-127, 	16bit: 64-95)
	uint32_t binCounter_56 = 0; // for (16bit: 160-191, 	16bit: 128-159)	
	uint32_t binCounter_78 = 0; // for (16bit: 224-255, 	16bit: 192-223)	

	#pragma unroll 
	for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
	{
		KeyT input_key = d_key_in[
			(blockIdx.x * blockDim.x * NUM_ROLLS) + 
			((warpId * NUM_ROLLS) << 5) + 
			(i_roll << 5) + 
			laneId];

		uint32_t myBucket = cub::UpperBound(levels_smem, 256, input_key) - 1;

		uint32_t myHisto_1 = 0xFFFFFFFF; // for bucket 0-31
		
		#pragma unroll
		for(int i = 0; i<5; i++)
		{
			uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
			myHisto_1 = myHisto_1 & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
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

		binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
		binCounter_34  += (__popc(myHisto_3) + (__popc(myHisto_4) << 16)); // (histo_96_127, histo_64_95)
		binCounter_56  += (__popc(myHisto_5) + (__popc(myHisto_6) << 16)); // (histo_160-191, histo_128_159)
		binCounter_78  += (__popc(myHisto_7) + (__popc(myHisto_8) << 16)); // (histo_224_255, histo_192_223)
	}

	// storing in bucket -> warp order
	smem_histogram[threadIdx.x] = binCounter_12;
	smem_histogram[128 + threadIdx.x] = binCounter_34;
	smem_histogram[256 + threadIdx.x] = binCounter_56;
	smem_histogram[384 + threadIdx.x] = binCounter_78;
	__syncthreads();

	uint32_t reduction = smem_histogram[(warpId << 7) + laneId];
	reduction += smem_histogram[(warpId << 7) + laneId + 32];
	reduction += smem_histogram[(warpId << 7) + laneId + 64];
	reduction += smem_histogram[(warpId << 7) + laneId + 96];

	// warp_0 -> histo_12, warp_1 -> histo_34, warp_2 -> histo_56, warp_3 -> histo_78
	// if(reduction){
		atomicAdd(&d_histogram[(warpId << 6) + laneId], reduction & 0x0000FFFF);
		atomicAdd(&d_histogram[32 + (warpId << 6) + laneId], reduction >> 16);
	// }
}
#endif