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

#ifndef MULTISPLIT_BMS_PRESCAN__
#define MULTISPLIT_BMS_PRESCAN__

#ifndef NUM_WARPS_4
#define NUM_WARPS_4 4
#endif
#ifndef NUM_WARPS_8
#define NUM_WARPS_8 8
#endif

template<
	uint32_t		NUM_ROLLS,
	uint32_t 		NUM_BUCKETS,
	uint32_t		LOG_BUCKETS,
	typename 		bucket_t,
	typename 		KeyT>
__global__ void 
BMS_prescan_256(
	KeyT* 			d_key_in,
	uint32_t 		num_elements,
	uint32_t* 	d_histogram, 
	bucket_t 		bucket_identifier)
{
	// NUM_ROLL: number of consecutive windows read by one warp
	// currently assume no scaling factor between prescan and post scan. Next: scaling those.

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_8 * NUM_BUCKETS)];

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter = 0;

	if(blockIdx.x == (gridDim.x - 1))
	{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + \
				((warpId * NUM_ROLLS) << 5) + \
				(i_roll << 5) + \
				laneId;
			KeyT input_key = 	(temp_address < num_elements)?d_key_in[temp_address]:0xFFFFFFFF;
			uint32_t myBucket = 	(temp_address < num_elements)?bucket_identifier(input_key):(NUM_BUCKETS - 1);

			uint32_t myHisto = 0xFFFFFFFF;
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			binCounter  += __popc(myHisto);
		}
	}
	else{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			KeyT input_key = d_key_in[\
				(blockIdx.x * blockDim.x * NUM_ROLLS) + \
				((warpId * NUM_ROLLS) << 5) + \
				(i_roll << 5) + \
				laneId];
			uint32_t myBucket = bucket_identifier(input_key);

			uint32_t myHisto = 0xFFFFFFFF;
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			binCounter  += __popc(myHisto);
		}
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
		if((laneId & 0x07) == 0) // bucket -> block
		{
			d_histogram[((warpId << 2) + (laneId >> 3)) * gridDim.x + 
				blockIdx.x] = reduction;
		}
	}
}
//==================================
template<
	uint32_t		NUM_ROLLS,
	uint32_t		NUM_BUCKETS,
	uint32_t		LOG_BUCKETS,
	typename 		bucket_t,
	typename 		KeyT>
__global__ void 
BMS_prescan_128(
	KeyT* 			d_key_in,
	uint32_t 		num_elements,
	uint32_t* 	d_histogram, 
	bucket_t 		bucket_identifier)
{
	// NUM_ROLL: number of consecutive windows read by one warp
	// currently assume no scaling factor between prescan and post scan. Next: scaling those.

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_4 * NUM_BUCKETS)];

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter = 0;

	if(blockIdx.x == (gridDim.x - 1))
	{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + \
				((warpId * NUM_ROLLS) << 5) + \
				(i_roll << 5) + \
				laneId;
			KeyT input_key = (temp_address < num_elements)?d_key_in[temp_address]:0xFFFFFFFF;
			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key):(NUM_BUCKETS - 1);

			uint32_t myHisto = 0xFFFFFFFF;
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			binCounter  += __popc(myHisto);
		}
	}
	else{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			KeyT input_key = d_key_in[\
				(blockIdx.x * blockDim.x * NUM_ROLLS) + \
				((warpId * NUM_ROLLS) << 5) + \
				(i_roll << 5) + \
				laneId];
			uint32_t myBucket = bucket_identifier(input_key);

			uint32_t myHisto = 0xFFFFFFFF;
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			binCounter  += __popc(myHisto);
		}
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
		if((laneId & 0x03) == 0)
			d_histogram[((warpId << 3) + (laneId >> 2)) * gridDim.x + 
				blockIdx.x] = reduction;			
	}
}
//==================================
template<
	uint32_t		NUM_ROLLS,
	typename 		bucket_t,
	typename 		KeyT>
__global__ void 
BMS_prescan_64bucket_256(
	KeyT* 			d_key_in,
	uint32_t 		num_elements,
	uint32_t* 	d_histogram, 
	bucket_t 		bucket_identifier)
{
	// NUM_ROLL: number of consecutive windows read by one warp
	// currently assume no scaling factor between prescan and post scan. Next: scaling those.

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_8 * 32)]; // we put each histogram results into 16bit memory units.

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter = 0;

	if(blockIdx.x == (gridDim.x - 1))
	{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + \
				((warpId * NUM_ROLLS) << 5) + \
				(i_roll << 5) + \
				laneId;
			KeyT input_key = (temp_address < num_elements)?d_key_in[temp_address]:0xFFFFFFFF;
			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key):63;

			uint32_t myHisto_lo = 0xFFFFFFFF; // for bucket 0-31
			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_lo = myHisto_lo & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			// for the last bit:
			uint32_t rx_buffer = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t myHisto_hi = myHisto_lo & rx_buffer;
			myHisto_lo = myHisto_lo & (~rx_buffer);
			binCounter  += (__popc(myHisto_lo) + (__popc(myHisto_hi) << 16)); // (histo_32_63, histo_0_31)
		}
	}
	else{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			KeyT input_key = d_key_in[
				(blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId];
			uint32_t myBucket = bucket_identifier(input_key);

			uint32_t myHisto_lo = 0xFFFFFFFF; // for bucket 0-31
			#pragma unroll
			for(int i = 0; i<5; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto_lo = myHisto_lo & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			// for the last bit:
			uint32_t rx_buffer = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t myHisto_hi = myHisto_lo & rx_buffer;
			myHisto_lo = myHisto_lo & (~rx_buffer);
			binCounter  += (__popc(myHisto_lo) + (__popc(myHisto_hi) << 16)); // (histo_32_63, histo_0_31)
		}
	}
	// aggregating the results from all warps:
	// == storing results into smem: bucket -> warp
	smem_histogram[laneId * NUM_WARPS_8 + warpId] = binCounter;
	__syncthreads();

	// segmented reduction stage:
	uint32_t reduction = smem_histogram[threadIdx.x];
	reduction += __shfl_xor(reduction, 4, 32);
	reduction += __shfl_xor(reduction, 2, 32);
	reduction += __shfl_xor(reduction, 1, 32);
	if((laneId & 0x07) == 0) // bucket -> block
	{
		d_histogram[((warpId << 2) + (laneId >> 3)) * gridDim.x + 
			blockIdx.x] = (reduction & 0x0000FFFF);
	}
	else if((laneId & 0x07) == 1)
	{
		d_histogram[((warpId << 2) + (laneId >> 3) + 32) * gridDim.x + 
			blockIdx.x] = (reduction >> 16);		
	}
}
//====================================
template<
	uint32_t		NUM_ROLLS,
	typename 		bucket_t,
	typename 		KeyT>
__global__ void 
BMS_prescan_128bucket_256(
	KeyT* 			d_key_in,
	uint32_t 		num_elements,
	uint32_t* 	d_histogram, 
	bucket_t 		bucket_identifier)
{
	// NUM_ROLL: number of consecutive windows read by one warp
	// currently assume no scaling factor between prescan and post scan. Next: scaling those.

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_8 * 64)]; // we put each histogram results into 16bit memory units.

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)
	uint32_t binCounter_34 = 0; // for (16bit: 96-127, 	16bit: 64-95)

	if(blockIdx.x == (gridDim.x - 1))
	{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + \
				((warpId * NUM_ROLLS) << 5) + \
				(i_roll << 5) + \
				laneId;
			KeyT input_key = (temp_address < num_elements)?__ldg(&d_key_in[temp_address]):0xFFFFFFFF;
			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key):127;
			
			uint32_t rx_buffer_1 = __ballot(myBucket & 0x01);
			uint32_t myHisto_1 = ((laneId & 0x01)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x02);
			myHisto_1 &= ((laneId & 0x02)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x04);
			myHisto_1 &= ((laneId & 0x04)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x08);
			myHisto_1 &= ((laneId & 0x08)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x10);
			myHisto_1 &= ((laneId & 0x10)?rx_buffer_1:(~rx_buffer_1));

			rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit

			uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2); 	// for 32-63
			uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2; 	// for 64-95
			uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2;			// for 96-127
			myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2);				// for 0-31

			binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
			binCounter_34  += (__popc(myHisto_3) + (__popc(myHisto_4) << 16)); // (histo_96_127, histo_64_95)
		}
	}
	else{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			KeyT input_key = __ldg(&d_key_in[
				(blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId]);
			uint32_t myBucket = bucket_identifier(input_key);
			
			uint32_t rx_buffer_1 = __ballot(myBucket & 0x01);
			uint32_t myHisto_1 = ((laneId & 0x01)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x02);
			myHisto_1 &= ((laneId & 0x02)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x04);
			myHisto_1 &= ((laneId & 0x04)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x08);
			myHisto_1 &= ((laneId & 0x08)?rx_buffer_1:(~rx_buffer_1));
			rx_buffer_1 = __ballot(myBucket & 0x10);
			myHisto_1 &= ((laneId & 0x10)?rx_buffer_1:(~rx_buffer_1));

			rx_buffer_1 = __ballot(myBucket & 0x20); // checking the 6th bit
			uint32_t rx_buffer_2 = __ballot(myBucket & 0x40); // checking the 7th bit

			uint32_t myHisto_2 = myHisto_1 & rx_buffer_1 & (~rx_buffer_2); 	// for 32-63
			uint32_t myHisto_3 = myHisto_1 & (~rx_buffer_1) & rx_buffer_2; 	// for 64-95
			uint32_t myHisto_4 = myHisto_1 & rx_buffer_1 & rx_buffer_2;			// for 96-127
			myHisto_1 = myHisto_1 & (~rx_buffer_1) & (~rx_buffer_2);				// for 0-31

			binCounter_12  += (__popc(myHisto_1) + (__popc(myHisto_2) << 16)); // (histo_32_63, histo_0_31)
			binCounter_34  += (__popc(myHisto_3) + (__popc(myHisto_4) << 16)); // (histo_96_127, histo_64_95)
		}
	}
	// aggregating the results from all warps:
	// == storing results into smem: bucket -> warp
	smem_histogram[threadIdx.x] = binCounter_12;
	smem_histogram[256 + threadIdx.x] = binCounter_34;
	__syncthreads();
	if(warpId <= 1)
	{
		uint32_t reduction = smem_histogram[(warpId << 8) + laneId];
		reduction += smem_histogram[(warpId << 8) + laneId + 32];
		reduction += smem_histogram[(warpId << 8) + laneId + 64];
		reduction += smem_histogram[(warpId << 8) + laneId + 96];
		reduction += smem_histogram[(warpId << 8) + laneId + 128];
		reduction += smem_histogram[(warpId << 8) + laneId + 160];
		reduction += smem_histogram[(warpId << 8) + laneId + 192];
		reduction += smem_histogram[(warpId << 8) + laneId + 224];
		d_histogram[((warpId << 6) + laneId) * gridDim.x + blockIdx.x] = reduction & 0x0000FFFF;
		d_histogram[((warpId << 6) + laneId + 32) * gridDim.x + blockIdx.x] = reduction >> 16;
	}
}
//================================================
template<
	uint32_t		NUM_ROLLS,
	typename 		bucket_t,
	typename 		KeyT>
__global__ void 
BMS_prescan_256bucket_256(
	KeyT* 			d_key_in,
	uint32_t 		num_elements,
	uint32_t* 	d_histogram, 
	bucket_t 		bucket_identifier)
{
	// NUM_ROLL: number of consecutive windows read by one warp
	// currently assume no scaling factor between prescan and post scan. Next: scaling those.

	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem_histogram[(NUM_WARPS_8 * 128)]; // we put each histogram results into 16bit memory units.

	// Histogramming stage: each warp histogramming NUM_ROLL consecutive windows and aggregate their results.
	uint32_t binCounter_12 = 0; // for (16bit: 32-63, 	16bit: 0-31)
	uint32_t binCounter_34 = 0; // for (16bit: 96-127, 	16bit: 64-95)
	uint32_t binCounter_56 = 0; // for (16bit: 160-191, 	16bit: 128-159)	
	uint32_t binCounter_78 = 0; // for (16bit: 224-255, 	16bit: 192-223)	

	if(blockIdx.x == (gridDim.x - 1))
	{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			uint32_t temp_address = (blockIdx.x * blockDim.x * NUM_ROLLS) + \
				((warpId * NUM_ROLLS) << 5) + \
				(i_roll << 5) + \
				laneId;
			KeyT input_key = (temp_address < num_elements)?d_key_in[temp_address]:0xFFFFFFFF;
			uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key):255;

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
	}
	else{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS; i_roll++)
		{
			KeyT input_key = d_key_in[
				(blockIdx.x * blockDim.x * NUM_ROLLS) + 
				((warpId * NUM_ROLLS) << 5) + 
				(i_roll << 5) + 
				laneId];
			uint32_t myBucket = bucket_identifier(input_key);

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
	}
	// aggregating the results from all warps:
	// == storing results into smem: bucket -> warp
	smem_histogram[laneId * NUM_WARPS_8 + warpId] = binCounter_12;
	smem_histogram[256 + laneId * NUM_WARPS_8 + warpId] = binCounter_34;
	smem_histogram[512 + laneId * NUM_WARPS_8 + warpId] = binCounter_56;
	smem_histogram[768 + laneId * NUM_WARPS_8 + warpId] = binCounter_78;
	__syncthreads();

	// segmented reduction stage:
	uint32_t reduction_12 = smem_histogram[threadIdx.x];
	reduction_12 += __shfl_xor(reduction_12, 4, 32);
	reduction_12 += __shfl_xor(reduction_12, 2, 32);
	reduction_12 += __shfl_xor(reduction_12, 1, 32);

	uint32_t reduction_34 = smem_histogram[256 + threadIdx.x];
	reduction_34 += __shfl_xor(reduction_34, 4, 32);
	reduction_34 += __shfl_xor(reduction_34, 2, 32);
	reduction_34 += __shfl_xor(reduction_34, 1, 32);

	uint32_t reduction_56 = smem_histogram[512 + threadIdx.x];
	reduction_56 += __shfl_xor(reduction_56, 4, 32);
	reduction_56 += __shfl_xor(reduction_56, 2, 32);
	reduction_56 += __shfl_xor(reduction_56, 1, 32);

	uint32_t reduction_78 = smem_histogram[768 + threadIdx.x];
	reduction_78 += __shfl_xor(reduction_78, 4, 32);
	reduction_78 += __shfl_xor(reduction_78, 2, 32);
	reduction_78 += __shfl_xor(reduction_78, 1, 32);

	if((laneId & 0x07) == 0) // bucket -> block
	{
		d_histogram[((warpId << 2) + (laneId >> 3)) * gridDim.x + 
			blockIdx.x] = (reduction_12 & 0x0000FFFF);
	}
	else if((laneId & 0x07) == 1)
	{
		d_histogram[((warpId << 2) + (laneId >> 3) + 32) * gridDim.x + 
			blockIdx.x] = (reduction_12 >> 16);		
	}
	else if((laneId & 0x07) == 2)
		d_histogram[((warpId << 2) + (laneId >> 3) + 64) * gridDim.x + blockIdx.x] = (reduction_34 & 0x0000FFFF);
	else if((laneId & 0x07) == 3)
		d_histogram[((warpId << 2) + (laneId >> 3) + 96) * gridDim.x + blockIdx.x] = (reduction_34 >> 16);
	else if((laneId & 0x07) == 4)
		d_histogram[((warpId << 2) + (laneId >> 3) + 128) * gridDim.x + blockIdx.x] = (reduction_56 & 0x0000FFFF);
	else if((laneId & 0x07) == 5)
		d_histogram[((warpId << 2) + (laneId >> 3) + 160) * gridDim.x + blockIdx.x] = (reduction_56 >> 16);
	else if((laneId & 0x07) == 6)
		d_histogram[((warpId << 2) + (laneId >> 3) + 192) * gridDim.x + blockIdx.x] = (reduction_78 & 0x0000FFFF);		
	else if((laneId & 0x07) == 7)
		d_histogram[((warpId << 2) + (laneId >> 3) + 224) * gridDim.x + blockIdx.x] = (reduction_78 >> 16);	
}
#endif