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
#ifndef MULTISPLIT_WMS_POSTSCAN__
#define MULTISPLIT_WMS_POSTSCAN__
template<
	uint32_t 		NUM_WARPS_,
	uint32_t		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	uint32_t 		NUM_BUCKETS_,
	uint32_t		LOG_BUCKETS_,
	typename 		bucket_t>
__global__ void 
multisplit2_WMS_postscan_4rolls(
	const uint32_t* 	__restrict__ d_key_in,
	uint32_t* 	__restrict__ d_key_out,
	uint32_t 		num_elements,
	const uint32_t* 	__restrict__ d_histogram, 
	bucket_t 		bucket_identifier)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem[(32 * NUM_WARPS_ * NUM_ROLLS_)];
	uint32_t *keys_ms_smem = 						smem;

	uint32_t input_key[NUM_ROLLS_]; 				// stores all keys regarding to this thread

	// ==== Loading back histogram results from gmem into smem:
	uint32_t global_offset = 0;
	if(laneId < NUM_BUCKETS_) // warp -> bucket (uncoalesced gmem access)
		global_offset = __ldg(&d_histogram[laneId * gridDim.x * NUM_WARPS_ + blockIdx.x * NUM_WARPS_ + warpId]);

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

			uint32_t myHisto = 0xFFFFFFFF;
			uint32_t myIndexBmp = 0xFFFFFFFF;

			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS_; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
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
		#pragma unroll
		for(int i = 0; i<LOG_BUCKETS_; i++)
		{
			uint32_t temp_scan = __shfl_up(scanned_binCount, 1<<i, 32);
			if(laneId >= (1<<i)) scanned_binCount += temp_scan;
		}
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
			uint32_t temp_input = keys_ms_smem[laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5)];
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
//====================================
template<
	uint32_t 		NUM_WARPS_,
	uint32_t		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	uint32_t 		NUM_BUCKETS_,
	uint32_t		LOG_BUCKETS_,
	typename 		bucket_t>
__global__ void 
multisplit2_WMS_postscan_4rolls_protected(
	const uint32_t* 	__restrict__ d_key_in,
	uint32_t* 	__restrict__ d_key_out,
	uint32_t 		num_elements,
	const uint32_t* 	__restrict__ d_histogram, 
	bucket_t 		bucket_identifier)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	__shared__ uint32_t smem[(32 * NUM_WARPS_ * NUM_ROLLS_)];
	uint32_t *keys_ms_smem = 						smem;

	uint32_t input_key[NUM_ROLLS_]; 				// stores all keys regarding to this thread

	// ==== Loading back histogram results from gmem into smem:
	uint32_t global_offset = 0;
	if(laneId < NUM_BUCKETS_) // warp -> bucket (uncoalesced gmem access)
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

				uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(NUM_BUCKETS_ - 1);

				myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

				uint32_t myHisto = 0xFFFFFFFF;
				uint32_t myIndexBmp = 0xFFFFFFFF;

				#pragma unroll
				for(int i = 0; i<LOG_BUCKETS_; i++)
				{
					uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
					myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
					myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
				}
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
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS_; i++)
			{
				uint32_t temp_scan = __shfl_up(scanned_binCount, 1<<i, 32);
				if(laneId >= (1<<i)) scanned_binCount += temp_scan;
			}
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
				uint32_t temp_input = keys_ms_smem[temp_address_smem];
				uint32_t myBucket_temp = ((laneId + (i_roll<<5) + ((i_tile * NUM_ROLLS_) << 5) + ((warpId * NUM_TILES_ * NUM_ROLLS_)<<5) + blockIdx.x * ((NUM_WARPS_*NUM_ROLLS_*NUM_TILES_)<<5)) < num_elements)?bucket_identifier(temp_input):(NUM_BUCKETS_ - 1);
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

				uint32_t myHisto = 0xFFFFFFFF;
				uint32_t myIndexBmp = 0xFFFFFFFF;

				#pragma unroll
				for(int i = 0; i<LOG_BUCKETS_; i++)
				{
					uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
					myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
					myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
				}
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
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS_; i++)
			{
				uint32_t temp_scan = __shfl_up(scanned_binCount, 1<<i, 32);
				if(laneId >= (1<<i)) scanned_binCount += temp_scan;
			}
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
				uint32_t temp_input = keys_ms_smem[laneId + (i_roll << 5) +((warpId * NUM_ROLLS_) << 5)];
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
//==========================================
template<
	uint32_t 		NUM_WARPS_,
	uint32_t		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	uint32_t 		NUM_BUCKETS_,
	uint32_t		LOG_BUCKETS_,
	typename 		bucket_t>
__global__ void 
DMS_postscan_4rolls_protected(
	const uint32_t* 	__restrict__ d_key_in,
	uint32_t* 	__restrict__ d_key_out,
	uint32_t 		num_elements,
	const uint32_t* 	__restrict__ d_histogram, 
	bucket_t 		bucket_identifier)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;

	uint32_t input_key[NUM_ROLLS_]; 				// stores all keys regarding to this thread

	// ==== Loading back histogram results from gmem into smem:
	uint32_t global_offset = 0;
	if(laneId < NUM_BUCKETS_) // warp -> bucket (uncoalesced gmem access)
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

				uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key[i_roll]):(NUM_BUCKETS_ - 1);

				myBucket_list |= (myBucket << (i_roll << 3)); // each byte myBucket per roll

				uint32_t myHisto = 0xFFFFFFFF;
				uint32_t myIndexBmp = 0xFFFFFFFF;

				#pragma unroll
				for(int i = 0; i<LOG_BUCKETS_; i++)
				{
					uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
					myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
					myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
				}
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
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS_; i++)
			{
				uint32_t temp_scan = __shfl_up(scanned_binCount, 1<<i, 32);
				if(laneId >= (1<<i)) scanned_binCount += temp_scan;
			}
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

				// adding the global offset
				myNewBlockIndex += __shfl(global_offset, (myBucket_list & 0xFF));

				myBucket_list >>= 8;
				myLocalIndex_list >>= 8;

				if(myNewBlockIndex < num_elements)
					d_key_out[myNewBlockIndex] = input_key[i_roll];
			}
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

				uint32_t myHisto = 0xFFFFFFFF;
				uint32_t myIndexBmp = 0xFFFFFFFF;

				#pragma unroll
				for(int i = 0; i<LOG_BUCKETS_; i++)
				{
					uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
					myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
					myIndexBmp &= (((myBucket >> i) & 0x01)?rx_buffer:(~rx_buffer));
				}
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
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS_; i++)
			{
				uint32_t temp_scan = __shfl_up(scanned_binCount, 1<<i, 32);
				if(laneId >= (1<<i)) scanned_binCount += temp_scan;
			}
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

				// adding the global offset
				myNewBlockIndex += __shfl(global_offset, (myBucket_list & 0xFF));

				myBucket_list >>= 8;
				myLocalIndex_list >>= 8;

				d_key_out[myNewBlockIndex] = input_key[i_roll];
			}
			// updating the global offsets for next tile
			global_offset += binCounter;
		}
	}
}
#endif