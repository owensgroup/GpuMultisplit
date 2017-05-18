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

#ifndef MULTISPLIT_WMS_PRESCAN__
#define MULTISPLIT_WMS_PRESCAN__
template<
	uint32_t 		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	uint32_t 		NUM_BUCKETS_,
	uint32_t		LOG_BUCKETS_,
	typename 		bucket_t>
	__launch_bounds__(256)
__global__ void 
multisplit2_WMS_prescan(
	uint32_t* 	d_key_in,
	uint32_t 		num_elements,
	uint32_t* 	d_histogram, 
	bucket_t 		bucket_identifier)
{
	uint32_t laneId = threadIdx.x & 0x1F;
	uint32_t warpId = threadIdx.x >> 5;
	uint32_t binCounter = 0;

	for(int i_tile = 0; i_tile < NUM_TILES_; i_tile++)
	{
		#pragma unroll 
		for(int i_roll = 0; i_roll < NUM_ROLLS_; i_roll++)
		{
			uint32_t input_key = (d_key_in[\
				(blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
				(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
				(i_tile * NUM_ROLLS_) + \
				i_roll) << 5) + \
				laneId]);

			uint32_t myBucket = bucket_identifier(input_key);

			uint32_t myHisto = 0xFFFFFFFF;
			#pragma unroll
			for(int i = 0; i<LOG_BUCKETS_; i++)
			{
				uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
				myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
			}
			binCounter  += __popc(myHisto);
		}
	}

	// writing back results per warp into gmem:
	if(laneId < NUM_BUCKETS_)
	{
		d_histogram[(laneId * (blockDim.x >> 5) * gridDim.x) + ((blockDim.x >> 5) * blockIdx.x) + warpId] = binCounter;
	}
}
//================================================
template<
	uint32_t 		NUM_TILES_,
	uint32_t		NUM_ROLLS_,
	uint32_t 		NUM_BUCKETS_,
	uint32_t		LOG_BUCKETS_,
	typename 		bucket_t>
	__launch_bounds__(256)
__global__ void 
multisplit2_WMS_prescan_protected(
	uint32_t* 	d_key_in,
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

				uint32_t input_key = (temp_address < num_elements)?(d_key_in[temp_address]):0xFFFFFFFF;

				// it's safe if we put the invalid element into the last bucket
				uint32_t myBucket = (temp_address < num_elements)?bucket_identifier(input_key):(NUM_BUCKETS_ - 1);

				uint32_t myHisto = 0xFFFFFFFF;
				#pragma unroll
				for(int i = 0; i<LOG_BUCKETS_; i++)
				{
					uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
					myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				}
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
				uint32_t input_key = (d_key_in[\
					(blockIdx.x * blockDim.x * NUM_TILES_ * NUM_ROLLS_) + \
					(((warpId * NUM_TILES_ * NUM_ROLLS_) +  \
					(i_tile * NUM_ROLLS_) + \
					i_roll) << 5) + \
					laneId]);

				uint32_t myBucket = bucket_identifier(input_key);

				uint32_t myHisto = 0xFFFFFFFF;
				#pragma unroll
				for(int i = 0; i<LOG_BUCKETS_; i++)
				{
					uint32_t rx_buffer = __ballot((myBucket >> i) & 0x01);
					myHisto = myHisto & (((laneId >> i) & 0x01)?rx_buffer:(~rx_buffer));
				}
				binCounter  += __popc(myHisto);
			}
		}
	}
	// writing back results per warp into gmem:
	if(laneId < NUM_BUCKETS_)
	{
		d_histogram[(laneId * (blockDim.x >> 5) * gridDim.x) + ((blockDim.x >> 5) * blockIdx.x) + warpId] = binCounter;
	}
}
#endif