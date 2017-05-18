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

#ifndef __REDUCED_BIT_SORT
#define __REDUCED_BIT_SORT
#define NUM_THREADS_REDUCED 256
#include <stdint.h>

template <typename bucket_t>
__global__ void markBins_general(uint32_t* d_buckets_out, uint32_t* d_key_in, uint32_t numElements, uint32_t numBuckets, bucket_t bucket_identifier)
{
	uint32_t myId = threadIdx.x + blockIdx.x*blockDim.x;
	uint32_t offset = blockDim.x*gridDim.x;

	for(int i = myId; i < numElements; i+=offset)
	{
		uint32_t input_key = d_key_in[i];
		uint32_t input_bucket = bucket_identifier(input_key);
		d_buckets_out[i] = input_bucket;
	}
}

__global__ void packingKeyValuePairs(uint64_t* packed, uint32_t* input_key, uint32_t* input_value, uint32_t numElements)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= numElements) return;

	uint32_t myKey 		= input_key[tid];
	uint32_t myValue 	= input_value[tid];
	uint64_t output = (static_cast<uint64_t>(myKey) << 32) + static_cast<uint64_t>(myValue);
	packed[tid] = output;
}
//===========================================
__global__ void unpackingKeyValuePairs(uint64_t* packed, uint32_t* out_key, uint32_t* out_value, uint32_t numElements)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= numElements) return;

	uint64_t myPacked = packed[tid];
	out_value[tid] = static_cast<uint32_t>(myPacked & 0x00000000FFFFFFFF);	
	out_key[tid] = static_cast<uint32_t>(myPacked >> 32);	
}
#endif