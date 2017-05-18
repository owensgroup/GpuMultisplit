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

#ifndef CPU_FUNCTIONS__
#define CPU_FUNCTIONS__
#include <algorithm>
inline char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
	char ** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}
//=======================
inline bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}
//========================
void randomPermute(uint32_t* input, uint32_t numElements)
{
	//Uses knuth's method to randomly permute
	for(int i = 0; i < numElements; i++)
		input[i] = i; //rand() + rand() << 15;

	for(int i = 0; i < numElements; i++)
	{
		uint32_t rand1 = rand();
		uint32_t rand2 = (rand() << 15) + rand1;
		uint32_t swap = i + (rand2%(numElements-i));

		uint32_t temp = input[i];
		input[i] = input[swap];
		input[swap] = temp;
	}
}
//==============================
template<typename bucket_t>
void cpu_prescan_test(uint32_t* key_input, uint32_t n, uint32_t* gpu_histogram, uint32_t num_histo_sub_prob, bucket_t bucket_identifier, int print_mode, uint32_t num_buckets, uint32_t num_roll, uint32_t num_warp)
{
	uint32_t* bins = new uint32_t[num_buckets];
	uint32_t* cpu_histogram = new uint32_t[num_buckets];
	for(int k = 0; k<num_buckets; k++)
		cpu_histogram[k] = 0;

	uint32_t size_effective_sub_prob = 32*num_roll*num_warp;
	bool valid = true;

	for(int i = 0; i<num_histo_sub_prob && valid; i++)
	{
		for(int k = 0; k<num_buckets; k++)
			bins[k] = 0;

		for(int j = 0; j<size_effective_sub_prob; j++)
		{
			uint32_t myKey = key_input[size_effective_sub_prob * i + j];
			uint32_t myBucket = bucket_identifier(myKey);
			bins[myBucket]++;
		}

		for(int k = 0; k<num_buckets && valid; k++)
		{
			if(gpu_histogram[k * num_histo_sub_prob + i] != bins[k])
				valid = false;		
		}
		if(!valid)
		{
			printf("Input keys starting from sub problem %d:\n", i);
			for(int k = 0; k<size_effective_sub_prob; k++)
			{
				if((k % 32) == 0) printf("\n");
				printf("%d, ", key_input[i*size_effective_sub_prob + k]);
			}
			printf("\n");
			for(int k = 0; k<num_buckets;k++)
				printf("starting from %d: %d:(gpu %d, cpu %d)\n", i, k, gpu_histogram[k * num_histo_sub_prob + i], bins[k]);
		}
	}
	if(valid)
		printf("Validation ran successfully!\n");
	else 
		printf("Validation failed!\n");

	delete[] cpu_histogram;
	delete[] bins;
}

template<typename bucket_t>
void cpu_prescan_test_fusion(uint32_t* key_input, uint32_t n, uint32_t* gpu_histogram, uint32_t num_histo_sub_prob, bucket_t bucket_identifier, int print_mode, uint32_t num_buckets, uint32_t num_roll, uint32_t num_warps, uint32_t fusion_factor)
{
	uint32_t* bins = new uint32_t[num_buckets];
	uint32_t* cpu_histogram = new uint32_t[num_buckets];
	for(int k = 0; k<num_buckets; k++)
		cpu_histogram[k] = 0;

	uint32_t size_effective_sub_prob = 32*num_roll*(1<<fusion_factor);
	bool valid = true;

	for(int i = 0; i<num_histo_sub_prob && valid; i++)
	{
		for(int k = 0; k<num_buckets; k++)
			bins[k] = 0;

		for(int j = 0; j<size_effective_sub_prob; j++)
		{
			uint32_t myKey = key_input[size_effective_sub_prob * i + j];
			uint32_t myBucket = bucket_identifier(myKey);
			bins[myBucket]++;
		}

		for(int k = 0; k<num_buckets && valid; k++)
		{
			if(gpu_histogram[k * num_histo_sub_prob + i] != bins[k])
				valid = false;		
		}
		if(!valid)
		{
			printf("Input keys starting from sub problem %d:\n", i);
			for(int k = 0; k<size_effective_sub_prob; k++)
			{
				if((k % 32) == 0) printf("\n");
				printf("%d, ", key_input[i*size_effective_sub_prob + k]);
			}
			printf("\n");
			for(int k = 0; k<num_buckets;k++)
				printf("starting from %d: %d:(gpu %d, cpu %d)\n", i, k, gpu_histogram[k * num_histo_sub_prob + i], bins[k]);
		}
	}
	if(valid)
		printf("Validation ran successfully!\n");
	else 
		printf("Validation failed!\n");

	delete[] cpu_histogram;
	delete[] bins;
}

template <typename bucket_t>
void cpu_multisplit_general(uint32_t* key_input, uint32_t* key_output, uint32_t n, bucket_t bucket_identifier, int print_mode, uint32_t num_buckets)
{
	// Performs the mutlisplit with arbitrary bucket distribution on cpu:
	// n: number of elements
	
	uint32_t *bins = new uint32_t[num_buckets]; // histogram results holder
	uint32_t *scan_bins = new uint32_t[num_buckets];
	uint32_t *current_idx = new uint32_t[num_buckets];
	// Computing histograms:
	uint32_t bucketId;
	
	for(int k = 0; k<num_buckets; k++)
		bins[k] = 0;

	for(int i = 0; i<n ; i++)
	{
		bucketId = bucket_identifier(key_input[i]);
		bins[bucketId]++;
	}
	if(print_mode){
		printf("Histograms:\n");
		for(int i = 0; i<num_buckets; i++)
			printf("%d ", bins[i]);
		printf("\n");
	}

	// computing exclusive scan operation on the inputs: 
	scan_bins[0] = 0;
	for(int j = 1; j<num_buckets; j++)
		scan_bins[j] = scan_bins[j-1] + bins[j-1];
	if(print_mode){
		printf("Scan Histograms:\n");
		for(int i = 0; i<num_buckets; i++)
			printf("%d ", scan_bins[i]);
		printf("\n");
	}
	// Placing items in their new positions:
	for(int k = 0; k<num_buckets; k++)
		current_idx[k] = 0;

	for(int i = 0; i<n; i++)
	{
		bucketId = bucket_identifier(key_input[i]);
		key_output[scan_bins[bucketId] + current_idx[bucketId]] = key_input[i];
		current_idx[bucketId]++;
	}

	if(print_mode){
		printf("Key Output\n");
		uint32_t count = 0;
		for(int i = 0; i<n; i++){
			printf("%d ", key_output[i]);
			count++;
			if(count >= 32){
				count = 0;
				printf("\n");
			}
		}
	}
	printf("\n");
	// releasing memory:
	delete[] bins;
	delete[] scan_bins;
	delete[] current_idx;
}

template <typename bucket_t>
void cpu_multisplit_pairs_general(uint32_t* key_input, uint32_t* key_output, uint32_t* value_input, uint32_t* value_output, uint32_t n, bucket_t bucket_identifier, int print_mode, uint32_t num_buckets)
{
	// Performs the mutlisplit with arbitrary bucket distribution on cpu:
	// n: number of elements
	
	uint32_t *bins = new uint32_t[num_buckets]; // histogram results holder
	uint32_t *scan_bins = new uint32_t[num_buckets];
	uint32_t *current_idx = new uint32_t[num_buckets];
	// Computing histograms:
	uint32_t bucketId;
	
	for(int k = 0; k<num_buckets; k++)
		bins[k] = 0;

	for(int i = 0; i<n ; i++)
	{
		bucketId = bucket_identifier(key_input[i]);
		bins[bucketId]++;
	}
	if(print_mode){
		printf("Histograms:\n");
		for(int i = 0; i<num_buckets; i++)
			printf("%d ", bins[i]);
		printf("\n");
	}

	// computing exclusive scan operation on the inputs: 
	scan_bins[0] = 0;
	for(int j = 1; j<num_buckets; j++)
		scan_bins[j] = scan_bins[j-1] + bins[j-1];
	if(print_mode){
		printf("Scan Histograms:\n");
		for(int i = 0; i<num_buckets; i++)
			printf("%d ", scan_bins[i]);
		printf("\n");
	}
	// Placing items in their new positions:
	for(int k = 0; k<num_buckets; k++)
		current_idx[k] = 0;

	for(int i = 0; i<n; i++)
	{
		bucketId = bucket_identifier(key_input[i]);
		key_output[scan_bins[bucketId] + current_idx[bucketId]] = key_input[i];
		value_output[scan_bins[bucketId] + current_idx[bucketId]] = value_input[i];
		current_idx[bucketId]++;
	}

	if(print_mode){
		printf("Key Output\n");
		uint32_t count = 0;
		for(int i = 0; i<n; i++){
			printf("%d ", key_output[i]);
			count++;
			if(count >= 32){
				count = 0;
				printf("\n");
			}
		}
	}
	printf("\n");
	// releasing memory:
	delete[] bins;
	delete[] scan_bins;
	delete[] current_idx;
}

template <typename T>
void printGPUArray(T *d_inp, unsigned int numElements, int elsPerRow = 10)
{

	T* cpu_inp = (T*) malloc(sizeof(T)*numElements);

	cudaMemcpy(cpu_inp, d_inp, sizeof(T)*numElements, cudaMemcpyDeviceToHost);
	for(int i = 0; i < numElements; i++)
	{
		std::cout << cpu_inp[i] << ' ';
		if((i+1)%elsPerRow == 0)
			std::cout<<"\n";
	}
	std::cout<<"\n";
	free(cpu_inp);
}
#endif