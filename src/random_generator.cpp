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

#include <random>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

void random_permute(uint32_t* input, uint32_t numElements)
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

//==============================================
unsigned int single_rnd_generator(uint32_t bucketId, uint32_t log_bucket, uint32_t mode, uint32_t delta = 1)
{
	uint32_t final_result = 0;
	if(mode == 0) // log_bucket bits from MSB are the bucket ID, the rest are the key
	{
		uint32_t result = static_cast<uint32_t>(rand() % (RAND_MAX >> log_bucket));
		final_result = (result & (0xFFFFFFFF>>log_bucket)) + (bucketId << (32-log_bucket));
	}
	else if(mode == 1) // key is randomly chosen between bucketId * delta and bucketId * (delta+1)
	{
		// useful for key/delta identification
		uint32_t temp = (rand() % delta) + (bucketId * delta);
		final_result = temp;
	}

	return final_result; 
}
//===============================================
void uniform_bucket_generator(uint32_t* input, uint32_t n, uint32_t num_buckets)
{
	// generating equal number of bucket IDs as input keys
	random_permute(input, n);

	for(int i = 0; i<n; i++){
		input[i] = (input[i] % num_buckets);
	}
}
//===============================================
void uniform_input_generator(uint32_t* input, uint32_t n, uint32_t num_buckets, uint32_t log_buckets, uint32_t mode, uint32_t delta = 1)
{
	std::default_random_engine gen(time(NULL));
	std::uniform_int_distribution<> dis(0, num_buckets-1);

	uint32_t *bins = new uint32_t[num_buckets];
	for(int i = 0; i<num_buckets; i++)
		bins[i] = 0;
	
	for(int i = 0; i<n; i++)
	{
		uint32_t number = dis(gen);
		input[i] = single_rnd_generator(number, log_buckets, mode, delta); 
		++bins[number];
	}

	// printf("Uniform random number generator:\n");
	uint32_t total_sum = 0;
	for(int j = 0; j<num_buckets; j++)
		total_sum += bins[j];

	// int count = 0;
	// for(int i = 0; i<num_buckets; i++)
	// {
	// 	printf("B%d: %.2f, ", i, float(bins[i])/float(total_sum));
	// 	count++;
	// 	if(count == 16){
	// 		printf("\n");
	// 		count = 0;
	// 	}
	// }
	// printf("\n");
	delete[] bins;
}
//===============================================
void binomial_input_generator(uint32_t* input, uint32_t n, uint32_t NUM_B, uint32_t LOG_B, uint32_t mode, uint32_t delta = 1)
{
	// n: total number of elements
	// m: number of buckets
	std::default_random_engine generator(time(NULL));
	std::binomial_distribution<unsigned int> distribution(NUM_B-1,0.5);

	uint *bins = new uint[NUM_B];
	for(int i = 0; i<NUM_B; i++)
		bins[i] = 0;

	for(int i =0; i<n; i++)
	{
		unsigned int number = distribution(generator);
		input[i] = single_rnd_generator(number, LOG_B, mode, delta); 
		++bins[number];
		// printf("number  = %d, input = %d, bucket = %d\n", number, (input[i] & mask), (input[i] >> (32-LOG_B)));
	}
	
	// printf("Binomial random number generator:\n");
	// int count = 0;
	// for(int i = 0; i<NUM_B; i++)
	// {
	// 	printf("B%d: %u, ", i, bins[i]);
	// 	count++;
	// 	if(count == 16){
	// 		printf("\n");
	// 		count = 0;
	// 	}
	// }
	// printf("\n");
	delete[] bins;
}
//==================================================
void hockey_stick_generator(uint32_t* input, uint32_t n, uint32_t  NUM_B, uint32_t LOG_B, double alpha, uint32_t mode, uint32_t delta = 1)
{
	// generating random numbers such that alpha percent are shared among all buckets, and (1-alpha) in one bucket
	std::default_random_engine gen(time(NULL));
	std::uniform_int_distribution<> dis(0, NUM_B-1);	

	uint32_t *bins = new uint32_t[NUM_B];
	for(int i = 0; i<NUM_B; i++)
		bins[i] = 0;

	uint32_t n1 = static_cast<uint32_t>(alpha * double(n));
	uint32_t n2 = n - n1;
	
	// first, shared items:
	for(int i = 0; i<n1; i++)
	{
		uint32_t number = dis(gen);
		input[i] = single_rnd_generator(number, LOG_B, mode, delta); 
		++bins[number];
	}

	// second, the stick:
	uint32_t number = dis(gen);
	for(int i = n1; i<n; i++)
	{
		input[i] = single_rnd_generator(number, LOG_B, mode, delta); 
		++bins[number];		
	}

	for(int i = 0; i < n; i++)
	{
		unsigned int rand1 = rand();
		unsigned int rand2 = (rand() << 15) + rand1;
		unsigned int swap = i + (rand2%(n-i));

		unsigned int temp = input[i];
		input[i] = input[swap];
		input[swap] = temp;
	}

	// printf("Hockey-stick random number generator, alpha = %f:\n", alpha);
	// int count = 0;
	// for(int i = 0; i<NUM_B; i++)
	// {
	// 	printf("B%d: %u, ", i, bins[i]);
	// 	count++;
	// 	if(count == 16){
	// 		printf("\n");
	// 		count = 0;
	// 	}
	// }
	// printf("\n");

	delete[] bins;
}
//==============================================
void random_input_generator(uint32_t* input, uint32_t n, uint32_t num_buckets, uint32_t log_buckets, uint32_t bucket_mode, uint32_t random_mode, uint32_t delta = 1, double alpha = 1.0)
{
	switch(bucket_mode){
		case 0: // uniform bucket distribution
			uniform_input_generator(input, n, num_buckets, log_buckets, random_mode, delta);
			break;
		case 1: // binomial bucket distribution
			binomial_input_generator(input, n, num_buckets, log_buckets, random_mode, delta);
			break;
		case 2: // hockey stick distribution
			hockey_stick_generator(input, n, num_buckets, log_buckets, alpha, random_mode, delta);
			break;
		case 3: // uniform bucket distribution, input keys = bucket Ids
			uniform_bucket_generator(input, n, num_buckets);
		break;
		default:
			printf("Wrong bucket distribution entered!\n");
			break;
	}
}
