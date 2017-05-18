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

#ifndef BMS_API__
#define BMS_API__
#include <stdio.h>
#include <stdint.h>
#include "config/config_wms.h"

int subproblem_size_wms_key_only(int num_buckets, uint32_t& size_block){
	int size_sub_prob = 1;
	switch(num_buckets){
		case 2:
			size_sub_prob = 	32 * NUM_ROLLS_K_1 * NUM_TILES_K_1;
			size_block = 			size_sub_prob * NUM_WARPS_K_1;
		break;
		case 4:
			size_sub_prob = 	32 * NUM_ROLLS_K_2 * NUM_TILES_K_2;
			size_block = 			size_sub_prob * NUM_WARPS_K_2;
		break;
		case 8:
			size_sub_prob = 	32 * NUM_ROLLS_K_3 * NUM_TILES_K_3;
			size_block = 			size_sub_prob * NUM_WARPS_K_3;
		break;			
		case 16:
			size_sub_prob = 	32 * NUM_ROLLS_K_4 * NUM_TILES_K_4;
			size_block = 			size_sub_prob * NUM_WARPS_K_4;
		break;			
		case 32:
			size_sub_prob = 	32 * NUM_ROLLS_K_5 * NUM_TILES_K_5;
			size_block = 			size_sub_prob * NUM_WARPS_K_5;
		break;		
		default:
			printf("Warning: number of buckets not yet supported.\n");
		break;
	}
	return size_sub_prob;
}

int subproblem_size_wms_key_value(int num_buckets, uint32_t& size_block){
	int size_sub_prob = 1;
	switch(num_buckets){
		case 2:
			size_sub_prob = 	32 * NUM_ROLLS_KV_1 * NUM_TILES_KV_1;
			size_block = 			size_sub_prob * NUM_WARPS_KV_1;
		break;
		case 4:
			size_sub_prob = 	32 * NUM_ROLLS_KV_2 * NUM_TILES_KV_2;
			size_block = 			size_sub_prob * NUM_WARPS_KV_2;
		break;
		case 8:
			size_sub_prob = 	32 * NUM_ROLLS_KV_3 * NUM_TILES_KV_3;
			size_block = 			size_sub_prob * NUM_WARPS_KV_3;
		break;			
		case 16:
			size_sub_prob = 	32 * NUM_ROLLS_KV_4 * NUM_TILES_KV_4;
			size_block = 			size_sub_prob * NUM_WARPS_KV_4;
		break;			
		case 32:
			size_sub_prob = 	32 * NUM_ROLLS_KV_5 * NUM_TILES_KV_5;
			size_block = 			size_sub_prob * NUM_WARPS_KV_5;
		break;		
		default:
			printf("Warning: number of buckets not yet supported.\n");
		break;
	}
	return size_sub_prob;
}
#endif