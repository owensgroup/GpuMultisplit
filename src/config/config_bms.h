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

#ifndef CONFIG_BMS__
#define CONFIG_BMS__
	#if COMPUTE__ >= 60
// tested with sm_61 on GeForce GTX 1080
	// == key-only case:
		#define NUM_WARPS_K_8 8
		#define NUM_ROLLS_K_8 4

		#define NUM_WARPS_K_7 8
		#define NUM_ROLLS_K_7 4

		#define NUM_WARPS_K_6 8
		#define NUM_ROLLS_K_6 4

		#define NUM_WARPS_K_5 8
		#define NUM_ROLLS_K_5 4

		#define NUM_WARPS_K_4 4
		#define NUM_ROLLS_K_4 8

		#define NUM_WARPS_K_3 4
		#define NUM_ROLLS_K_3 8

		#define NUM_WARPS_K_2 4
		#define NUM_ROLLS_K_2 7

		#define NUM_WARPS_K_1 4
		#define NUM_ROLLS_K_1 7

		// == key-value case:
		#define NUM_WARPS_KV_8 8
		#define NUM_ROLLS_KV_8 4

		#define NUM_WARPS_KV_7 8
		#define NUM_ROLLS_KV_7 4

		#define NUM_WARPS_KV_6 8
		#define NUM_ROLLS_KV_6 4

		#define NUM_WARPS_KV_5 8
		#define NUM_ROLLS_KV_5 4

		#define NUM_WARPS_KV_4 8
		#define NUM_ROLLS_KV_4 4

		#define NUM_WARPS_KV_3 8
		#define NUM_ROLLS_KV_3 4

		#define NUM_WARPS_KV_2 8
		#define NUM_ROLLS_KV_2 4

		#define NUM_WARPS_KV_1 8
		#define NUM_ROLLS_KV_1 4
	#else // default case: (tested with sm_35 on K40c)
		// == key-only case:
		#define NUM_WARPS_K_8 8
		#define NUM_ROLLS_K_8 4

		#define NUM_WARPS_K_7 8
		#define NUM_ROLLS_K_7 4

		#define NUM_WARPS_K_6 8
		#define NUM_ROLLS_K_6 4

		#define NUM_WARPS_K_5 4
		#define NUM_ROLLS_K_5 8

		#define NUM_WARPS_K_4 4
		#define NUM_ROLLS_K_4 7

		#define NUM_WARPS_K_3 4
		#define NUM_ROLLS_K_3 7

		#define NUM_WARPS_K_2 4
		#define NUM_ROLLS_K_2 7

		#define NUM_WARPS_K_1 4
		#define NUM_ROLLS_K_1 6

		// == key-value case:
		#define NUM_WARPS_KV_8 8
		#define NUM_ROLLS_KV_8 4

		#define NUM_WARPS_KV_7 8
		#define NUM_ROLLS_KV_7 4

		#define NUM_WARPS_KV_6 8
		#define NUM_ROLLS_KV_6 4

		#define NUM_WARPS_KV_5 8
		#define NUM_ROLLS_KV_5 4

		#define NUM_WARPS_KV_4 8
		#define NUM_ROLLS_KV_4 4

		#define NUM_WARPS_KV_3 8
		#define NUM_ROLLS_KV_3 4

		#define NUM_WARPS_KV_2 8
		#define NUM_ROLLS_KV_2 4

		#define NUM_WARPS_KV_1 8
		#define NUM_ROLLS_KV_1 4
	#endif
#endif