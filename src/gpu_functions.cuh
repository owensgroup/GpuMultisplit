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
#ifndef __GPU_FUNCTIONS__
#define __GPU_FUNCTIONS__
template<typename T>
__global__ void compare_vectors(T* d_array_1, T* d_array_2, uint32_t num_elements, uint32_t* num_diffs)
{
	uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid >= num_elements)
		return;

	if(d_array_1[tid] != d_array_2[tid])
		atomicAdd(num_diffs,1);
}
#endif