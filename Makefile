#GpuMultisplit is the proprietary property of The Regents of the University of California ("The Regents") and is copyright © 2016 The Regents of the University of California, Davis campus. All Rights Reserved. 

#Redistribution and use in source and binary forms, with or without modification, are permitted by nonprofit educational or research institutions for noncommercial use only, provided that the following conditions are met:

#* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
#* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
#* The name or other trademarks of The Regents may not be used to endorse or promote products derived from this software without specific prior written permission.

#The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.

#THE SOFTWARE PROVIDED IS ON AN "AS IS" BASIS, AND THE REGENTS HAVE NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS. THE REGENTS SPECIFICALLY DISCLAIM ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES, INCLUDING BUT NOT LIMITED TO  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES, LOSS OF USE, DATA OR PROFITS, OR BUSINESS INTERRUPTION, HOWEVER CAUSED AND UNDER ANY THEORY OF LIABILITY WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#If you do not agree to these terms, do not download or use the software.  This license may be modified only in a writing signed by authorized signatory of both parties.

# For license information please contact copyright@ucdavis.edu re T11-005.


# Device ID 
DEVICE		:= 	0
# CUDA Architecture 
COMPUTE_CAPABILITY := 35
ARCH 			:= 	-arch=sm_$(COMPUTE_CAPABILITY)

CUB  			:= 	~/Softwares/cub-1.6.4
CUDA 			:= 	/usr/local/cuda-8.0
INCLUDES 	:= 	-I src/ -I $(CUDA)/include/ -I $(CUB)/
NVCC 			:= 	nvcc -O3 

# Block-wide Multisplit (BMS)
bms: obj/main_bms.o obj/random_generator.o src/kernels/bms/*.cuh src/main/main_bms.cu src/cpu_functions.h 
	$(NVCC) $(ARCH) -o bin/out_bms obj/main_bms.o obj/random_generator.o $(INCLUDES)

obj/main_bms.o: src/kernels/bms/*.cuh obj/random_generator.o src/main/main_bms.cu src/cpu_functions.h 
	$(NVCC) $(ARCH) -D DEVICE_ID__=$(DEVICE) -D COMPUTE__=$(COMPUTE_CAPABILITY) -c -o obj/main_bms.o src/main/main_bms.cu $(INCLUDES)

# Warp-wide Multisplit (WMS)
wms: obj/main_wms.o obj/random_generator.o src/kernels/wms/*.cuh src/main/main_wms.cu src/cpu_functions.h 
	$(NVCC) $(ARCH) -o bin/out_wms obj/main_wms.o obj/random_generator.o $(INCLUDES)

obj/main_wms.o: src/kernels/wms/*.cuh obj/random_generator.o src/main/main_wms.cu src/cpu_functions.h 
	$(NVCC) $(ARCH) -D DEVICE_ID__=$(DEVICE) -D COMPUTE__=$(COMPUTE_CAPABILITY) -c -o obj/main_wms.o src/main/main_wms.cu $(INCLUDES)

# Multisplit API (using BMS)
multisplit: obj/main_multisplit.o obj/random_generator.o src/kernels/bms/*.cuh src/main/main_multisplit.cu src/cpu_functions.h 
	$(NVCC) $(ARCH) -o bin/out_multisplit obj/main_multisplit.o obj/random_generator.o $(INCLUDES)

obj/main_multisplit.o: src/kernels/bms/*.cuh obj/random_generator.o src/main/main_multisplit.cu src/cpu_functions.h 
	$(NVCC) $(ARCH) -D DEVICE_ID__=$(DEVICE) -D COMPUTE__=$(COMPUTE_CAPABILITY) -c -o obj/main_multisplit.o src/main/main_multisplit.cu $(INCLUDES)

# Key-only radix sort 
sort: obj/main_sort.o src/api/*.cuh src/kernels/bms/*.cuh src/gpu_functions.cuh
	$(NVCC) $(ARCH) -o bin/out_sort obj/main_sort.o $(INCLUDES)

obj/main_sort.o: src/main/main_sort.cu src/api/*.cuh src/kernels/bms/*.cuh src/gpu_functions.cuh
	$(NVCC) $(ARCH) -D DEVICE_ID__=$(DEVICE) -D COMPUTE__=$(COMPUTE_CAPABILITY) -c -o obj/main_sort.o src/main/main_sort.cu $(INCLUDES)

# key-value radix sort:
sort_pairs: obj/main_sort_pairs.o src/api/*.cuh src/kernels/bms/*.cuh src/gpu_functions.cuh
	$(NVCC) $(ARCH) -o bin/out_sort_pairs obj/main_sort_pairs.o $(INCLUDES)

obj/main_sort_pairs.o: src/main/main_sort_pairs.cu src/api/*.cuh src/gpu_functions.cuh
	$(NVCC) $(ARCH) -D DEVICE_ID__=$(DEVICE) -D COMPUTE__=$(COMPUTE_CAPABILITY) -c -o obj/main_sort_pairs.o src/main/main_sort_pairs.cu $(INCLUDES)

# histogram simulations:
histogram: obj/main_histogram.o src/kernels/histogram/multisplit2_histograms.cuh
	$(NVCC) $(ARCH) -o bin/out_histogram obj/main_histogram.o $(INCLUDES)

obj/main_histogram.o: src/main/main_histogram.cu src/kernels/histogram/multisplit2_histograms.cuh
	$(NVCC) $(ARCH) -D DEVICE_ID__=$(DEVICE) -D COMPUTE__=$(COMPUTE_CAPABILITY) -c -o obj/main_histogram.o src/main/main_histogram.cu $(INCLUDES)	

# compaction:
compaction: obj/main_compaction.o src/kernels/compaction/multisplit2_compaction.cuh
	$(NVCC) $(ARCH) -o bin/out_compact obj/main_compaction.o $(INCLUDES)

obj/main_compaction.o: src/main/main_compaction.cu src/kernels/compaction/multisplit2_compaction.cuh src/cpu_functions.h
	$(NVCC) $(ARCH) -D DEVICE_ID__=$(DEVICE) -D COMPUTE__=$(COMPUTE_CAPABILITY) -c -o obj/main_compaction.o src/main/main_compaction.cu $(INCLUDES)

# our simple random generator
obj/random_generator.o: src/random_generator.cpp
	g++ -std=c++11 -c src/random_generator.cpp -o obj/random_generator.o

new:
	mkdir bin
	mkdir obj

clean: 
	rm -f bin/*
	rm -f obj/*.o