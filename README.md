# GpuMultisplit

## Abstract
Multisplit is a broadly useful parallel primitive that permutes its input data into contiguous _buckets_ or _bins_, where the function that categorizes an element into a bucket is provided by the programmer.
Due to the lack of an efficient multisplit on GPUs, programmers often choose to implement multisplit with a sort.
One way is to first generate an auxiliary array of bucket IDs and then sort input data based on it.
In case smaller indexed buckets possess smaller valued keys, another way for multisplit is to directly sort input data.
Both methods are inefficient and require more work than necessary: the former requires more expensive data movements while the latter spends unnecessary effort in sorting elements within each bucket.
In this work, we provide a parallel model and multiple implementations for the multisplit problem. Our principal focus is multisplit for a small (up to 256) number of buckets.
We use warp-synchronous programming models and emphasize warp-wide communications to avoid branch divergence and reduce memory usage.
We also hierarchically reorder input elements to achieve better coalescing of global memory accesses.
On a GeForce GTX 1080 GPU, we can reach a peak throughput of 18.93 Gkeys/s (or 11.68 Gpairs/s) for a  key-only (or key-value) multisplit.
Finally, we demonstrate how multisplit can be used as a building block for radix sort. In our multisplit-based sort implementation, we achieve comparable performance to the fastest GPU sort routines, sorting 32-bit keys (and key-value pairs) with a throughput of 3.0 G keys/s (and 2.1 Gpair/s).

## Publications
1. Saman Ashkiani, Andrew A. Davidson, Ulrich Meyer, John D. Owens. **GPU Multisplit: an extended study of a parallel algorithm**. To appear on _ACM Transactions on Parallel Computing, Special Issue: Invited papers from PPoPP 2016_. September 2017. Preprint: https://arxiv.org/abs/1701.01189
2. Saman Ashkiani, Andrew A. Davidson, Ulrich Meyer, John D. Owens. **GPU Multisplit**. _In Proceedings of the 21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP 2016)_. 12:1 - 12:13. 
DOI: http://dx.doi.org/10.1145/2851141.2851169 

## What does this code include?
1. **Warp-wide Multisplit (WMS)**: This version of multisplit is suitable for small number of buckets up to 32. It operates independently on all warps within a thread-block without any intra-block synchronizations. [[test code](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_wms.cu)][[kernels](https://github.com/owensgroup/GpuMultisplit/tree/master/src/kernels/wms)] 
2. **Block-wide Multisplit (BMS)**: This version is suitable for up to 256 buckets. Warps within a thread-block cooperate with each other to perform the multisplit. [[test code](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_bms.cu)][[kernels](https://github.com/owensgroup/GpuMultisplit/tree/master/src/kernels/bms)]
3. **Multisplit**: A simplified API for multisplit (it uses BMS), supporting up to 256 buckets [[multisplit test](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_multisplit.cu)][[API](https://github.com/owensgroup/GpuMultisplit/blob/master/src/api/multisplit.cuh)]. For more buckets (> 256), you should refer to the Reduced-bit sort method, as shown here [[RB-sort example](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_wms.cu#L534)]. 
4. **Multisplit-sort**: We use our BMS method to iteratively sort consecutive bits of an input element, i.e., building a radix sort. Our sort is competetive to CUB, especially when dealing with key-value scenarios. [[key-only](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_sort.cu)][[key-value](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_sort_pairs.cu)][[kernels](https://github.com/owensgroup/GpuMultisplit/blob/master/src/api/multisplit_sort.cuh)]   
5. **Multisplit-histogram**: Modifying our WMS's prescan stage, we can implement a simple single-channel device-wide histogram suitable for up to 256 bins [[test code](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_histogram.cu)][[kernels](https://github.com/owensgroup/GpuMultisplit/tree/master/src/kernels/histogram)]  
6. **Multisplit-compaction**: slightly modified version of our WMS method just for two buckets. [[test code](https://github.com/owensgroup/GpuMultisplit/blob/master/src/main/main_compaction.cu)][[kernels](https://github.com/owensgroup/GpuMultisplit/blob/master/src/kernels/compaction/multisplit2_compaction.cuh)]
## How to use this code?
1. Set the CUB directory accordingly in the Makefile [[CUB](https://github.com/NVlabs/cub)]
2. Set the DEVICE and COMPUTE_CAPABILITY in the Makefile
3. `make new`
4. `make <label>`: For example, for running BMS test files `make bms`.
5. binary files will be stored in bin/: For example, for running key-only bms over 10 iterations (mode 1 in the main_bms.cu) we run: `./bin/out_bms -mode 1 -iter 10`

## How to use this code in another project?
`GpuMultisplit/src/` should be added to the build directory.
#### Multisplit
Example of using Multisplit in a code:
  ```
  #include "api/multisplit.cuh"
  
  // Initializing the multisplit:
  multisplit_context ms_context(num_buckets);
  multisplit_allocate_key_only(num_elements, ms_context);
  
  // key-only multisplit: 
  multisplit_key_only(d_key_in, d_key_out, num_elements, ms_context, bucket_identifier);
  
  // releasing the allocated memory
  multisplit_release_memory(ms_context);	
  
  ```
#### Multisplit-sort:  
```
#include api/multisplit_sort.cuh

// a radix sort with 7-bit radixes:
ms_sort_context sort_context;
multisplit_sort_7bit_allocate(num_elements, sort_context);

// sorting:
multisplit_sort_7bit(d_key_in, d_key_out, num_elements, sort_context);

// releasing memory:
multisplit_sort_release_memory(sort_context);

```
## Reporting problems 
To report bugs, please file an issue [here](https://github.com/owensgroup/GpuMultisplit/issues). 
## Developer:
* [Saman Ashkiani](http://www.ece.ucdavis.edu/~ashkiani/), University of california, Davis.
