## Parallelized String Art

### Author 
Catherine Yu (tianhony) Nanxi Li (nanxil1)

### Summary
We aim to parallelize the the problem of converting an input image to string art instructions on NVIDIA GPUs(using CUDA). 

### Background
First of all, what is string art? String art is a technique for the creation of visual artwork where images emerge from a set of strings that are spanned between pins. It is traditionally done with strings wounding around a grid of nails hammered into a velvet-covered wooden board. Here is a [video](https://vimeo.com/175653201) explaining how it is done.

![thread art](images/stringartsteps.png)
![thread art](images/vrellis.jpg)

We are going to use the algorithm presented by [Birsak et al](https://www.cg.tuwien.ac.at/research/publications/2018/Birsak2018-SA/Birsak2018-SA-preprint.pdf) in his research paper. This algorithm will not produce a solution using a single continuous thread, but many discrete edges. 

>Our algorithm computes a subset of all possible strings which are used together to reassemble the input image. In each iter ation we pick the edge that allows the biggest norm reduction, and we stop when further addition would cause an increase of the error.The big disadvantage of a greedy approach as we use it is that a decision at an early stage can later turn out to be a bad choice. In our case, an addition of one edge might bring the biggest benefit when it is chosen, but can then prevent a better solution later on. Thus, we try to further improve the results by an iterative removal of edges. In particular, if the initial addition stage terminates, we sequentially remove those edges that allow to improve the norm. If it is not possible anymore, we start to add edges again, pick one edge at a time, and continue as long as the norm can be improved. We alternate those two stages until it is not possible to further im- prove the norm, neither by removal nor by addition, in which case the algorithm terminates. (Birsak et al, 2018)

With `N` needles, there are `N*(N-1)` edges total(each needle can be connected with all `N-1` needles, so each iteration of the while loop runs in `N^2` to find the argmin. This `O(N^2)` computation can be parallelized and finding L2-norm for each edge is independent from one another.

In the adding/removing steps, the order of adding and removing here is fixed in the sequential version as it always adds/removes one that minimizes the L2 norm. Potentially in our parallelized version, we can have a less greedy algorithm to take advantage of the parallelism.

The paper also explains how to use auxiliary edges to transform the solution to a fabricable string art image. But for our project, we initially will not implement this part and only output a solution of edges, and if time allows, we will implement adding auxiliary edges.

### The Challenge
The first challenge of this project is implementing the algorithm itself in C/C++. In addition to the psuedocode presented in the paper, there are [matlab](https://github.com/Exception1984/StringArt) and [Java](https://github.com/jblezoray/stringart) implementations. We need to first translate the implementation into C/C++.

To parallelize finding the minimum of L2-norm for all edges, each thread needs to have access to a minimum. For CUDA, there will be a lot of latency to access global minimum. With the greedy algorithm, a barrier is always needed to find `argmin`. so another challenge would be dividing the work to take advantage of parallelism to reduce these sequential work.

### Resources
We have 2 implementations of the algorithm for reference ([matlab](https://github.com/Exception1984/StringArt), [java](https://github.com/jblezoray/stringart)) and the [paper by Birsak et al](https://www.cg.tuwien.ac.at/research/publications/2018/Birsak2018-SA/Birsak2018-SA-preprint.pdf), but we will be implementing it in C/C++ from scratch. We still need to investigate to find the libraries that can be useful. We also have access to documentations of CUDA. As for machines, we will use the GHC machines for CUDA implementation.

### Goals and Deliverable
PLAN TO ACHIEVE:
We plan to implement the edges solution(instead of a continuous thread) sequentially, in parallel using CUDA and OPENMP.
We will benchmark their performances on GHC and lateday machines and analyze them.
For the demo, we want to be able to take input images, and output the string art version of it using our parallel algorithms, and show precomputed speedup graphs.

HOPE TO ACHIEVE:
We hope to implement the fabricable solution(using a continoous thread), but even if we do, we highly doubt that we would have the time to make this additional part parallelized, so it would not really contribute to the speedup.

BARE MINIMUM:
If our progress is slower as planned, we will only implement a parallel version using OPENMP, but all the analysis and demo will be the same: without Cuda.

### Platform Choice
We chose CUDA because the same set of instructions are applied to many different edges, and the problem itself lives in the world of computer graphics, so we are curious to see how CUDA could improve performance. 

### Schedule
* November 2 
  * Discuss Idea with Instructors
* November 4 
  * Project Proposal Due Wednesday
* November 5-9 (5 days)
  * Research and find useful libraries to use in C and C++ 
* November 10-18 (9 days)
  * Implement the sequential solution
* November 19-23 (5 days) 
  * Implement a baseline CUDA version
* November 25-December 3 (10 days) 
  * Add optimizations to the CUDA version
* November 27 
  * Checkpoint Report 
* December 3-8 (6 days) 
  * Benchmark CUDA 
* December 8-13 (6 days)
  * Write report
* TBD(after December 13th, day before final exam slot): 
  * Final Report Due
* TBD(after December 14th, during final exam slot): 
  * Virtual Poster Session

### References
Michael Birsak et al. “String art: towards computational fabrication of string images”. In:ComputerGraphics Forum. Vol. 37. 2. Wiley Online Library. 2018, pp. 263–274 [link](https://www.cg.tuwien.ac.at/research/publications/2018/Birsak2018-SA/Birsak2018-SA-preprint.pdf)

Exception1984. Exception1984/StringArt. [link](https://github.com/Exception1984/StringArt).

Jblezoray. jblezoray/stringart. [link](https://github.com/jblezoray/stringart)

petros vrellis petros. A new way to knit (2016). [link](http://artof01.com/vrellis/works/knit.html)
