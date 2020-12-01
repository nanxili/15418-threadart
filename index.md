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

>Our algorithm computes a subset of all possible strings which are used together to reassemble the input image. In each iter ation we pick the edge that allows the biggest norm reduction, and we stop when further addition would cause an increase of the error.The big disadvantage of a greedy approach as we use it is that a decision at an early stage can later turn out to be a bad choice. In our case, an addition of one edge might bring the biggest benefit when it is chosen, but can then prevent a better solution later on. Thus, we try to further improve the results by an iterative removal of edges. In particular, if the initial addition stage terminates, we sequentially remove those edges that allow to improve the norm. If it is not possible anymore, we start to add edges again, pick one edge at a time, and continue as long as the norm can be improved. We alternate those two stages until it is not possible to further im- prove the norm, neither by removal nor by addition, in which case the terminates. (Birsak et al, 2018)

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
We plan to implement the edges solution(instead of a continuous thread) sequentially, in parallel using CUDA.
We will benchmark their performances on GHC and lateday machines and analyze them.
For the demo, we want to be able to take input images, and output the string art version of it using our parallel algorithms, and show precomputed speedup graphs.

HOPE TO ACHIEVE:
We hope to implement the fabricable solution(using a continoous thread), but even if we do, we highly doubt that we would have the time to make this additional part parallelized, so it would not really contribute to the speedup.

BARE MINIMUM:
If our progress is slower as planned, we will only implement a parallel version using OPENMP, but all the analysis and demo will be the same: without Cuda.

### Platform Choice
We chose CUDA because the same set of instructions are applied to many different edges, and the problem itself lives in the world of computer graphics, so we are curious to see how CUDA could improve performance. 

### Schedule (updated)
November 2 
* Discuss Idea with Instructors

November 4 
* Project Proposal Due Wednesday

November 5-11 (7 days)
* Research and find useful libraries to use in C and C++ 

November 12-29 (18 days)
  * Implement the sequential solution

November 30 
  * Checkpoint Report 

December 1-3  (3 days) 
  * Implement a baseline CUDA version

December 4-10 (7 days) 
  * Add optimizations to the CUDA version
  * Parallel processing the dots on the lines when drawing a single line
  * Parallel processing the lines and compute L2 norms when adding/deleting lines

December 11-14 (4 days)
  * Benchmark CUDA and Write report

December 14: 
  * Final Report Due

December 15 8.30-11.30am: 
  * Virtual Poster Session

### Work Completed until Checkpoint 
We have researched different implementations for thread art online, and we were aiming to find an implementation that's suitable for coding in CUDA and parallelizable. The two references we found when proposing the project both had theoritically parallelizable algorithm, but it was hard to directly translate them to the baseline code we wanted. We looked into other implementations online as well, and we reached the conclusion that we need to write our baseline code from scratch.

We implemented the baseline code in two parts: (1) image pre-processing according to the algorithm proposed in paper by Birsak et al; (2) greedy computation to find the most accurate string representation of the image by adding and subtracting threads. We used the stb_image library in image pre-processing, since it is robust and easy to use. We implemented the greedy computation by storing all pixels and lines in int[], and only used arithmetic operations for all computation in terms of drawing/removing lines, analyzing images and generating L2-norms. The approach was difficult to implement in terms of correctness, because there are many non-trival steps that are intertwined with each other. We chose this approach regardless because this approach is easy to optimize as CUDA code, which was what we proposed.

We now finished outlining CUDA code for part (2) described above. We want to find the pair of pins that maximizes reduction of the l2 norm, there are n(n-1) problems always, we use cuda to have each thread compute the l2 norm for each problem, and then reduce them to find the one with the largest l2 norm

### Goals past Checkpoint
Although we have spent more time on the base line code than what we proposed, now we have a steady grasp of the algorithm and our implementation that we believe the optimization will take less time then what we expected. We expect to deliver the same as stated in PLAN TO ACHIEVE. We would not be able to achieve the NICE TO HAVES since we are tight on scheduling. We want to implement the edges solution(instead of a continuous thread) and optimize with CUDA.

### Poster Session Plan
We will not be able to do a live demo since thread art projects usually takes hours to synthesize. (14 hours as mentioned in the paper.) We want to show the images that we synthesized beforehand, and we also want to show our speedup analysis graphs during poster session. 

### Concerns
The main concern is that GHC machines kill the jobs that timed out. It would be difficult to benchmark our baseline solution on a high-resolution picture because it would take too long. We will try to benchmark the solution on low-resolution pictures, which will not have the resemblance we want, while running high-resolution pictures at the same time to ensure correctness.  

### References
Michael Birsak et al. “String art: towards computational fabrication of string images”. In:ComputerGraphics Forum. Vol. 37. 2. Wiley Online Library. 2018, pp. 263–274 [link](https://www.cg.tuwien.ac.at/research/publications/2018/Birsak2018-SA/Birsak2018-SA-preprint.pdf)

Exception1984. Exception1984/StringArt. [link](https://github.com/Exception1984/StringArt).

Jblezoray. jblezoray/stringart. [link](https://github.com/jblezoray/stringart)

petros vrellis petros. A new way to knit (2016). [link](http://artof01.com/vrellis/works/knit.html)
