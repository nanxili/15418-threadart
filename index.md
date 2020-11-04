## Parallelized String Art

### Author 
Catherine Yu (tianhony) Nanxi Li (nanxil1)

### Summary
We aim to parallelize the the problem of converting an input image to string art instructions on NVIDIA GPUs(using CUDA) and Multicore Machines(Using OPENMP). 

### Background
First of all, what is string art? String art is a technique for the creation of visual artwork where images emerge from a set of strings that are spanned between pins. It is traditionally done with strings wounding around a grid of nails hammered into a velvet-covered wooden board. Here is a [video](https://vimeo.com/175653201) explaining how it is done.

**some pictures here**
![thread art](https://github.com/nanxili/15418-threadart/blob/gh-pages/images/vrellis.jpg)


We are going to use the algorithm presented by Birsak et al.

**one picture of stringartalgo.png here**

What the greedy algorithm does is that it keeps finding an edge that minimizes the L2-norm until the L2-norm cannot decrease further, and then it keeps removes found edges until the L2-norm cannot decrease further. It repeats the adding edges and removing edges until L2-norm can no longer be minimized.

This algorithm will not produce a solution using a single continuous thread, but many discrete edges. The paper also explains how to use auxiliary edges to transform the solution to a fabricable string art image. But for our project, we initially will not implement this part and only output a solution of edges, and if time allows, we will implement adding auxiliary edges.

With `N` needles, there are `N*(N-1)` edges total(each needle can be connected with all `N-1` needles, so each iteration of the while loop runs in `N^2` to find the argmin. This `O(N^2)` computation can be parallelized and finding L2-norm for each edge is independent from one another.

In the adding/removing steps, the order of adding and removing here is fixed in the sequential version as it always adds/removes one that minimizes the L2 norm. Potentially in our parallelized version, we can have a less greedy algorithm to take advantage of the parallelism.

### The Challenge
The first challenge of this project is implementing the algorithm itself in C/C++. In addition to the psuedocode presented in the paper, there is are matlab\cite{matlabgithub} and Java\cite{javagithub} implementations. We need to first translate the implementation into C/C++.

To parallelize finding the minimum of L2-norm for all edges, each thread needs to have access to some sort of the minimum. For CUDA, there will be a lot of latency to access global minimum, but this is not a limit for OPENMP.

With the greedy algorithm, a barrier is always needed to find \texttt{argmin}. so another challenge would be thinking about how to divide the work to take advantage of parallelism to reduce these sequential work.

### Resources
We have 2 implementations of the algorithm to look at\cite{matlabgithub, javagithub}, and the paper\cite{stringArtPaper} itself, but we will be implementing it in C/C++ from scratch. We still need to investigate to find the libraries that can be useful. We also have access to documentations of CUDA and OPENMP. As for machines, we will use the GHC machines for CUDA implementation, and lateday machines for OPENMP implementation.

### Goals and Deliverable
\textbf{PLAN TO ACHIEVE}:\\
We plan to implement the edges solution(instead of a continuous thread) sequentially, in parallel using CUDA and OPENMP.\\
We will benchmark their performances on GHC and lateday machines and analyze them.\\
For the demo, we want to be able to take input images, and output the string art version of it using our parallel algorithms, and show precomputed speedup graphs.\\
\textbf{HOPE TO ACHIEVE}:\\
We hope to implement the fabricable solution(using a continoous thread), but even if we do, we highly doubt that we would have the time to make this additional part parallelized, so it would not really contribute to the speedup.\\
\textbf{BARE MINIMUM}:\\
If our progress is slower as planned, we will only implement a parallel version using OPENMP, but all the analysis and demo will be the same: without Cuda.\\
\break

### Platform Choice
We chose CUDA because the same set of instructions are applied to many different edges, and the problem itself lives in the world of computer graphics, so we are curious to see how CUDA could improve performance.\\
We chose OPENMP because when we looked into this problem, the first thing that came into our mind is that there is a lot of shared memory access, and each thread does not really need to keep track of many of its own memory.\\
\break

### Schedule
Monday,  November 2nd: Discuss Idea with Instructors Monday\\
Wednesday, November 4th: Project Proposal Due Wednesday\\
Wednesday, November 9th: Research and find useful libraries to use in C and C++\\
Monday, November 16th: Implement the sequential solution\\
Friday, November 20th: Implement a baseline OPENMP version\\
Monday, November 26th: Add optimizations to the OPENMP version\\
Monday, November 27th: Benchmark OPENMP\\
Friday, November 27rd: Checkpoint Report\\
Tuesday, December 1st: Implement a baseline CUDA version\\
Monday, December 7th: Add optimizations to the CUDA version\\
Tuesday, December 8th: Benchmark CUDA\\
TBD(after December 13th, day before final exam slot): Final Report Due \\
TBD(after December 14th, during final exam slot): Virtual Poster Session \\

### References
\printbibliography
\break
\newpage
\end{document}

