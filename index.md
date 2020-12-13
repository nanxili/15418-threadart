## Parallelized String Art

### Author 
Catherine Yu (tianhony) Nanxi Li (nanxil1)

### 1 Summary
String art is an image solely composed of strings between pins around a circular canvas. We implemented a parallelized string art solver in C++ and CUDA that computes the string art best resembling the input image. We developed our algorithm from scratch based on the sequential greedy approach proposed in paper by Brisak et al\cite{stringArtPaper}. We modified the proposed algorithm while implementing our sequential version of the solver, so that algorithm would have more parallelism to exploit while outputting more accurate string art image. We then developed our parallel version of the solver, which produces the same output as the sequential solver in a considerably shorter runtime. We were able to achieve an over 221x speedup on a 512*512 image with 128 pins.

### 2 Background
First of all, what is string art? String art is a technique for the creation of visual artwork where images emerge from a set of strings that are spanned between pins. It is traditionally done with strings wounding around a grid of nails hammered into a velvet-covered wooden board. Here is a [video](https://vimeo.com/175653201) explaining how it is done.

![thread art](images/stringartsteps.png)
![thread art](images/vrellis.jpg)

We are going to use the algorithm presented by [Birsak et al](https://www.cg.tuwien.ac.at/research/publications/2018/Birsak2018-SA/Birsak2018-SA-preprint.pdf) in his research paper. This algorithm will not produce a solution using a single continuous thread, but many discrete edges. 

Let P be the number of pins that spans the edge of the circular canvas evenly, and As we researched for solutions to this problem, there were 2 main types of solutions, both of which assume opaque threads:

* Greedy Approach: 

  Starting from a pin $p_0$, the algorithm finds the pin p_1 (p_1 != p_0) that the string between the 2 pins best fits the given image. 
  Then starting from $p_1$, the algorithm finds the pin p_2 (p_2 != p_0) that best fits the given image, until there is no new connecting between 2 pins can make the fit better, or a maximum number of strings reached. 
  Note, the algorithm finds strings that are continuous (i.e, a string can only start from a pin where the previous string ends).

* Modified Greedy Approach: 

  The modified greedy approach compensates for the fact that the greedy approach has the disadvantage of making decision of adding a string at an early stage that will later turn out to be a bad choice.
  In our case, an addition of one edge might bring the biggest benefit when it is chosen, but can then prevent a better solution later on. 
  Thus, Brisak et al. proposed to further improve the results by an iterative removal of edges. 
  In particular, when the initial addition stage terminates, the modified greedy approach algorithm iteratively choose a string to test if removing that string will result in a better fit. If the removal of the string will result in a better fit, the string will be removed. The algorithm continues to search for strings whose removal can result in a better fit until no better fit from removal can be found. The algorithm then starts a second round of string adding. 
  The algorithm alternates between the addition stage and the removal stage, until it is not possible to further improve the norm and the algorithm terminates. 
  In addition, this algorithm further improves the quality of output by allowing strings to start from an arbitrary pin. 

  Another algorithm is then applied to make the found pin pairs into a fabricable string art image. Since fabricability is not the goal of this project, we will not include this part of the algorithm. 

For the Greedy approach, when looking for line to add, it checks at max $P$ pin pairs(fixed starting pin). 
For the modified greedy approach, when adding, it checks for at max $P^2$ pin pairs, and when removing, it checks all the added pin pairs, which is bounded by $P^2$.
We chose to use the Modified Greedy Approach because the output resembles the input better. However, since the problem is more complex, it is more challenging to implement a solver based on this algorithm. 

We removed the assumption for completely opaque strings in our implementation. Our implementation allows overlay of strings. In other words, the cross point of 2 strings would appear to be darker than the strings themselves. This results in better resemblance at low resolution, which is crucial since high resolution images would take too long to be processed by the sequential solver. More importantly, this change allows us to exploit the string-level parallelism in the removal stage (refer to section 3 for more details), which contributes to the overall speedup massively. 
### 3 Our Work Flow
* Overall Work Flow

  Our work flow is illustrated in the figure and our solver is shown as the pink box.
  The input to our framework is a colored image(for benchmarking purpose, we used grayscale images for easier contrast tuning).
  We convert it into a integer number representation in the range of [0, 255] and denote it further as the column vector: 
  \begin{align}
      y \in [0, 255]^m \subset  \mathbb{Z}^m
  \end{align}
  where $m$ is the number of all pixels, concatenated row-wise.
  The output of the optimization algorithm is a binary array, where each
  entry corresponds to an edge which could be drawn on the canvas.
  Activated bins reflect edges which need to be drawn.
  We denote the output as the column vector
  \begin{align}
      x \in \mathbb{B}^n
  \end{align}
  where $n$ is the number of all possible string edges.
  It has the dimensional of $P(P-1)$.\\
  The goal of the optimization problem is to determine the best way to to define a mapping F from the space of edges to the space of
  pixels, i.e.,
  \begin{align}
      F: \mathbb{B}^n \rightarrow [0, 255]^m \text{ with }  x \mapsto F(x) 
  \end{align}
  and to determine the values of the elements of the vector x such that
  it delivers the best approximation of the input image.
  We cast the problem into a binary non-linear least squares optimization task:
  \begin{align}
      \min _{x} \lVert F(x)-y  \lVert^2 \text{ s.t. } x \in \mathbb{B}
  \end{align}
  And the optimization is solved using the algorithm in Figure \ref{fig:algo}.\\

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
We have researched different implementations for thread art online, and we were aiming to find an implementation that's suitable for coding in CUDA and parallelizable. The two references we found when proposing the project both had theoretically parallelizable algorithm, but it was hard to directly translate them to the baseline code we wanted. We looked into other implementations online as well, and we reached the conclusion that we need to write our baseline code from scratch.

We implemented the baseline code in two parts: (1) image pre-processing according to the algorithm proposed in paper by Birsak et al; (2) greedy computation to find the most accurate string representation of the image by adding and subtracting threads. We used the stb_image library in image pre-processing, since it is robust and easy to use. We implemented the greedy computation by storing all pixels and lines in int[], and only used arithmetic operations for all computation in terms of drawing/removing lines, analyzing images and generating L2-norms. The approach was difficult to implement in terms of correctness, because there are many non-trivial steps that are intertwined with each other. We chose this approach regardless because this approach is easy to optimize as CUDA code, which was what we proposed.

We now finished outlining CUDA code for part (2) described above. We want to find the pair of pins that maximizes reduction of the l2 norm, there are n(n-1) problems always, we use cuda to have each thread compute the l2 norm for each problem, and then reduce them to find the one with the largest l2 norm. We will also parallelize pixel processing of the line generation. 

### Goals past Checkpoint
Although we have spent more time on the baseline code than what we proposed, now we have a steady grasp of the algorithm and our implementation that we believe the optimization will take less time then what we expected. We expect to deliver the same as stated in PLAN TO ACHIEVE. We would not be able to achieve the NICE TO HAVES since we are tight on scheduling. We want to implement the edges solution(instead of a continuous thread) and optimize with CUDA.

### Poster Session Plan
We will not be able to do a live demo since thread art projects usually take hours to synthesize. (14 hours as mentioned in the paper.) We want to show the images that we synthesized beforehand, and we also want to show our speedup analysis graphs during poster session. 

### Concerns
The main concern is that GHC machines kill the jobs that timed out. It would be difficult to benchmark our baseline solution on a high-resolution picture because it would take too long. We will try to benchmark the solution on low-resolution pictures, which will not have the resemblance we want, while running high-resolution pictures at the same time to ensure correctness.  

### References
Michael Birsak et al. “String art: towards computational fabrication of string images”. In:ComputerGraphics Forum. Vol. 37. 2. Wiley Online Library. 2018, pp. 263–274 [link](https://www.cg.tuwien.ac.at/research/publications/2018/Birsak2018-SA/Birsak2018-SA-preprint.pdf)

Exception1984. Exception1984/StringArt. [link](https://github.com/Exception1984/StringArt).

Jblezoray. jblezoray/stringart. [link](https://github.com/jblezoray/stringart)

petros vrellis petros. A new way to knit (2016). [link](http://artof01.com/vrellis/works/knit.html)
