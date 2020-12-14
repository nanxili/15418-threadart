## Parallelized String Art

### Author 
Catherine Yu (tianhony) Nanxi Li (nanxil1)

### Summaryg
String art is an image solely composed of strings between pins around a circular canvas. We implemented a parallelized string art solver in C++ and CUDA that computes the string art best resembling the input image. We developed our algorithm from scratch based on the sequential greedy approach proposed in paper by Brisak et al. We modified the proposed algorithm while implementing our sequential version of the solver, so that algorithm would have more parallelism to exploit while outputting more accurate string art image. We then developed our parallel version of the solver, which produces the same output as the sequential solver in a considerably shorter runtime. We were able to achieve an over 221x speedup on a 512*512 image with 128 pins. 

### Running the Code
To run the sequential version of the code:
```
  g++ -o3 sequentialThreatArt.cpp
  ./a.out -f [INPUT_FILE_NAME] -w [OUTPUT_WIDTH] -p [NUMBER_OF_PINS] -c [CONTRAST]
```

To run the parallel version of the code:
```
  make
  ./cudaThreadArt  -f [INPUT_FILE_NAME] -w [OUTPUT_WIDTH] -p [NUMBER_OF_PINS] -c [CONTRAST]
```

Both implementations have the following options:

```
Program Options:
  -f  --file_name <FILE_TO_READ>    must be of format jpg
  -w  --width <OUTPUT_WIDTH>        must be power of 2
  -p  --numPins  <NUMBER_OF_PINS>   must be power of 2
  -c  --contrast  <CONTRAST>        must be between -200 and 200
  -h  --help                        This message
```

For more information regarding the implementation, visit the [project page](https://nanxili.github.io/15418-threadart/).

