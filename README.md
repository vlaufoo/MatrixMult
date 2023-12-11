<div align="justify">

# Index

- [Tiled Multiplication](https://github.com/vlaufoo/MatrixMult#tiled-multiplication)
- [Program Structure](https://github.com/vlaufoo/MatrixMult#program-structure)
- [Test Results](https://github.com/vlaufoo/MatrixMult#test-results)
- [Optimizations](https://github.com/vlaufoo/MatrixMult#optimizations)
- [Conslusions](https://github.com/vlaufoo/MatrixMult#Conclusions)
- [CUDA Variant](https://github.com/vlaufoo/MatrixMult/tree/master#cuda-variant)
- [Compiling](https://github.com/vlaufoo/MatrixMult#compilation)

# Tiled Multiplication

Traditional matrix multiplication is an iterative operation that repeats a **multiply & accumulate** step, for each element in the result matrix, a number of times that is equal to the columns of the first operand (rows of the second operand).
In the case of $C = A \cdot B$, the C code that describes the multiplication is:

```c++
for(i=0; i<A_rows; i++){ //row of the result matrix C (equal to row of the operand A)
  for(j=0; j<B_columns; j++){ //column of the reslt matrix C (equal to column of the operand B)
    for(k=0; k<A_columns; k++){ //column of operand A (equal to row of operand B)
  	   C[i][j] += A[i][k] * B[k][j];  //multiply and add
    }
  }
}
```

## Parallelization

When looking for ways to parallelize this process, to speed up execution, an immediate answer is to have each separate processing unit (or thread) calculate one element of the result matrix. Since they are completely independent of each other no data races should be caused. This approach of course will require a number of threads or processing units that is equal to the number of elements of the result matrix.
A more flexible approach is to specify the target number of threads, based on the available resources, and devide the process in exactly that number of **independent** operations. A good solution for this would be **tiled multiplication**.
The normal multiplication algorithm can be expanded to be used not with single elements of the matrices, but with **square tiles** of a set size.

```c++
for(i=0; i<A_rows/tile_size; i++){ //row of tiles in the result matrix C
  for(j=0; j<B_columns/tile_size; j++){ //column of tiles in the reslt matrix C
    for(k=0; k<A_columns/tile_size; k++){ //column of tiles in operand A
  	   C[i][j] += A[i][k] * B[k][j];  //multiply and add
    }
  }
}
```

In this second case the `C[i][j]` identifies a square tile in the `C` matrix, and the `*` operator should be considered a normal matrix multiplication. All the main properties of the operation transfer to the tiled version, so each result tile is only dependent on the row of tiles and the column of tiles it lies in, in the `A` matrix and `B` matrix respectively, exactly like the single elements in a normal matrix multiplication. Now we can apply the same parallelization approach to this tiled version of the algorithm and, by changing the `tile_size`, we can set the number of threads that are going to be needed to calculate the result.

![Tiled_Mult.png](https://github.com/vlaufoo/MatrixMult/blob/master/Tiled_Mult.png?raw=true)

## Generalization of the algorithm

When looking at this approach, it seems that the result matrix must be **exactly divisible** by the intended number of square tiles, which would greatly limit our possibilities. This problem is easily solvable with the addition of **padding** rows and/or columns. By adding rows and columns of zeros to pad the matrices to the most suitable size for our tiling, we can generalize this type of multiplication, at the cost of an overhead in computation, since these rows and columns will have to be added, and the results of their multiplication will still be calculated (returning zero). To obtain a perfectly divisible `C` result matrix then, we have to add **padding rows** to the `A` matrix and **padding columns** to the `B` matrix, but making the columns of `A` (and thus the rows of `B`) divisible by `tile_size` is not strictly necessary, since the multiplication of these last tiles yields a square result tile, compatible with the result matrix new dimensions.
In theory, if the result matrix was a perfect square, padding it to use two threads would be useless, since the second thread would only operate on a tile made entirely of padding, but to keep the results of the tests consistent, uniform and informative, these cases were included.






# Program Structure

The program was written in c++ and includes a Matrix class, written to facilitate operations between matrices and to include the tiling functions, and of course the main function that initiates all the threads.

The Matrix class includes:

- A constructor, to dynamically allocate the matrix.
- A copy constructor.
- A destructor, to correctly deallocate the memory.
- A few operators (sum, normal multiplication, a custom || operator to aid with padding, a copy operator to avoid double delete errors) which override their normal meaning.
- A few get/set methods, since the members are private.
- A few methods specifically created to prep the matrices for multiplication (padding addition, padding removal etc.)
- The tiled multiplication function, that will be described in the following paragraph.

## The tiled multiplication method

The `MultiplyTilesOnce` method in the Matrix class is the "kernel" of the tiled matrix multiplication. In ths scaled down version of the tiled multiplication algorithm, each thread will repeat this kernel for as many times as there are columns of tiles in the `A` matrix, so that it is completely independent of the other threads. In other, more complete forms, the threads could first do all multiplications between tiles, and then all the additions to get the final result, splitting the operation in its *"data-independent"* and *"data-dependent"* components, and doing the second part only when all the partial results are ready. In our case, the function itself is nothing more than an adaptation of the normal matrix multiplication, using the index of the result tile and the number of the kernel iteration as points of reference, applied only to one tile:

```c++
        void MultiplyTilesOnce(Matrix& A, Matrix& B, int IdxAcol, int IdxArow, int IdxBcol, int tSize){
      if(A.columns != B.rows){
        std::cout<<"The matrices are incompatible!\n";
        exit(1);
      }
      //all indices are referred to A matrix and transposed for the B matrix
      int tileCstart = IdxBcol*tSize;
      int tileCend = (IdxBcol+1)*tSize;
      int tileRstart = IdxArow*tSize;
      int tileRend = (IdxArow+1)*tSize;
      int ThisTileEnd = (IdxAcol+1)*tSize;
      int ThisTileStart = IdxAcol*tSize;

      if(ThisTileEnd > A.columns){
        ThisTileEnd = A.columns;
      }
      
      if(IdxAcol == 0){
        //if it's the first iteration set destination matrix to 0)
        if(IdxArow == 0 && IdxBcol == 0){
          padded_rows = A.padded_rows;
          padded_columns = B.padded_columns;
        }
      }

      //normal matrix multiplication for one tile
      for(int i=tileRstart; i<tileRend; i++){
        for(int j=tileCstart; j<tileCend; j++){
          for(int k=ThisTileStart; k<ThisTileEnd; k++){
            matrixpt[i][j] += A.matrixpt[i][k]*B.matrixpt[k][j];
          }
        }
      }
    }
```

The intialization of threads is done in the **main** function, where a vector of type `std::thread` is created and then extended, using `emplace_back`, as new threads are initialized.

```c++
    std::vector<std::thread> threads;

    for(i=0; i<A.Rows()/tSize; i++){
      for(j=0; j<B.Columns()/tSize; j++){
        threads.emplace_back(SingleTileThread, ThN, std::ref(Y), std::ref(A), std::ref(B), iterations, i, j, tSize);
        ThN++;
      }
    }

    for(auto& thread :threads){
      thread.join();
    }
```

`emplace_back` takes as arguments the name of the function that the thread will execute, and then its arguments, which are passed as references, using `std::ref()`. This is done to optimize the multiplication, by avoiding the creation of local variables, and is possible only because the locations of memory each thread writes to never overlap with the other threads. The `std::ref()` wrapper is used instead of the typical `T&` reference in this case, because the `std::thread` constructor typically copies all arguments passed to the start function of the thread (in this case `SingleTileThread`) and passes those copied values to the function. A way to circumvent that and obtain a behavior that resembles the one of a reference, when initiating a thread, is to use `std::ref()`.
After all threads have concluded their part of the operation, they are rejoined to the main thread with the for loop at the end, so the parallel operation is only as fast as the slowest thread.

## Challenges

The main challenges of this project, after the first steps of creating the class and the main methods, have been regarding the correct initialization of threads, and have been caused in part by the late implementation of a readable debug output, to troubleshoot the problems as they came up. It is possible that some of the difficulties encountered in making the parallel execution work were due to the use of the `std::cout` function, and in particular the improper use of the `<<` operator.
When using `std::cout` the `<<` operator can be used to concatenate strings to send them to the output buffer. The operator itself is atomic, so if used only once it cannot be scrambled or interrupted by opther threads, but using it many times in the same `std::cout` has the same effect of calling the function multiple times. Between each call other threads can insert their own output, making the result unreadable. The solution was to compose the message before sending it to buffer, using another standard library and then send it all at once. This does not guarantee that all messages are in "chronological order", but only that they are all readable.

```c++
#include<iostream>
#include<sstream>

int main(){
    std::stringstream msg;
    msg << "\nTile row "<<IdxArow<<" column "<<IdxBcol<<". Iteration "<<IdxAcol<<"\n\n";
    std::cout<< msg.str(); //will always output the complete message without interruptions

    return 0;
}
```

Another problem encountered was the double delete problem, before the copy constructor and operator were added to fix it. This was due to the standard copy constructor only doing a **shallow copy** of the other object when called, and then causing problems when it was time to deallocate the memory for it, because the same pointer to the matrix elements was used for both the original and the copy.
While the parallelization problem had not yet been solved, an attempt to solve is was made with the creation of a new class, the [Tensor](https://github.com/vlaufoo/MatrixMult#compilation) class, to be able to store the partial results in different layers of the same tensor object, and finally add the layers together, thus dividing the operation in the two *dependent* and *independent* sections already described.






# Test Results

The main program includes also the `ctime` library, to measure execution time of the serial multiplication and of the multi-threaded one, and outputs the results to a log file. After importing the log file as a table into Matlab, the data was formatted to be graphed.
In the first test the program (which cycles through matrix dimensions in a range set by the arguments), was run repeatedly, using the `tests.sh` script, changing parameters each time. The following table describes all the combinations tested.

|     |     |
| --- | --- |
| Rows | 150-1050 |
| Threads | 2, 4, 6, 8, 10 |
| Operand Form Factor | 1, 1.3, 1.7, 2 |
| Result Form Factor | 1, 1.3, 1.7, 2 |

## Plots
The following graph shows the execution times of all the multiplications, for square matrices (both result and operands), with all possible degrees of parallelization.

![Square_Parallel_Comparison.png](https://github.com/vlaufoo/MatrixMult/blob/master/Square_Parallel_Comparison.png?raw=true)
It is clear from the picture that the only configuration with an advantage over the normal one is the one with **four threads** operating in parallel. This is because the shape of the matrix lends itself to a perfectly symmetrical division in all directions. In the following image, this advantage will begin to fade as we change the shape of the result matrix (while keeping operands square).

![Square_Parallel_Comparison_plus_width.png](https://github.com/vlaufoo/MatrixMult/blob/master/Square_Parallel_Comparison_plus_width.png?raw=true)
The 4-threaded operation has lost its advantage and the 6-threaded one, more suitable for a 1:1.3 form factor, has gained the lead. Of course the trend continues as we make the matrix even wider, and the more parallel configurations start to become useful as they can spread their tiles wider in a 2x4 or 2x5 grid.

![Square_Parallel_Comparison_double_width.png](https://github.com/vlaufoo/MatrixMult/blob/master/Square_Parallel_Comparison_double_width.png?raw=true)


Once the matrix becomes a 1:2 rectangle, the 2 thread, 8 thread and 10 thread solutions become more suitable for the operation as they fit perfectly the shape of the result matrix, but the overhead associated with initializing the threads is too costly to allow the more parallelized versions to outperform the simpler two-threaded operation. Let's try to understand if this overhead is indeed caused by the initialization of the threads, or is linked to another factor. If it was given, at least in part, by the threads, increasing the load of each thread, should in theory counter this problem, by spreading the lost time among more operations (multiply & add).

With the same dataset, graphing the average Speedup across all matrix dimensions, against the operands' form factor, yields the following result:

![Double_width_speedup_change_with_FFO.png](https://github.com/vlaufoo/MatrixMult/blob/master/Double_width_speedup_change_with_FFO.png?raw=true)

Indeed a small increment in the speedup is visible, as the number of operations performed by the threads increases. To better test this theory, a new dataset was created, with more changes in the opeand form factor value. Once again, the graph shows an increase in the relative speed of the parallel operation when more multiplications are needed on each thread.

![Square_speedup_change_with_FFO.png](https://github.com/vlaufoo/MatrixMult/blob/master/Square_speedup_change_with_FFO.png?raw=true)

In these two images we can also notice that the improvement seems to be greatest in the configuation that is already the most efficient.

## Analytical model
In theory, this positive trend should continue indefinitely. The speedup will gradually increase as the operations increase, approaching asymptotically a value equal to the threads employed. 
The number of operations (multiply & add) done by one thread in this type of tiled multiplication is:
$$Op={FF_{op} \times FF_{res} \times R^3 \over T}+OH={MADD \over T}+OH$$
Where $FF_{res}$ and $FF_{op}$ are the result matrix and operands form factors respectively, $R$ is the number of rows of the result matrix, and $OH$ is the overhead. Under these conditions, the speedup can be calculated as:
$$Speedup={MADD \over {MADD \over T}+OH}={T \over 1+{T \times OH \over MADD}}$$
We can conclude that, to improve the speedup, any increase in the number of *Multiply & Add (**MADD**)* operations is a welcome one, and when $MADD\to\infty$, then $Speedup \to T$.
If we were to graph the actual curves of the execution time against the matrix size, and the estimations obtained using this simple model, we would get something like this:

![Time_vs_operand_FF_vs_rows.png](https://github.com/vlaufoo/MatrixMult/blob/master/Time_vs_operand_FF_vs_rows.png?raw=true)
In this picture, we have plotted the data extracted from the test program, and compared it with an estimation, for each operand form factor.
$$t_{EX}={R^3 \times FF_{op} \times FF_{res} \times t_{MADD} \over 4}$$
except that in this case the result matrix was always square, so:
$$t_{EX}={R^3 \times FF_{op} \times t_{MADD} \over 4}$$
The value of $t_{MADD}$ was calculated from the dataset, as the average of ${t_{EX} \over R^3}$ with $FF_{op} = 1$ and $FF_{res} = 1$, and was estimated at ***3.3 ns***. The comparison with the ideal case clearly underlines the presence of **overhead**, increasing proportionally with $R^3$.

![Time_vs_operand_FF_vs_rows_Serial.png](https://github.com/vlaufoo/MatrixMult/blob/master/Time_vs_operand_FF_vs_rows_Serial.png?raw=true)
The serial operation is instead well modeled, as seen in the above picture.

# Optimizations
In the provious tests we have employed a very general version of tiled multiplication. We can adapt the concept to make it more efficient in the case of a small number of processing units or threads, like ours, skipping the calculation of the **partial** result tiles (in our case obtained by the `MultiplyTilesOnce` method), and instead create a method that returns the **final** result of a specific tile. This should decrease the time we spend in between functions, and dedicate it to the actual multiplication. Another improvement that comes to mind is to **remove the need for padding**, by stopping the multiplication before it goes out of bounds, simply knowing the number of rows and columns of the intended result. This way we remove two of the most wasteful sections of the original design, and should obtain a visible improvement in **speedup**.

The now method is called `GetResultTile`, and will be called only once by each thread. We call it in the **main** with a slightly different syntax:
```c++
    std::vector<std::thread> tiles;
    clock_t tic_2 = clock();
    //parallel execution
    for(i=0; i<Rdiv; i++){
      for(j=0; j<Cdiv; j++){
        tiles.emplace_back(&Matrix::GetResultTile, ref(T), ref(A), ref(B), i, j, tR, tC);
      }
    }

    for(auto& thread :tiles){
      thread.join();
    }

    clock_t toc_2 = clock();
```
And the function uses the values o `tR` and `tC`, together with the indices, to set the boundaries of the result tile it's calculating, within the result matrix.

## More plots
Now that we have implemented this new operation, we will test its performance, with a few more plots.
In the following figure, we show once again the execution times of two different operations, but now they are the ***"unoptimized"*** parallel version and the ***"optimized"*** version.


![Square_optimized_comparison.png](https://github.com/vlaufoo/MatrixMult/blob/master/Square_optimized_comparison.png?raw=true)
As is clearly visible by the curves in almost all degrees of parallelization, removing the unnecessary overhead has been baneficial.

![Double_speedup_change_with_FFO_optimized.png](https://github.com/vlaufoo/MatrixMult/blob/master/Double_speedup_change_with_FFO_optimized.png?raw=true)
From this second figure we can desume that the biggest weight that is lifted by the new optimizations is the wasted operations on the padding elements. The very inefficient **4-threaded** configuration, which was doing 3 times as many useless operations as the useful ones, has jumped up in speed since the removal of padding. The more efficient configurations though, like the **2-threaded** one, have not gained anythuing from the change: the amounts of thread initializations necessary in both approaches is two, and since the optimization, in this case, is limited to that, and no padding needs to be removed, the ***"optimized"*** version is even marginally slower than the original one.
Looking at the other configuations, we can see that the same reasoning applies the **8-threaded** solution, which again is very suitable for this form factor, whereas the **10-threaded** one, relatively inefficient in the original operation, has definately been improved. 
Now we can explore once again the question of why the theoetical estimations were sensibly faster than the real operations, in the parallel case. Let us produce the same graph, but this time only using the execution times of the ***"optimized"*** version of the multiplication. The model used is again the ideal one, but the dashed lines are now drawn with data from the ***"optimized"*** test:

![Time_vs_operand_FF_vs_rows_optimized.png](https://github.com/vlaufoo/MatrixMult/blob/master/Time_vs_operand_FF_vs_rows_optimized.png?raw=true)
The operation still takes more than anticipated. This added time is reflected also in the results we have already commented on, where we have seen that the new method does not in fact surpass the old one, when it is tested in the best possible conditions (like this case).

# Conclusions
In summation, the tiled multiplication experiment has proven reasonably successful. The algorithm has in many cases improved the speed of the multiplication, but has shown in many others its incompatibility with small arrays of processing units. If the number of **parallel** therads was considerably greater than the one used for this experiment, like for example in the use of **GPUs**, the tiles could be much smaller, and the load of the unnecessary operations on padding elements would be shared across many more units. In our case this approach has proven at times extremely inefficient and has been partially improved by the modifications described in the paragraphs above. The overhead caused by padding elements has been completely removed and approaches that were preaviously unusable have become feasable. In its most efficient form, and in the most favorable conditions, the ***original*** tiled multiplication algorithm still won over the ***optimized*** one.

# CUDA variant
After these tests, another section was added to the program that calls a CUDA kernel, which will be executed on the discrete GPU. This new functionality is only included in the main_CUDA program and requires the CUDA toolkit to be installed on the system and of course a compatible graphics card.

## Basic concepts of CUDA programming
Cuda is an interface offered by nvidia to allow General Purpose application development on nvidia GPUs. Our application will use an extremely limited set of the features offered by CUDA, but will nonetheless require an understanding of the formalizations that are used by this programming model to define how the workload is distributed among the many processors in the GPU.
To define and divide the workload to different degrees of parallelization CUDA defines a hyerarchy of structures. The hyerarchy goes as follows (with increasing parallelism): 
- Streaming Multiprocessors
- Thread Block Grids
- Thread Blocks
- Threads

Most of the steps in this hierarchy do not necessarily map to a corresponding hardware structure. In fact above the threads level, the boundaries between different elements at the same level are given by the intrinsic parallelization of the task, and are often also based on memory limitations.

The kernel, which is the function performed concurrently by all threads in the GPU, is similar to the one performed in the optimized CPU operation and reads as follows:
```c++
template <typename T = int>
__global__ void CudaMultKernel(struct mat<T> A, struct mat<T> B, struct mat<T> C)
{
  T Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.x;
  int column = blockIdx.x * blockDim.x + threadIdx.y;


  for(int i=0; i<A.width; i++){
    Cvalue += A.elements[row * A.width + i] * B.elements[i * B.width + column];
  }

  C.elements[row * C.width + column] = Cvalue;
}
```
In this kernel, each thread, identified by its indices, `threadIdx.x` and `threadIdx.y` calculates one element of the matrix `C`. The following graphs show the speedup relative to the optimized CPU opration, and the difference in execution times relative to the unoptimized CPU operation.

![Cuda_vs_CPUtile_ratio.png](https://github.com/vlaufoo/MatrixMult/blob/master/Cuda_vs_CPUtile_ratio.png?raw=true)
![CPU_time_vs_cuda_time.png](https://github.com/vlaufoo/MatrixMult/blob/master/CPU_time_vs_cuda_time.png?raw=true)

From the first graph we can conclude that the CUDA version takes an amount of time grat grows like a poynomial one order lower than the CPU version. We have already concluded that the CPU operation is proportional to $R^3$, so the CUDA version must be proportional to $R^2$.

# Compilation
The log program used for this experiment is compilable through the `main_old` make target, and can then be run, giving the intended 7 arguments:
```
make [check] [block] [debug] [numbers]
```
Where:
- check is used to add a result validity check for every iteration (used for testing) by setting it to 1 and is 0 by default.
- block is the value that will be used as the side dimension of the square thread block in the cuda kernel call.
- type is used to define the type of the matrices' elements. The templates all have a default value of `int`, but with this flag that can be changed.
- debug and numbers are more flags used for testing, and control the amount of information printed to stdout. Both are 0 by default.
and then
```
./main_old <threads> <random_seed> <step_n> <step_size> <operand_FF> <result_FF> <starting_step>
```
or
```
./main_CUDA <threads> <random_seed> <step_n> <step_size> <operand_FF> <result_FF> <starting_step>
```
Where:
- `threads` is the number of threads to be used, and **forces** the program to use that many to make the multiplication, even if the corresponding tiling is extremely wateful (e.g. in the case of a square matrix, when `threads` is 2).

- `random_seed` is the seed used to randomly generate the matrix, and is given as an argument to make it easier to replicate the same testing conditions.

- `step_n` `step_size` and `starting_step` are the number of steps by which the matrix size will be increased, and then the size of the steps, and the step from which the program will start.

- `operand_FF` and `result_FF` are the two form factors mentioned in the exposition of the previous graphs.

</div>
