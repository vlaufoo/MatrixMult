# Index

- Tiled Multiplication
- Program Structure
- Test Results
- Compiling

# Tiled Multiplication

Traditional matrix multiplication is an iterative operation that repeats a **multiply & accumulate** step, for each element in the result matrix, a number of times that is equal to the columns of the first operand (or rows of the second operand).
In the case of `C = A*B`, the C code that describes the multiplication is:

```c++
for(i=0; i<A_rows; i++){ //row of the result matrix C (equal to row of the operand A)
  for(j=0; j<B_columns; j++){ //column of the reslt matrix C (equal to column of the operand B)
    for(k=0; k<A_columns; k++){ //column of operand A (equal to row of operand B)
  	   C[i][j] += A[i][k] * B[k][j];  //multiply and add
    }
  }
}
```

## Parallelizaiton

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

![Tiled_Mult.png]()

## Generalization of the algorithm

When looking at this approach, it seems that the result matrix must be **exactly divisible** by the intended number of square tiles, which would greatly limit our possibilities. This problem is easily solvable with the addition of **padding** rows and/or columns. By adding rows and columns of zeros to pad the matrices to the most suitable size for our tiling, we can generalize this type of multiplication, at the cost of an overhead in computation, since these rows and columns will have to be added, and the results of their multiplication will still be calculated (returning zero). To obtain a perfectly divisible `C` result matrix then, we have to add **padding rows** to the `A` matrix and **padding columns** to the `B` matrix, but making the columns of `A` (and thus the rows of `B`) divisible by `tile_size` is not strictly necessary, since the multiplication of these last tiles yields a square result tile, compatible with the result matrix new dimensions.
In theory, if the result matrix was a perfect square, padding it to use two threads would be useless, since the second thread would only operate on a tile made entirely of padding, but to keep the results of the tests consistent, uniform and informative, these cases were included.

# Program Structure

The program was written in c++ and includes a Matrix class, written to facilitate operations between matrices and to include the tiling functions, and of course the main function that initiates all the threads.

The Matrix class includes:

- A constructor, to dynamically allocate the matrix.
- A copy constructor.
- A destructor, to correctly deallocate the memory.
- A few operators (sum, normal multiplication, an custom || operator to aid with padding, a copy operator to avoid double delete errors) which override their normal meaning.
- A few get/set methods, since the members are private.
- A few methods specifically created to prep the matrices for multiplication (padding addition, padding removal etc.)
- The tiled multiplication function, that will be described in the following paragraph.

## The tiled multiplication method

The `MultiplyTilesOnce` method in the Matrix class is the "kernel" of the tiled matrix multiplication. In ths scaled down version of the tiled multiplication algorithm, each thread will repeat this kernel for as many times as there are columns of tiles in the `A` matrix, so that it is completely independent of the other threads, but in other, more complete forms, the threads could first do all matrix multiplications between tiles, and then all the additions to get the final result, splitting the operation in its *"independent"* and *"dependent"* components, and doing the second part only when all the partial results are ready. In our case, the function itself is nothing more than an adaptation of the normal matrix multiplication, using the index of the result tile and the number of the kernel iteration as points of reference, applied only to one tile:

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

The main challenges of this project, after the first steps of creating the class and the main methods, have been regarding the correct initialization of threads, and have been caused possibly by the late implementation of a readable debug output, to troubleshoot the problems as they came up. It is possible that some of the difficulties encountered in making the parallel execution work were due to the use of the `tsd::cout` function, and in particular the improper use of the `<<` operator.
When using `std::cout` the `<<` operator can be used to concatenate strings to send them to the output buffer. The operator itself is atomic, so if used only once it cannot be scrambled or interrupted by opther threads, but using it may times in the same `std::cout` hase the same oeffect of calling the function multiple times. Between each call other threads can insert their own output, making the result unreadable. The solution was to compose the message before sending it to buffer, and the send it all at once. This does not guarantee that all messages are in "chronological order", but that they are all readable.

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

Another problem encountered was the double delete problem, before the copy constructor and operator were added to fix it. This was due to the standard copy constructor only doing a shallow copy of the other object when called, and then causing problems when it was time to deallocate the memory for it, because the same pointer was used for the original and the copy.

# Test Results

The main program includes also the `ctime` library, to measure execution time of the serial multiplication and of the multi-threaded one, and outputs the results to a log file. After importing the log file as a table into Matlab, the data was formatted to be graphed.
In the first test the program (which cycles through matrix dimensions in a range set by the arguments), was run repeatedly, using the `tests.sh` script, changing parameters each time. The following table describes all the combinations tested.

|     |     |
| --- | --- |
| Rows | 150-1050 |
| Threads | 2, 4, 6, 8, 10 |
| Operand Form Factor | 1, 1.3, 1.7, 2 |
| Result Form Factor | 1, 1.3, 1.7, 2 |

## Graphs
The following graph shows the execution times of all the multiplications, for square matrices (both result and operands), with all possible degrees of parallelization.

![Square_Parallel_Comparison.png](:/7fd798350c2a40e6aaf728857020bf83)

![Square_Parallel_Comparison_plus_width.png](:/03d7d7cc67914077bf7e7b467479f577)

![Square_Parallel_Comparison_double_width.png](:/d75fcd4fe02040b0b11b8016848f9530)

It is clear from the picture that the only configuration that has an advantage over the normal one is the one with four threads operating in parallel. This is because the shape of the matrix lends itself to a perfectly symmetrical division in all directions. In the following image, this advantage will begin to fade as we change the shape of the result matrix (while keeping operands square). The 4 threaded operation has lost its advantage and the 6 threaded one, more suitable for a 1:1.3 form factor, has gained the lead. Of course the trend continues as we make the matrix even wider, and the more parallel configurations start to become useful as they can spread their tiles wider in a 2x4 or 2x5 grid.
Once the matrix becomes a 1:2 rectangle, the 2 thread, 8 thread and 10 thread solutions become more suitable for the operation as thay fit perfectly the shape of the result matrix, but the overhead associated with initializing the threads is too costly to allow the more parallelized versions to outperform the simpler two-threaded operation. Let's try to understand if this overhead is indeed caused by the initialization of the threads, or is linked to another factor. If it was only given by the threads, increasing the load of each thread, should in theory counter this problem, by spreading the lost time among more operations (multiply & add).
With the same dataset, graphing the average Speedup across all matrix dimensions, against the operands' form factor, yields the following result:

![Double_width_speedup_change_with_FFO.png](:/26aaaff970294f16b5c66013bbd45cac)

To better test this theory, a new dataset was created, with more changes in the opeand form factor value. Once again, the graph shows an increase in the speed of the parallel operation as more multiplications are needed on each thread.

![Square_speedup_change_with_FFO.png](:/f5e28f66d3b840079348dafb97b2f827)

From these two images we can see not only an improvement in all configurations as the form factor increases (and thus more multiplications are done by each thread), but also that it seems to be greatest in the configuation that is already the most efficient.
In theory this trend should continue indefinitely. The number of operations (multiply & add) done by one thead in this type of tiled multiplication is:
&& OP = {R^3FFrFFo} &&