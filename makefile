
#enable=0
#
#testing: testing.cpp Classes.h makefile
#	g++ testing.cpp Classes.h -D VERBOSE -o testing -O0 -Wall 
#main_old: main_old.o
#	g++ main_old.o -o main_old
#main_debug: main_old.cpp Classes.h makefile
#ifeq($(verbose), $(enable))
#	g++ main_old.cpp Classes.h -o main_debug -D PRINT_NUMBERS -Wall -std=c++11
#else
#	g++ main_old.cpp Classes.h -o main_debug -D VERBOSE -D PRINT_NUMBERS -Wall -std=c++11
#endif
#main_old.o: main_old.cpp Classes.h makefile
#	g++ main_old.cpp Classes.h -c  -Wall -std=c++11
#
#.PHONY:clean
#clean:
#	rm *.o main_old main_new main_debug
#.PHONY:tar
#tar:
#	tar cfvz MatrixMult.tar.gz *.cpp *.h* makefile
#

disable=0

main_CUDA: main_old.cpp Functions.hpp UnopTile.cpp OpTile.cpp BestSquareTiling.cpp makefile
	g++ Functions.hpp main_old.cpp OpTile.cpp UnopTile.cpp SingleTileThread.cpp BestSquareTiling.cpp -o main_CUDA -D PRINT_NUMBERS -D VERBOSE 

main_new: main_new.cpp Tensor.hpp makefile
	g++ main_new.cpp Tensor.hpp -o main_new -Wall

testing: testing.cu Functions.hpp CudaFunctions.cu makefile
	nvcc testing.cu -I./Common -D PRINT_NUMBERS -D CUDA -D VERBOSE -o testing 

main_old: main_old.o
	g++ main_old.o -o main_old

main_debug: main_old.cpp Classes.h makefile
ifeq ($(verbose), $(disable))
	g++ main_old.cpp Classes.h -o main_debug -D PRINT_NUMBERS -Wall  
else
	g++ main_old.cpp Classes.h -o main_debug -D VERBOSE -D PRINT_NUMBERS -Wall 
endif

main_old.o: main_old.cpp Classes.h makefile
	g++ main_old.cpp Classes.h -c -Wall 

.PHONY: clean
clean:
	rm *.o main_old main_new main_debug

.PHONY: tar
tar:
	tar cfvz MatrixMult.tar.gz *.cpp *.h* makefile
