enable := 1

DEFINES := -D BLOCK_SIZE=16

ifeq ($(numbers), $(enable))
	DEFINES += -D PRINT_NUMBERS
endif

ifeq ($(debug), $(enable))
	DEFINES += -D VERBOSE
endif

ifeq ($(check), $(enable))
	DEFINES += -D CHECK_RESULT
endif


main_CUDA: main_old.cpp CudaFunctions.cu Functions.hpp makefile 
	@./cudaprep.sh
	nvcc main_old.cpp.cu -I./Common -o main_CUDA $(DEFINES) -D CUDA
	@rm *.cpp.cu
	@echo removed .cpp.cu files

testing: testing.cpp Functions.hpp CudaFunctions.cu makefile
	@./cudaprep.sh
	nvcc testing.cpp.cu -I./Common $(DEFINES) -D CUDA -o testing
	@rm *.cpp.cu
	@echo removed .cpp.cu files

main_old: main_old.cpp Functions.hpp makefile
	g++-12 main_old.cpp -o main_old $(DEFINES) -Wall

.PHONY: clean
clean:
	rm *.o main_old main_new main_debug testing

.PHONY: tar
tar:
	tar cfvz MatrixMult.tar.gz *.cpp *.h* makefile
