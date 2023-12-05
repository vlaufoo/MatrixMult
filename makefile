enable := 1
disable := 0
#default block size is 16, but it can be specified to a different number
block ?= 0
type ?= 0

ifneq ($(type), $(disable))
	DEFINES := -D TYPE=$(type)
endif

ifneq ($(block), $(disable))
	DEFINES += -D BLOCK_SIZE=$(block)
endif

ifeq ($(numbers), $(enable))
	DEFINES += -D PRINT_NUMBERS
endif

ifeq ($(debug), $(enable))
	DEFINES += -D VERBOSE
endif

ifeq ($(check), $(enable))
	DEFINES += -D CHECK_RESULT
endif

all: main_CUDA main_old

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
