NVCC = nvcc
OPTIONS = -O3

all: vecadd00 vecadd01 matmul00 matmul01

vecaddKernel00.o: vecaddKernel00.cu
	$(NVCC) -c $< -o $@ $(OPTIONS)

vecaddKernel01.o: vecaddKernel01.cu
	$(NVCC) -c $< -o $@ $(OPTIONS)

matmultKernel00.o: matmultKernel00.cu
	$(NVCC) -c $< -o $@ $(OPTIONS)

matmultKernel01.o: matmultKernel01.cu
	$(NVCC) -c $< -o $@ $(OPTIONS)

timer.o: timer.cu
	$(NVCC) -c timer.cu -o timer.o $(OPTIONS)

vecadd00: vecadd.cu vecaddKernel00.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

vecadd01: vecadd.cu vecaddKernel01.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

matmul00: matmult.cu matmultKernel00.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

matmul01: matmult.cu matmultKernel01.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

clean:
	rm -f *.o vecadd00 vecadd01 matmul00 matmul01

# vecadd00: Non-coalesced vector addition
# vecadd01: Coalesced vector addition
# matmul00: Naive matrix multiply
# matmul01: Optimized shared-memory & unrolled version