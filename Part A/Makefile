NVCC     = nvcc
OPTIONS  = -O3

all: vecadd00 vecadd01 matmul00 matmul01

# ----- common objs -----
timer.o: timer.cu
	$(NVCC) -c $< -o $@ $(OPTIONS)

# ----- vecadd (Q1/Q2) -----
vecaddKernel00.o: vecaddKernel00.cu vecaddKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

vecaddKernel01.o: vecaddKernel01.cu vecaddKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

# compile vecadd.cu twice: w/o & with -DCOALESCED
vecadd00.o: vecadd.cu vecaddKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

vecadd01.o: vecadd.cu vecaddKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS) -DCOALESCED

vecadd00: vecadd00.o vecaddKernel00.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

vecadd01: vecadd01.o vecaddKernel01.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

# ----- matmul (Q3/Q4) -----
matmultKernel00.o: matmultKernel00.cu matmultKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

matmultKernel01.o: matmultKernel01.cu matmultKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

# compile matmult.cu twice: w/o & with -DOPTIMIZED
matmul00.o: matmult.cu matmultKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

matmul01.o: matmult.cu matmultKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS) -DOPTIMIZED

matmul00: matmul00.o matmultKernel00.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

matmul01: matmul01.o matmultKernel01.o timer.o
	$(NVCC) $^ -o $@ $(OPTIONS)

clean:
	rm -f *.o vecadd00 vecadd01 matmul00 matmul01
