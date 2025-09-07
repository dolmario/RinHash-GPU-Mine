NVCC      = nvcc
ARCH      = -arch=sm_86
STD       = --std=c++17
NVCCFLAGS = -O3 $(ARCH) $(STD) -use_fast_math -rdc=true -I.
LIBS      = -lcrypto -lssl -lm

rinhash_cuda: main_cuda.o rinhash_optimized.o sha3-256.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

main_cuda.o: main_cuda.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

rinhash_optimized.o: rinhash_optimized.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

sha3-256.o: sha3-256.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f rinhash_cuda *.o
.PHONY: clean
