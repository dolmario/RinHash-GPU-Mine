NVCC      = nvcc
ARCH      = -arch=sm_86
STD       = --std=c++17
HOSTSTD   = -std=c++17
NVCCFLAGS = -O3 $(ARCH) $(STD) -use_fast_math -rdc=true -Xcompiler "$(HOSTSTD)" -I.
# (optional, hilft bei Argon2d-Last): NVCCFLAGS += -Xptxas -dlcm=cg

LIBS = -lcrypto -lssl -lm

rinhash_cuda: main_cuda.o rinhash_optimized.o sha3-256.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

main_cuda.o: main_cuda.cpp rinhash_params.h sha3_256_device.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

rinhash_optimized.o: rinhash_optimized.cu argon2d_device.cuh blake3_device.cuh sha3_256_device.cuh rinhash_params.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

sha3-256.o: sha3-256.cu sha3_256_device.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f rinhash_cuda *.o

.PHONY: clean

