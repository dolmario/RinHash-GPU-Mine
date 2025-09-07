
### Build
```bash
make clean && make -j"$(nproc)"


Run (Beispiel)
export WALLET="your_Wallet"
export POOL_PASS="c=Coin,ID=cuda,sd=0.001"
unset CUDA_LAUNCH_BLOCKING
export RIN_BATCH=512
export RIN_CHUNK=512
./rinhash_cuda
