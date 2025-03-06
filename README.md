# wasm_gpu_experiment

nvcc -c matrix_mul.cu -o matrix_mul_cuda.o

g++ host_app.cpp matrix_mul_cuda.o -o host_app -lwasmtime -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -I/usr/local/include -L/usr/local/lib

''' Build the C api manually for wasmtime'''

cd wasmtime/crates/c-api

cargo build --release

Copy the Shared Library:

sudo cp target/release/libwasmtime.* /usr/local/lib/

Copy the Headers:

sudo cp -r crates/c-api/include/* /usr/local/include/

Update the Dynamic Linker Cache:

sudo ldconfig
