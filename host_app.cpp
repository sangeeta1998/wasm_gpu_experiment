// host_app.cpp
#include <iostream>
#include <wasm.h>
#include <wasmtime.h>

// Function declaration for the CUDA kernel
extern "C" void call_cuda_kernel(float* A, float* B, float* C, int N);

int main() {
    // Initialize Wasmtime
    wasm_engine_t* engine = wasm_engine_new();
    wasm_store_t* store = wasm_store_new(engine);

    // Load the WebAssembly module
    const char* wasm_path = "target/wasm32-wasip1/release/wasm_gpu_experiment.wasm";
    FILE* file = fopen(wasm_path, "rb");
    if (!file) {
        std::cerr << "Failed to open WebAssembly file: " << wasm_path << std::endl;
        return 1;
    }
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    wasm_byte_vec_t wasm_bytes;
    wasm_byte_vec_new_uninitialized(&wasm_bytes, file_size);
    fread(wasm_bytes.data, 1, file_size, file);
    fclose(file);

    // Compile the WebAssembly module
    wasm_module_t* module = wasm_module_new(store, &wasm_bytes);
    if (!module) {
        std::cerr << "Failed to compile WebAssembly module" << std::endl;
        return 1;
    }
    wasm_byte_vec_delete(&wasm_bytes);

    // Create an instance of the WebAssembly module
    wasm_extern_vec_t imports = WASM_EMPTY_VEC;
    wasm_instance_t* instance = wasm_instance_new(store, module, &imports, nullptr);
    if (!instance) {
        std::cerr << "Failed to instantiate WebAssembly module" << std::endl;
        return 1;
    }

    // Get the exported function from the WebAssembly module
    wasm_extern_vec_t exports;
    wasm_instance_exports(instance, &exports);
    if (exports.size == 0) {
        std::cerr << "No exports found in WebAssembly module" << std::endl;
        return 1;
    }
    wasm_func_t* matrix_mul = wasm_extern_as_func(exports.data[0]);
    if (!matrix_mul) {
        std::cerr << "Export is not a function" << std::endl;
        return 1;
    }

    // Define matrix size
    int N = 4;
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    // Prepare arguments for the WebAssembly function
    wasm_val_t args_val[4] = {
        WASM_I32_VAL((int32_t)(uintptr_t)A),
        WASM_I32_VAL((int32_t)(uintptr_t)B),
        WASM_I32_VAL((int32_t)(uintptr_t)C),
        WASM_I32_VAL(N)
    };
    wasm_val_vec_t args = WASM_ARRAY_VEC(args_val);

    // Call the WebAssembly function
    wasm_val_t results_val[1];
    wasm_val_vec_t results = WASM_ARRAY_VEC(results_val);
    wasm_trap_t* trap = wasm_func_call(matrix_mul, &args, &results);
    if (trap) {
        std::cerr << "Failed to call WebAssembly function" << std::endl;
        wasm_trap_delete(trap);
        return 1;
    }

    // Call the CUDA kernel
    call_cuda_kernel(A, B, C, N);

    // Print the result
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free host memory
    free(A);
    free(B);
    free(C);

    // Clean up
    wasm_extern_vec_delete(&exports);
    wasm_instance_delete(instance);
    wasm_module_delete(module);
    wasm_store_delete(store);
    wasm_engine_delete(engine);

    return 0;
}