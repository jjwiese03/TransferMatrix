GPU: ./src/GPU_acc/module.cu
	nvcc -std=c++17 -shared -Xcompiler -fPIC -o ./src/GPU_acc/module.so ./src/GPU_acc/module.cu -lcufft

julia_test:
	./test/runtest.jl