GPU: ./src/GPU_acc/module.cpp
	g++-15 -std=c++23 -shared -fPIC -I ./include -o ./src/GPU_acc/module.so ./src/GPU_acc/module.cpp

julia_test:
	./test/runtest.jl