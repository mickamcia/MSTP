all: build

build: main.cu
	/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -o run main.cu -L/usr/lib/nvidia-compute-utils-530 -lGLEW -lGL -lglfw -lcudart -lcufft
	