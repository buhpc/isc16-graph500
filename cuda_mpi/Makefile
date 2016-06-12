# the compiler: nvcc for CUDA program
CUDA = nvcc

# the build target executable:
TARGET = bfs_cuda

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(CUDA) $(CFLAGS) -o $(TARGET) $(TARGET).cu

clean:
	$(RM) $(TARGET)