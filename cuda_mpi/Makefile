# the compiler: nvcc for CUDA program
MPI = mpicc

# the build target executable:
TARGET = bfs_cuda

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(MPI) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

clean:
	$(RM) $(TARGET)