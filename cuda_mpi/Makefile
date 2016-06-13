# the compiler: nvcc for CUDA program
MPI = mpicc

# the build target executable:
TARGET = bfs_mpi_cuda

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(MPI) $(CFLAGS) -o $(TARGET) $(TARGET).cu

clean:
	$(RM) $(TARGET)