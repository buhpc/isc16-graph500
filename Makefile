CC=mpicxx-openmpi-mp
FLAGS=-std=c++11
INCLUDE= -Iinclude -I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib
LIB=-lcudart
EXE=main
NVCC=nvcc
Q=@

CPP=$(wildcard src/*.cpp)
CU=$(wildcard src/*.cu)
OBJ=$(addprefix obj/,$(notdir $(CPP:.cpp=.o)))
CUOBJ=$(addprefix cuobj/,$(notdir $(CU:.cu=.o)))
.PHONY: all clean

all: $(EXE)

cuobj/%.o: src/%.cu
	$(Q)$(NVCC) $(INCLUDE) -c $^ -o $@
	$(Q)echo NVCC $<

obj/%.o: src/%.cpp
	$(Q)$(CC) $(FLAGS) $(INCLUDE) -c $^ -o $@
	$(Q)echo CXX $<

$(EXE): $(OBJ) $(CUOBJ)
	$(Q)$(CC) $(FLAGS) -o $@ $^ $(LDFLAGS) $(LIB)
	$(Q)echo CXX $@

clean: 
	$(Q)rm -f obj/* cuobj/* $(EXE)
