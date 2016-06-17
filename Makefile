CC=mpicxx-openmpi-mp
FLAGS=-std=c++11
INCLUDE= -Iinclude -I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib
LIB=-lcudart
EXE=main
NVCC=nvcc
Q=@

SOURCES=$(shell find src -name "*.cpp") $(shell find src -name "*.cu")
OBJECTS=$(SOURCES:.cpp=.o)
OBJECTS:=$(OBJECTS:.cu=.o)

.PHONY: all clean

all: $(EXE)

%.o: %.cu
	$(Q)echo NVCC $<
	$(Q)$(NVCC) $(INCLUDE) -c $^ -o $@

.cpp.o:
	$(Q)echo CXX $<
	$(Q)$(CC) $(FLAGS) $(INCLUDE) -c $^ -o $@

$(EXE): $(OBJECTS)
	$(Q)echo CXX $@
	$(Q)$(CC) $(FLAGS) -o $@ $^ $(LDFLAGS) $(LIB)

clean: 
	$(Q)rm -f $(OBJECTS) $(EXE)
