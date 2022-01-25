CC = gcc mpicc
CXX = mpicxx
CXXFLAGS = -O3 -pthread
CFLAGS = -O3 -lm -pthread -march=native -fopenmp -msse4 -msse2 -msse3 -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize
TARGETS = hw3-1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
