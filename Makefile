CXX=g++
CXXFLAGS=-Wall -pedantic -O3 -std=c++17 -march=native -fopenmp


.PHONY: all
all: BK1 BK3 BK5 bkstream scripts/logspace


BK1: BK1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

BK3: BK3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

BK5: BK5.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

bkstream: bkstream.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Plain-portable helper for scripts/run_benchmarks.sh: no OpenMP, no -march,
# so it builds anywhere a C++ compiler exists.
scripts/logspace: scripts/logspace.cpp
	$(CXX) -O2 -std=c++17 -o $@ $<

.PHONY: clean
clean:
	$(RM) BK1 BK3 BK5 bkstream scripts/logspace *.o
