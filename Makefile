CXX=g++
CXXFLAGS=-Wall -pedantic -O3 -std=c++17 -march=native -fopenmp


.PHONY: all
all: BK1 BK3 BK5 bkstream


BK1: BK1.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

BK3: BK3.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

BK5: BK5.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

bkstream: bkstream.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) BK1 BK3 BK5 bkstream *.o
