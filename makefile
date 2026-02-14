CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp
INCLUDES = -Isrc/include -Iheader
LDFLAGS = -Lsrc/lib -fopenmp
INCLUDE = $(INCLUDES)

SRC = $(wildcard src/*.cpp)
all:
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -o main main.cpp $(SRC)

%:
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -o $@ $@.cpp $(SRC)
	.\$@

clean:
	del /f *.exe main
