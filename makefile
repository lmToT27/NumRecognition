CXX = g++
INCLUDE = -Isrc/include -Iheader
LIB = -Lsrc/lib

SRC = $(wildcard src/*.cpp)

# target mặc định
all:
	$(CXX) $(INCLUDE) $(LIB) -o main main.cpp $(SRC)

# ===== build & run theo tên gõ =====
%:
	$(CXX) $(INCLUDE) $(LIB) -o $@ $@.cpp $(SRC)
	.\$@

clean:
	del /f *.exe main
