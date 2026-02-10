all:
	g++ -Isrc/include -Iheader -Lsrc/lib -o main main.cpp src/*.cpp
run:
	.\main