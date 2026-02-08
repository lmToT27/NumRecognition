#pragma once
#include <vector>
#include "Matrix.h"

int ReverseInt(int i);
std::vector <Matrix> ReadImages(std::string fullPath);
std::vector <int> ReadLabels(std::string fullPath);