#pragma once

#include <fstream>
#include <cassert>
#include <vector>
#include <string>
#include "Matrix.h"
#include "Math.h"

int ReverseInt(int i);
std::vector <Matrix> ReadImages(std::string fullPath);
std::vector <int> ReadLabels(std::string fullPath);