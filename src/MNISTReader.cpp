#include "MNISTReader.h"

int ReverseInt(int i) {
    int b0 = i & 0xff;
    int b1 = i >> 8 & 0xff;
    int b2 = i >> 16 & 0xff;
    int b3 = i >> 24 & 0xff;
    return ((b0 << 24) + (b1 << 16) + (b2 << 8) + b3);
}

std::vector <Matrix> ReadImages(std::string path) {
    std::ifstream file(path, std::ios::binary);
    assert(file.is_open() && ("Failed to open file " + path).c_str());
    std::vector <Matrix> images;
    int MagicNumber;
    int NumberOfImages;
    int Rows;
    int Cols;
    file.read((char*)&MagicNumber, sizeof(MagicNumber));
    MagicNumber = ReverseInt(MagicNumber);
    file.read((char*)&NumberOfImages, sizeof(NumberOfImages));
    NumberOfImages = ReverseInt(NumberOfImages);
    file.read((char*)&Rows, sizeof(Rows));
    Rows = ReverseInt(Rows);
    file.read((char*)&Cols, sizeof(Cols));
    Cols = ReverseInt(Cols);
    for (int i = 0; i < NumberOfImages; i++) {
        Matrix img(Rows, Cols);
        for (int r = 0; r < Rows; r++) {
            for (int c = 0; c < Cols; c++) {
                unsigned char tmp = 0;
                file.read((char*)&tmp, sizeof(tmp));
                img(r, c) = static_cast <double> (tmp) / 255.0;
            }
        }
        images.push_back(img);
    }
    return images;
}

std::vector <int> ReadLabels(std::string path) {
    std::ifstream file(path, std::ios::binary);
    assert(file.is_open() && ("Failed to open file " + path).c_str());
    std::vector <int> labels;
    int MagicNumber;
    int NumberOfLabels;
    file.read((char*)&MagicNumber, sizeof(MagicNumber));
    MagicNumber = ReverseInt(MagicNumber);
    file.read((char*)&NumberOfLabels, sizeof(NumberOfLabels));
    NumberOfLabels = ReverseInt(NumberOfLabels);
    for (int i = 0; i < NumberOfLabels; i++) {
        unsigned char tmp = 0;
        file.read((char*)&tmp, sizeof(tmp));
        labels.push_back(static_cast <int> (tmp));
    }
    return labels;
}