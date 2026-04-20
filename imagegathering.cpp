#include <algorithm>
#include "multilayernn.cpp"
#include  <random>
#include <chrono>
#include <map>
#include <fstream>


std::string labelsfilename = "C:/Users/cshep/Downloads/t10k-labels.idx1-ubyte";
std::string imagesfilename = "C:/Users/cshep/Downloads/t10k-images.idx3-ubyte";


//reading mnist images
std::vector<std::vector<float>> read_images(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::binary);

    char magicNumber[4];
    char numImages[4];
    char numRows[4];
    char numCols[4];
    file.read(magicNumber, 4);
    file.read(numImages, 4);
    file.read(numRows, 4);
    file.read(numCols, 4);

    int numims = (static_cast<unsigned char> (numImages[0]) << 24) | (static_cast<unsigned char> (numImages[1]) << 16) | (static_cast<unsigned char> (numImages[2]) << 8) | static_cast<unsigned char> (numImages[3]);
    int numrs =  (static_cast<unsigned char> (numRows[0]) << 24) | (static_cast<unsigned char> (numRows[1]) << 16) | (static_cast<unsigned char> (numRows[2]) << 8) | static_cast<unsigned char> (numRows[3]);
    int numcs = (static_cast<unsigned char> (numCols[0]) << 24) | (static_cast<unsigned char> (numCols[1]) << 16) | (static_cast<unsigned char> (numCols[2]) << 8) | static_cast<unsigned char> (numCols[3]);

    std::vector<std::vector<unsigned char>> cimages;
    for (size_t i = 0; i < numims; i++)
    {
        std::vector<unsigned char> image(numrs*numcs);
        file.read((char*)(image.data()), numrs*numcs);
        cimages.push_back(image);
    }
    file.close();
    
    std::vector<std::vector<float>> images;
    
    for (size_t i = 0; i < cimages.size(); i++)
    {
        std::vector<float> tempim;
        for (size_t j = 0; j < cimages[0].size(); j++)
        {
            tempim.push_back((float)cimages[i][j]/255.f);
        }
        images.push_back(tempim);
    }
    return images;
}

//read in labels for the images.
std::vector<std::vector<int>> read_labels(const std::string& filename, int num_classes)
{
    std::vector<std::vector<unsigned char>> clabels;
    std::ifstream file(filename, std::ios::binary);

    char magicNumber[4];
    char numLabels[4];
    file.read(magicNumber, 4);
    file.read(numLabels,4);
    int numlabs = (static_cast<unsigned char> (numLabels[0]) << 24) | (static_cast<unsigned char> (numLabels[1]) << 16) | (static_cast<unsigned char> (numLabels[2]) << 8) | static_cast<unsigned char> (numLabels[3]);
    for (size_t i = 0; i < numlabs; i++)
    {
        std::vector<unsigned char> label(1);
        file.read((char*)(label.data()),sizeof(char));
        clabels.push_back(label);
    }
    file.close();

    std::vector<std::vector<int>> labels;
    for (size_t i = 0; i < numlabs; i++)
    {
        std::vector<int> templabel;
        for (size_t j = 0; j < num_classes; j++)
        {
            if (j == (int)clabels[i][0])
            {
                templabel.push_back(1);
            }  
            else templabel.push_back(0);
        }
        labels.push_back(templabel);
    }

    return labels;
}

std::vector<int> getrandvec(int size, int num_samples)
{
    std::vector<int> randvec;
    
    for (size_t i = 0; i < num_samples; i++)
    {
        randvec.push_back(rand() % 10000);
    }
    return randvec;
}

Mat randomizeImages(std::vector<std::vector<float>> images, std::vector<int> randvec, int num_samples, int n_inputs)
{
    Mat trainimages = mat_alloc(num_samples, n_inputs);
    int i = 0;
    
    for (int randnum : randvec)
    {
        for (size_t j = 0; j < n_inputs; j++)
        {
            MAT_AT(trainimages, i,j) = images[randnum][j];  
        }   
        i++;
    }
    return trainimages;
}

Mat randomizeLabels(std::vector<std::vector<int>> labels, std::vector<int> randvec, int num_samples, int num_classes)
{
    Mat trainlabels = mat_alloc(num_samples, num_classes);
    int i = 0;
    for (int randnum : randvec)
    {
        for (size_t k = 0; k < num_classes; k++)
        {
            MAT_AT(trainlabels, i,k) = labels[randnum][k]; 
        }
        i++;
    }
    return trainlabels;
}