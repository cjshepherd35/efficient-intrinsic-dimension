#include <iostream>
#include <assert.h>

#define MAT_AT(m, i, j) m.data[i*(m).cols+j]

struct Mat
{
    int rows;
    int cols;
    float* data;
};


Mat mat_alloc(int rows,int cols);
void mat_fill(Mat m, float val);
void mat_rand(Mat m, float low, float high);
void mat_print(Mat m);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias);
void mat_transpose(Mat transpose, Mat original);


Mat mat_alloc(int rows, int cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float*)malloc(rows*cols*sizeof(float));
    return m;
}

void mat_fill(Mat m, float val)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = val;
        }
    }
    
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = ((float)rand() / (float)RAND_MAX) * (high - low) + low;
        }
        
    }
    
}


void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            std::cout << MAT_AT(m, i, j) << " "; //1 3 5
        }
        std::cout << std::endl;
        
    }
    
}


void  mat_dot(Mat dst, Mat a, Mat b)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    assert(a.cols == b.rows);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = 0.f;
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
        
    }
    
}


void mat_dot_bias(Mat dst, Mat a, Mat b, Mat bias)
{
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);
    assert(a.cols == b.rows);
    assert(bias.cols == dst.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst, i, j) = MAT_AT(bias, 0, j);
            for (size_t k = 0; k < a.cols; k++)
            {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
        
    }
}


void mat_transpose(Mat transpose, Mat original)
{
    for (size_t i = 0; i < original.rows; i++)
    {
        for (size_t j = 0; j < original.cols; j++)
        {
            MAT_AT(transpose, j, i) = MAT_AT(original, i, j);
        }
        
    }
    
}