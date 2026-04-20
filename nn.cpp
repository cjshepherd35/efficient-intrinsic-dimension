#ifndef NN_H
#define NN_H
#include <algorithm>
#include  <random>
#include <vector>
#include "matrixops.cpp"


class LayerDense
{
public:
    LayerDense(int input_dim, int n_out, bool bias, int num_samples);
    void forward(Mat inputs);
    void backward(Mat dvalues); 

    Mat output;
    Mat dinputs;
    Mat biases;
    Mat dbiases;
    
    Mat weights;
    Mat weights_T;
    Mat dweights;
    bool layerbias;
    int n_samples;
    Mat layerinputs;
    Mat layerinputs_T;

};
#endif

LayerDense::LayerDense(int input_dim, int n_out, bool bias, int num_samples)
{
    n_samples = num_samples;
    layerbias = bias;
    weights = mat_alloc(input_dim, n_out);
    mat_rand(weights, -0.1f, 0.1f);
    weights_T = mat_alloc(n_out, input_dim);
    if (layerbias)
    {
        biases = mat_alloc(1,n_out);
        mat_rand(biases, -0.1f, 0.1f);
        dbiases = mat_alloc(1, n_out);
    }
    dweights = mat_alloc(input_dim, n_out);
    dinputs = mat_alloc(num_samples, input_dim);
    layerinputs = mat_alloc(num_samples, input_dim);
    layerinputs_T = mat_alloc(input_dim, num_samples);
    output = mat_alloc(n_samples, n_out);
}

/*
a b

c d    
e f

o u
*/

void LayerDense::forward(Mat inputs)
{
    layerinputs = inputs;
    if (layerbias)
    {
        mat_dot_bias(output, inputs, weights, biases);
    }
    else
    {
        mat_dot(output, inputs, weights);
    }
    
}


void LayerDense::backward(Mat dvalues)
{
    mat_fill(dweights, 0.f);
    mat_transpose(layerinputs_T, layerinputs);
    mat_dot(dweights, layerinputs_T, dvalues);
    if (layerbias)
    {
        for (size_t i = 0; i < dvalues.cols; i++)
        {
            for (size_t j = 0; j < dvalues.rows; j++)
            {
                MAT_AT(dbiases, 0, i) += MAT_AT(dvalues, j, i);
            }
        }
    }

    mat_transpose(weights_T, weights);
    mat_dot(dinputs, dvalues, weights_T);
}




class Lora
{
public:
    Lora(int n_inputs, int n_out, bool bias, int num_samples, int r);
    void forward(Mat inputs);
    void backward(Mat dvalues);
    void bigBack(Mat dvalues);


    Mat output;
    Mat dinputs;
    Mat dbiases;

    Mat wA, wB;
    Mat biases;
    bool layerbias;
    int n_samples, rank;

    Mat dwA, dwB;
    Mat layerinputs;
};



Lora::Lora(int n_inputs, int n_out, bool bias, int num_samples, int r)
{
    n_samples = num_samples;
    layerbias = bias;
    rank = r;
    wA = mat_alloc(n_inputs, rank);
    wB = mat_alloc(rank, n_out);

    mat_rand(wA, -1, 1);
    mat_rand(wB, -0.5f, 0.5f);
    if (layerbias)
    {
        biases = mat_alloc(1, n_out);
        mat_rand(biases, -1, 1);
        dbiases = mat_alloc(1,n_out);
    }
    dwA = mat_alloc(n_inputs, rank);
    dwB = mat_alloc(rank, n_out);
    dinputs = mat_alloc(num_samples, n_inputs);
    layerinputs = mat_alloc(num_samples, n_inputs);
    output = mat_alloc(n_samples, n_out);

}

void Lora::forward(Mat inputs)
{
    assert(inputs.rows == output.rows);
    assert(wB.cols == output.cols);
    assert(wA.cols == wB.rows);
    assert(wA.rows ==  inputs.cols);
    assert(wB.cols == output.cols);
    
    layerinputs = inputs;
    for (size_t i = 0; i < output.rows; i++)
    {
        for (size_t j = 0; j < output.cols; j++)
        {
            if(layerbias) MAT_AT(output, i, j) = MAT_AT(biases, 0, j);
            else MAT_AT(output, i, j) = 0.0f;
            for (size_t l = 0; l < inputs.cols; l++)
            {
                float totalweight{0.0f};
                for (size_t k = 0; k < rank; k++)
                {
                    totalweight += MAT_AT(wA, l, k) * MAT_AT(wB, k, j);
                }
                MAT_AT(output, i, j) += MAT_AT(inputs, i, l) * totalweight;
                
            }
            

        }
        
    }
    
}



void Lora::bigBack(Mat dvalues)
{
    // size_t inputrows = layerinputs.rows;
    // size_t inputcols = layerinputs.cols;
    // size_t outputcols = dvalues.cols;
    // size_t R = rank;

    if(layerbias)
    {
        for (size_t i = 0; i < dvalues.rows; i++)
        {
            MAT_AT(dbiases, i, 0) = 0;
            for (size_t j = 0; j < dvalues.cols; j++)
            {
                MAT_AT(dbiases, 0, j) += MAT_AT(dvalues, i, j);
            }
        }
    }

    Mat inpwa = mat_alloc(layerinputs.rows, wA.cols);
    Mat inpwa_t = mat_alloc(wA.cols, layerinputs.rows);
    Mat layerinputs_t = mat_alloc(layerinputs.cols, layerinputs.rows);
    Mat dvalwb = mat_alloc(dvalues.rows, wB.rows);
    Mat wa_t = mat_alloc(wA.cols, wA.rows);
    Mat wb_t = mat_alloc(wB.cols, wB.rows);

    //calculate dvalwb 
    mat_transpose(wb_t, wB);
    mat_dot(dvalwb, dvalues, wb_t);

    //inputs and wa combined
    mat_dot(inpwa, layerinputs, wA);

    //calculate dwb
    mat_transpose(inpwa_t, inpwa);
    mat_dot(dwB, inpwa_t, dvalues);

    //calculate dwa
    mat_transpose(layerinputs_t, layerinputs);
    mat_dot(dwA, layerinputs_t, dvalwb);

    mat_dot(dinputs, dvalwb, wa_t);
}


void Lora::backward(Mat dvalues)
{
    // IMPORTANT: grad_wA and grad_wB must start at 0.0 because we accumulate +=
    mat_fill(dwA, 0);
    mat_fill(dwB, 0);

    //gemini code....... 
    size_t N = layerinputs.rows;       // Batch size
    size_t I = layerinputs.cols;      // Input features
    size_t J = dvalues.cols;         // Output features
    size_t R = rank;                // LoRA Rank

    // 1. Bias Gradients
    // Simple accumulation, no extra memory needed
    if (layerbias)
    {
        for (size_t j = 0; j < J; j++)
        {
            double sum = 0.0;
            for (size_t n = 0; n < N; n++) sum += MAT_AT(dvalues, n, j);
            MAT_AT(dbiases, 0, j) = sum;
        }
    }

    // 2. Weights and Input Gradients
    // We iterate through the Batch (N) and Rank (R) first.
    // Inside these loops, we calculate temporary scalars on the fly.
    
    for (size_t n = 0; n < N; n++)
    {
        for (size_t r = 0; r < R; r++)
        {
            // --- Scalar A: Project Gradient backward through B ---
            // Represents (grad_output[n] . wB[r]^T)
            // We compute this single double value by looping over J
            double grad_proj_scalar = 0.0;
            for (size_t j = 0; j < J; j++)
            {
                grad_proj_scalar += MAT_AT(dvalues, n, j) * MAT_AT(wB, r, j);
            }
    
            // --- Scalar B: Project Input forward through A ---
            // Represents (inputs[n] . wA[r])
            // We compute this single double value by looping over I
            double input_proj_scalar = 0.0;
            for (size_t i = 0; i < I; i++)
            {
                input_proj_scalar += MAT_AT(layerinputs, n, i) * MAT_AT(wA, i, r);
            }

            // --- Apply to Gradients ---
            // Update grad_wB (using Scalar B)
            // dL/dwB += (Input * wA)^T * grad_output
            for (size_t j = 0; j < J; j++)
            {
                MAT_AT(dwB, r, j) += input_proj_scalar * MAT_AT(dvalues, n, j);
            }

            // Update grad_wA and grad_inputs (using Scalar A)
            // dL/dwA    += Input^T * (grad_output * wB^T)
            // dL/dInput += (grad_output * wB^T) * wA^T
            for (size_t i = 0; i < I; i++)
            {
                // Update Weight A Gradient
                MAT_AT(dwA, i, r) += MAT_AT(layerinputs, n, i) * grad_proj_scalar;
                // Update Input Gradient (to pass to previous layer)
                MAT_AT(dinputs, n, i) += grad_proj_scalar * MAT_AT(wA, i, r);
            }
        }
       
    }


}

/*
lora layer
x x x x
x x x x
x x x x

x x
x x
x x
x x
--------
x x x x
x x x x

=

x x x x    x x
x x x x    x x
x x x x    x x
           x x
*/





class Relu_Activation
{
public:
    Relu_Activation(int num_samples, int num_neurons);
    void forward(Mat inputs);
    void backward(Mat dvalues);

    Mat output;
    Mat layerinputs;
    Mat dinputs;
};


Relu_Activation::Relu_Activation(int num_samples, int num_neurons)
{
    dinputs = mat_alloc(num_samples, num_neurons);
    output = mat_alloc(num_samples, num_neurons);
    layerinputs = mat_alloc(num_samples, num_neurons);
}


void Relu_Activation::forward(Mat inputs)
{
    for (size_t i = 0; i < inputs.rows; i++)
    {
        for (size_t j = 0; j < inputs.cols; j++)
        {
            MAT_AT(output, i,j) = std::max(0.f, MAT_AT(inputs, i, j));
            MAT_AT(layerinputs, i, j) = MAT_AT(inputs, i, j);
        }
    }
}


void Relu_Activation::backward(Mat dvalues)
{
    for (size_t i = 0; i < dinputs.rows; i++)
    {
        for (size_t j = 0; j < dinputs.cols; j++)
        {
            MAT_AT(dinputs, i, j) = MAT_AT(dvalues, i, j);
            if (MAT_AT(layerinputs, i, j) <= 0.0f)
            {
                MAT_AT(dinputs, i, j) = 0.0f;
            }
        }
    }
}


class Sigmoid
{
public:
    Sigmoid(int num_samples, int n_out);
    void forward(Mat inputs);
    void backward(Mat dvalues);

    Mat output;
    Mat dinputs;
};


Sigmoid::Sigmoid(int num_samples, int n_out)
{
    output = mat_alloc(num_samples,  n_out);
    dinputs = mat_alloc(num_samples, n_out);
}


void Sigmoid::forward(Mat inputs)
{
    for (size_t i = 0; i < inputs.rows; i++)
    {
        for (size_t j = 0; j < inputs.cols; j++)
        {
            MAT_AT(output, i, j) = 1 / (1+ std::exp(-MAT_AT(inputs, i, j)));
        }
    }
}


void Sigmoid::backward(Mat dvalues)
{
    for (size_t i = 0; i < dinputs.rows; i++)
    {
        for (size_t j = 0; j < dinputs.cols; j++)
        {
            // MAT_AT(dinputs, i, j) = MAT_AT(dvalues, i, j) * (1 - MAT_AT(output, i, j)) * MAT_AT(output, i, j);
            float sigmoid_der = MAT_AT(output, i, j) * (1 - MAT_AT(output, i, j));
            MAT_AT(dinputs, i,j) = MAT_AT(dvalues, i, j) * sigmoid_der;
        }
    }
}


class Binarycrossentropy_loss
{
public:
    Binarycrossentropy_loss(int num_samples, int num_cols);
    void forward(Mat y_pred, Mat y_true);
    void backward(Mat dvalues,  Mat y_true);

    Mat sample_losses;
    Mat y_pred_clipped;
    Mat clipped_dvalues;
    Mat dinputs;
};


Binarycrossentropy_loss::Binarycrossentropy_loss(int num_samples, int num_cols)
{
    y_pred_clipped = mat_alloc(num_samples, num_cols);
    sample_losses = mat_alloc(num_samples, num_cols);

    //for backward
    clipped_dvalues = mat_alloc(num_samples, num_cols);
    dinputs = mat_alloc(num_samples, num_cols);
}

void Binarycrossentropy_loss::forward(Mat y_pred, Mat y_true)
{   
    for (size_t i = 0; i < y_pred.rows; i++)
    {
        MAT_AT(y_pred_clipped, i, 0) = std::clamp(MAT_AT(y_pred, i, 0), 0.0001f, 0.999f);
        MAT_AT(sample_losses, i, 0) = -1*((MAT_AT(y_true, i, 0)*std::log(MAT_AT(y_pred_clipped, i, 0))) + (1-MAT_AT(y_true, i, 0)) * std::log(1- MAT_AT(y_pred_clipped, i, 0)));
    }
    
}


void Binarycrossentropy_loss::backward(Mat dvalues, Mat y_true)
{
    for (size_t i = 0; i < dvalues.rows; i++)
    {
        // MAT_AT(clipped_dvalues, i, 0) = std::clamp(MAT_AT(dvalues, i, 0), 0.0001f, 0.999f);
        // MAT_AT(dinputs, i, 0) = (MAT_AT(clipped_dvalues, i, 0) - MAT_AT(y_true, i, 0) / (MAT_AT(clipped_dvalues, i, 0) * (1 - MAT_AT(clipped_dvalues, i, 0)) ) );
        // MAT_AT(dinputs, i, 0) /= dvalues.rows;
        MAT_AT(clipped_dvalues, i,0) = std::clamp(MAT_AT(dvalues,i,0),0.0001f, 0.999f);
        MAT_AT(dinputs,i,0) = (MAT_AT(clipped_dvalues, i, 0) - MAT_AT(y_true, i, 0)) /(MAT_AT(clipped_dvalues, i, 0) * (1 - MAT_AT(clipped_dvalues, i, 0)));
        MAT_AT(dinputs,i,0) /= dvalues.rows;

    }
    
}


class Optimizer
{
public:
    Optimizer(float learning_rate=0.01){ lr = learning_rate;}
    void update_params(LayerDense layer);
    void update_params(Lora layer);

private:
    float lr;
};


void Optimizer::update_params(LayerDense layer)
{
    for (size_t i = 0; i < layer.weights.rows; i++)
    {
        for (size_t j = 0; j < layer.weights.cols; j++)
        {
            MAT_AT(layer.weights, i, j) -= lr* MAT_AT(layer.dweights, i, j);
        }
    }
    
    if (layer.layerbias)
    {
        for (size_t i = 0; i < layer.biases.cols; i++)
        {
            MAT_AT(layer.biases, 0,i) -= lr * MAT_AT(layer.dbiases, 0, i);
        }
    }
    
}




class Activation_softmax
{
public:
    Activation_softmax(int num_samples, int num_classes);
    void forward(Mat inputs);
    void backward(Mat dvalues);

    // std::vector<Mat> jacobians;
    // Mat diagflat;
    Mat output;
    Mat dinputs;
    // Mat temp;
    // Mat temp_t;
    // Mat singleoutput;

    // Mat singledval;

    // Mat singleoutput_T;
    // Mat soutdot;

    Mat exp_vals;
    Mat exp_sum;
    Mat maxes;

    Mat dval_T;
    Mat tempjac;

};

Activation_softmax::Activation_softmax(int num_samples, int num_classes)
{
    dinputs = mat_alloc(num_samples,num_classes);
    // diagflat = mat_alloc(num_classes,num_classes);

    // mat_fill(diagflat, 0.0);
    // temp = mat_alloc(num_classes,1);
    // temp_t = mat_alloc(1,num_classes);

    exp_vals = mat_alloc(num_samples, num_classes);
    exp_sum = mat_alloc(num_samples, 1);
    maxes = mat_alloc(num_samples, 1);
    output = mat_alloc(num_samples,  num_classes);
}


void Activation_softmax::forward(Mat inputs)
{
    for (size_t i = 0; i < inputs.rows; i++)
    {
        MAT_AT(maxes, i, 0) = MAT_AT(inputs, i, 0);
        for (size_t j = 1; j < inputs.cols; j++)
        {
            if (MAT_AT(inputs, i, j) > MAT_AT(maxes, i, 0))
            {
                MAT_AT(maxes, i, 0) = MAT_AT(inputs, i, j);
            }
        }
    }
    mat_fill(exp_sum, 0.f);
    for (size_t i = 0; i < inputs.rows; i++)
    {
        for (size_t j = 0; j < inputs.cols; j++)
        {
            MAT_AT(exp_vals, i,j) = std::exp(MAT_AT(inputs, i, j) - MAT_AT(maxes, i, 0));
            MAT_AT(exp_sum, i, 0) += MAT_AT(exp_vals, i, j);
        }
        for (size_t j = 0; j < inputs.cols; j++)
        {
            MAT_AT(output, i, j) = MAT_AT(exp_vals, i, j) / MAT_AT(exp_sum, i, 0);
        }
        
        
    }
    
}


void Activation_softmax::backward(Mat dvalues)
{
    for (size_t i = 0; i < output.rows; i++)
    {
        float dot(0);
        for (size_t j = 0; j < output.cols; j++)
        {
            dot += MAT_AT(output, i, j) * MAT_AT(dvalues, i, j);
        }

        for (size_t j = 0; j < output.cols; j++)
        {
            MAT_AT(dinputs, i, j) = MAT_AT(output, i, j) * (MAT_AT(dvalues, i,  j) - dot);
        }
    }
    
}





class Loss_categoricalCrossentropy
{
public:
    Loss_categoricalCrossentropy(int num_classes, int num_samples);
    void forward(Mat y_pred, Mat y_true);
    void backward(Mat dvalues, Mat y_true);

    Mat dinputs;
    Mat negative_log_likelihood;

    Mat y_pred_clipped;
    Mat correct_confidences;
};



Loss_categoricalCrossentropy::Loss_categoricalCrossentropy(int num_classes, int num_samples)
{

    y_pred_clipped = mat_alloc(num_classes, num_classes);
    correct_confidences = mat_alloc(num_samples, 1);
    dinputs = mat_alloc(num_samples, num_classes);
    negative_log_likelihood = mat_alloc(1, num_samples);
}



void Loss_categoricalCrossentropy:: forward(Mat y_pred, Mat y_true)
{
    
    for (size_t i = 0; i < y_pred.rows; i++)
    {
        float sum = 0.f;
        for (size_t j = 0; j < y_pred.cols; j++)
        {
            MAT_AT(y_pred_clipped,i,j) = std::clamp(MAT_AT(y_pred, i, j), 0.0001f, .9999f);
            sum += MAT_AT(y_pred_clipped,i,j) * MAT_AT(y_true, i,j);
        }
        MAT_AT(correct_confidences, i,0) = sum;
        MAT_AT(negative_log_likelihood,0,i) = -std::log(MAT_AT(correct_confidences, i,0));
    }
}

void Loss_categoricalCrossentropy::backward(Mat dvalues, Mat y_true)
{
    int samples = dvalues.rows;
    for (size_t i = 0; i < dinputs.rows; i++)
    {
        for (size_t j = 0; j < dinputs.cols; j++)
        {
            MAT_AT(dinputs, i,j) = (-MAT_AT(y_true, i,j) / MAT_AT(dvalues,i,j))/(float) samples;
        }   
    }
}




class Optimizer_SGD
{
public:
    Optimizer_SGD(double learning_rate=0.01){  lr = learning_rate;}
    void update_params(LayerDense Layer);
    void update_params(Lora layer);
private:
    double lr;
};

void Optimizer_SGD::update_params(LayerDense layer)
{
    for (int i = 0; i < layer.weights.rows; i++)
    {
        for (int j = 0; j < layer.weights.cols; j++)
        {
            MAT_AT(layer.weights, i, j) -= lr* MAT_AT(layer.dweights, i, j);
        }
    }

    if (layer.layerbias)
    {
        for (int i = 0; i < layer.biases.cols; i++)
        {
            MAT_AT(layer.biases, 0, i) -= lr * MAT_AT(layer.dbiases, 0, i);
        }
    }
}



void Optimizer_SGD::update_params(Lora layer)
{

    for (size_t i = 0; i < layer.wA.rows; i++)
    {
        for (size_t j = 0; j < layer.wA.cols; j++)
        {
            MAT_AT(layer.wA, i, j) -=  lr * MAT_AT(layer.dwA, i, j);
        }
    }

    for (size_t i = 0; i < layer.wB.rows; i++)
    {
        for (size_t j = 0; j < layer.wB.cols; j++)
        {
            MAT_AT(layer.wB, i, j) -=  lr * MAT_AT(layer.dwB, i, j);
        }
    }
    

    if (layer.layerbias)
    {
        for (size_t i = 0; i < layer.biases.cols; i++)
        {
            MAT_AT(layer.biases, 0,i) -= lr * MAT_AT(layer.dbiases, 0, i);
        }
    }
}
