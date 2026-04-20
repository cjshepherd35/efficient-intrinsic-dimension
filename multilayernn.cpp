#include "nn.cpp"

class Multilayer
{
public:
    Multilayer(int n_inputs, int n_out, bool bias, int num_samples, const std::vector<std::vector<std::tuple<int, int>>> &randgrads, int numlayers, float percentweights);
    // LayerDense(int embed_dim, int n_out, bool bias, int batch_size, int seq_len);
    void forward(Mat &inputs, int layernum);
    // void forward(Tensor inputs);
    
    void backward(Mat dvalues, int layernum);
    
    int num_layers;
    std::vector<Mat> outputs;
    
    Mat dinputs;
    std::vector<Mat> dbiases;

//private:
    Mat weights;
    Mat weights_T;
    Mat biases;
    bool layerbias;
    int n_samples;
    
    std::vector<Mat> dweights;
    // Mat doubleweights;
    std::vector<Mat*> layerinputs;
    Mat layerinputs_T;
    const std::vector<std::vector<std::tuple<int, int>>>* fewgrads;
    Mat curbiases;
    std::vector<Mat> layer_results;
    std::vector<Mat> layer_dinputs;
};

Multilayer::Multilayer(int n_inputs, int n_out, bool bias, int num_samples, const std::vector<std::vector<std::tuple<int, int>>> &randgrads, int numlayers, float percentweights)
{
    assert(numlayers*percentweights < 1);
    num_layers = numlayers;
    n_samples = num_samples;
    layerbias = bias;
    weights = mat_alloc(n_inputs, n_out);
    mat_rand(weights, -0.1, 0.1);
    weights_T = mat_alloc(n_out, n_inputs);
    if (layerbias)
    {
        biases = mat_alloc(num_layers,n_out);
        mat_rand(biases, -1.0,1.0);
        curbiases = mat_alloc(1,n_out);
    }
    dinputs = mat_alloc(num_samples,n_inputs);
    layerinputs_T = mat_alloc(n_inputs, num_samples);
    for (size_t i = 0; i < numlayers; i++)
    {
        outputs.push_back(mat_alloc(num_samples, n_out));
        layerinputs.push_back(nullptr);
    }
    
    
    fewgrads = &randgrads;

    for (int i = 0; i < num_layers; ++i)
    {
        dweights.push_back(mat_alloc(n_inputs, n_out));
        if (layerbias) dbiases.push_back(mat_alloc(1, n_out));
        if (i < num_layers - 1)
        {
            layer_results.push_back(mat_alloc(num_samples, n_out));
            layer_dinputs.push_back(mat_alloc(num_samples, n_inputs));
        }
    }
}

void Multilayer::forward(Mat& inputs, int layernum)
{
    assert(layernum <= num_layers);
    layerinputs[layernum] = &inputs;
     
    if(layerbias)
    {
        mat_dot_bias(outputs[layernum], inputs, weights, curbiases);
    }
    else
    {
        mat_dot(outputs[layernum], inputs, weights);
    }
}


void Multilayer::backward(Mat dvalues, int layernum)
{
    assert(layernum <= num_layers);
   mat_fill(dweights[layernum], 0.f);
   mat_transpose(layerinputs_T, *layerinputs[layernum]);
   mat_dot(dweights[layernum], layerinputs_T, dvalues);

    if (layerbias)
    {
        for (size_t i = 0; i < dvalues.cols; i++)
        {
            for (size_t j = 0; j < dvalues.rows; j++)
            {
                MAT_AT(dbiases[layernum], 0, i) += MAT_AT(dvalues, j, i);
            }
        }
    }

    mat_transpose(weights_T, weights);
    mat_dot(dinputs, dvalues, weights_T);
}



class Fewopt
{
public:
    float learning_rate;
    Fewopt(float lr){learning_rate = lr;}
    void update_params(Multilayer layer, std::vector<std::vector<std::tuple<int,int>>> randvecs);
};


void Fewopt::update_params(Multilayer layer, std::vector<std::vector<std::tuple<int,int>>> randvecs)
{
    for (size_t i = 0; i < randvecs.size(); i++)
    {
        for(const auto& [j, k] : randvecs[i])
        {
            MAT_AT(layer.weights,j,k) -= MAT_AT(layer.dweights[i],j,k)*learning_rate;
        }
    }
    
}
