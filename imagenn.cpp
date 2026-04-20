#include "imagegathering.cpp"
#include <cmath>

int num_samples = 10;
double lr = 0.01;
int n_inputs = 28*28;
int  n_neurons = 6;
int num_classes = 10;
int num_iters = 201;
int check_iter = 50;
int rank = 4;



void calc_weight_stats(LayerDense layer, int layernum)
{
    float greatest_dw = 0;
    float least_dw = 100.f;
    float avg_dw = 0;
    float sum = 0;
    float sum_sq = 0;
    int n = layer.dweights.rows * layer.dweights.cols;

    for (size_t j = 0; j < layer.dweights.rows; j++)
    {
        for (size_t k = 0; k < layer.dweights.cols; k++)
        {
            float val = MAT_AT(layer.dweights, j, k);
            float abs_val = std::abs(val);
            if (abs_val > greatest_dw)
            {
                greatest_dw = abs_val;
            }
            if (abs_val < least_dw)
            {
                least_dw = abs_val;
            }
            avg_dw += abs_val;
            sum += val;
            sum_sq += val * val;
        }
    }
    avg_dw /= n;
    float mean = sum / n;
    float variance = (sum_sq / n) - (mean * mean);
    float std_dev = std::sqrt(std::max(0.0f, variance));
    std::cout << "layer: " << layernum << "avg: " << avg_dw << " min: " << least_dw << " max: " << greatest_dw << " std: " << std_dev << std::endl;
}

int main()
{
    
    srand(69);
    std::vector<std::vector<float>> images = read_images(imagesfilename);
    std::vector<std::vector<int>> labels = read_labels(labelsfilename, num_classes);

    Mat trainingimages = mat_alloc(num_samples, n_inputs);
    Mat trainlabels = mat_alloc(num_samples, num_classes);

    LayerDense ld1(n_inputs, n_neurons, false, num_samples);
    Relu_Activation relu(num_samples, n_neurons);
    LayerDense ld2(n_neurons, n_neurons, false, num_samples);
    Relu_Activation relu2(num_samples, n_neurons);
    LayerDense ld3(n_neurons, num_classes, false, num_samples);
    
    Activation_softmax soft(num_samples, num_classes);
    Loss_categoricalCrossentropy loss(num_classes, num_samples);
    
    Optimizer_SGD opt(lr);
    std::vector<int> randvec;

    for (size_t i = 0; i < num_iters; i++)
    {
        
        randvec = getrandvec(num_samples, num_samples);
        
        trainingimages = randomizeImages(images, randvec, num_samples, n_inputs);
        
        trainlabels = randomizeLabels(labels, randvec, num_samples, num_classes);
        
        ld1.forward(trainingimages);
        relu.forward(ld1.output);
        ld2.forward(relu.output);
        relu2.forward(ld2.output);
        ld3.forward(relu2.output);
        soft.forward(ld3.output);
       
        loss.forward(soft.output, trainlabels);
         if (i% check_iter ==0)
        {
            printf("iter: %d\n", i);
            float total_loss = 0.0;
            
            for (size_t l = 0; l < loss.negative_log_likelihood.cols; l++)
            {
                total_loss += MAT_AT(loss.negative_log_likelihood, 0, l);
            }
            float avg_loss = total_loss/num_samples;
            printf("average loss: %f\n", avg_loss);
        }

        loss.backward(soft.output, trainlabels);
        
        soft.backward(loss.dinputs);
        
        ld3.backward(soft.dinputs);
        relu2.backward(ld3.dinputs);
        ld2.backward(relu2.dinputs);
        relu.backward(ld2.dinputs);
        ld1.backward(relu.dinputs);

        if (i == 1 | i == 200)
        {
           calc_weight_stats(ld1, 1);
           calc_weight_stats(ld3, 3); 
        }
        

        opt.update_params(ld3);
        opt.update_params(ld2);
        opt.update_params(ld1);
        
    }
    
}