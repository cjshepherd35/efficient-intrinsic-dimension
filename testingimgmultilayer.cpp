#include "imagegathering.cpp"
#include <tuple>

int num_samples = 10;
double lr = 0.01;
int n_inputs = 28*28;
int  n_neurons = 256;
int num_classes = 10;
int num_iters = 401;
int check_iter = 50;
float percent_weights = 0.4f;
int numlayers = 2;

std::vector<std::vector<std::tuple<int, int>>> generate_random_tuples(int numlayers, float percent_weights, int samples, int inputs)
{
    // 1. Create a pool of all possible unique tuples within the ranges
    std::vector<std::tuple<int, int>> pool;
    pool.reserve(samples * inputs);
    for (int i = 0; i < samples; ++i)
    {
        for (int j = 0; j < inputs; ++j)
        {
            pool.emplace_back(i, j);
        }
    }

    // 2. Shuffle the pool to ensure random, unique selection
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(pool.begin(), pool.end(), g);

    // 3. Partition the pool into numlayers vectors
    int count_per_layer = static_cast<int>(percent_weights * samples * inputs);
    // Safety check to ensure we don't exceed the pool size
    if (numlayers * count_per_layer > samples * inputs) count_per_layer = (samples * inputs) / numlayers;

    std::vector<std::vector<std::tuple<int, int>>> result;
    std::vector<std::tuple<int, int>> all_appended; // This vector appends all generated tuples

    auto current_it = pool.begin();
    for (int l = 0; l < numlayers; ++l)
    {
        std::vector<std::tuple<int, int>> layer_vec(current_it, current_it + count_per_layer);
        std::sort(layer_vec.begin(), layer_vec.end()); // Sort ascending by first, then second
        result.push_back(layer_vec);
        all_appended.insert(all_appended.end(), layer_vec.begin(), layer_vec.end());
        current_it += count_per_layer;
    }

    return result;
}



int main()
{

    std::vector<std::vector<std::tuple<int,int>>> randmatvecs = generate_random_tuples(numlayers, percent_weights, num_samples, n_inputs);
    
    
    srand(69);
    std::vector<std::vector<float>> images = read_images(imagesfilename);
    std::vector<std::vector<int>> labels = read_labels(labelsfilename, num_classes);

    Mat trainingimages = mat_alloc(num_samples, n_inputs);
    Mat trainlabels = mat_alloc(num_samples, num_classes);

    LayerDense ld1(n_inputs, n_neurons, false, num_samples);
    Relu_Activation relu(num_samples, n_neurons);
    Multilayer ld2(n_neurons, n_neurons, false, num_samples, randmatvecs, numlayers, percent_weights);
    Relu_Activation relu2(num_samples, n_neurons);
    Relu_Activation relu3(num_samples, n_neurons);
    LayerDense ld3(n_neurons, num_classes, false, num_samples);
    
    Activation_softmax soft(num_samples, num_classes);
    Loss_categoricalCrossentropy loss(num_classes, num_samples);
    
    Optimizer_SGD opt(lr);
    Fewopt fewopt(lr);
    std::vector<int> randvec;
    for (size_t i = 0; i < num_iters; i++)
    {
        randvec = getrandvec(num_samples, num_samples);
        trainingimages = randomizeImages(images, randvec, num_samples, n_inputs);
        trainlabels = randomizeLabels(labels, randvec, num_samples, num_classes);
        
        ld1.forward(trainingimages);
        relu.forward(ld1.output);
        ld2.forward(relu.output, 0);
        relu2.forward(ld2.outputs[0]);
        ld2.forward(relu2.output, 1);
        relu3.forward(ld2.outputs[1]);
        ld3.forward(relu3.output);
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
        relu3.backward(ld3.dinputs);

        ld2.backward(relu3.dinputs,1);
        relu2.backward(ld2.dinputs);
        ld2.backward(relu2.dinputs,0);
        relu.backward(ld2.dinputs);
        
        ld1.backward(relu.dinputs);

        opt.update_params(ld3);
        
        fewopt.update_params(ld2, randmatvecs);
        opt.update_params(ld1);
        
    }
    


}