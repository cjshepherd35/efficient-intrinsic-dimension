#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <cassert>
#include <chrono>

// ============================================================================
// MNIST IDX file parser
// ============================================================================

uint32_t readBigEndianU32(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8)  | uint32_t(bytes[3]);
}

std::vector<std::vector<float>> loadMNISTImages(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open image file: " << filename << std::endl;
        exit(1);
    }
    uint32_t magic = readBigEndianU32(f);
    if (magic != 2051) { std::cerr << "ERROR: Invalid magic for images: " << magic << std::endl; exit(1); }
    uint32_t numImages = readBigEndianU32(f);
    uint32_t numRows   = readBigEndianU32(f);
    uint32_t numCols   = readBigEndianU32(f);
    uint32_t imgSize   = numRows * numCols;
    std::cout << "Loading " << numImages << " images (" << numRows << "x" << numCols << ")..." << std::endl;
    std::vector<std::vector<float>> images(numImages, std::vector<float>(imgSize));
    std::vector<uint8_t> buffer(imgSize);
    for (uint32_t i = 0; i < numImages; i++) {
        f.read(reinterpret_cast<char*>(buffer.data()), imgSize);
        for (uint32_t j = 0; j < imgSize; j++)
            images[i][j] = buffer[j] / 255.0f;
    }
    return images;
}

std::vector<int> loadMNISTLabels(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open label file: " << filename << std::endl;
        exit(1);
    }
    uint32_t magic = readBigEndianU32(f);
    if (magic != 2049) { std::cerr << "ERROR: Invalid magic for labels: " << magic << std::endl; exit(1); }
    uint32_t numLabels = readBigEndianU32(f);
    std::cout << "Loading " << numLabels << " labels..." << std::endl;
    std::vector<int> labels(numLabels);
    for (uint32_t i = 0; i < numLabels; i++) {
        uint8_t label;
        f.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
    }
    return labels;
}

// ============================================================================
// Matrix utilities
// ============================================================================

using Matrix = std::vector<std::vector<float>>;
using Vector = std::vector<float>;

Matrix makeMatrix(int rows, int cols, float val = 0.0f) {
    return Matrix(rows, Vector(cols, val));
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    int m = (int)A.size(), k = (int)A[0].size(), n = (int)B[0].size();
    assert((int)B.size() == k);
    Matrix C = makeMatrix(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) sum += A[i][p] * B[p][j];
            C[i][j] = sum;
        }
    return C;
}

Matrix transpose(const Matrix& A) {
    int m = (int)A.size(), n = (int)A[0].size();
    Matrix T = makeMatrix(n, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            T[j][i] = A[i][j];
    return T;
}

Matrix hadamard(const Matrix& A, const Matrix& B) {
    int m = (int)A.size(), n = (int)A[0].size();
    Matrix C = makeMatrix(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] * B[i][j];
    return C;
}

Matrix addBias(const Matrix& A, const Vector& bias) {
    Matrix B = A;
    for (int i = 0; i < (int)B.size(); i++)
        for (int j = 0; j < (int)B[0].size(); j++)
            B[i][j] += bias[j];
    return B;
}

Matrix relu(const Matrix& A) {
    Matrix B = A;
    for (auto& row : B) for (auto& v : row) v = v > 0 ? v : 0;
    return B;
}

Matrix reluDerivative(const Matrix& A) {
    Matrix B = makeMatrix((int)A.size(), (int)A[0].size());
    for (int i = 0; i < (int)A.size(); i++)
        for (int j = 0; j < (int)A[0].size(); j++)
            B[i][j] = A[i][j] > 0 ? 1.0f : 0.0f;
    return B;
}

Matrix softmax(const Matrix& A) {
    Matrix B = A;
    for (int i = 0; i < (int)B.size(); i++) {
        float maxVal = *std::max_element(B[i].begin(), B[i].end());
        float sumExp = 0.0f;
        for (auto& v : B[i]) { v = std::exp(v - maxVal); sumExp += v; }
        for (auto& v : B[i]) v /= sumExp;
    }
    return B;
}

// ============================================================================
// Neural Network with N Shared-Weight Masked Hidden Layers
// ============================================================================
//
// Architecture:
//   Input(784) -> W_input -> ReLU
//     -> [W_shared * mask[0]] -> ReLU    (shared layer 1)
//     -> [W_shared * mask[1]] -> ReLU    (shared layer 2)
//     -> ... (N shared layers total)
//     -> W_out -> Softmax -> 10 classes
//
// Each mask[i] selects maskFraction of the weights in W_shared.
// Masks are mutually exclusive (no weight appears in two masks).
// Constraint: numSharedLayers * maskFraction <= 1.0
//

class SharedMaskNN {
public:
    int inputSize;
    int hiddenSize;
    int outputSize;
    int numSharedLayers;
    float maskFraction;
    float learningRate;

    // Weights
    Matrix W_input;                    // [inputSize x hiddenSize]
    Vector b_input;                    // [hiddenSize]
    Matrix W_shared;                   // [hiddenSize x hiddenSize]
    std::vector<Vector> b_shared;      // one bias per shared layer [numSharedLayers][hiddenSize]
    Matrix W_out;                      // [hiddenSize x outputSize]
    Vector b_out;                      // [outputSize]

    // Masks: one per shared layer, mutually exclusive subsets of W_shared
    std::vector<Matrix> masks;         // [numSharedLayers][hiddenSize][hiddenSize]

    std::mt19937 rng;

    SharedMaskNN(int inputSize, int hiddenSize, int outputSize,
                 int numSharedLayers, float maskFraction, float lr, unsigned seed = 42)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize),
          numSharedLayers(numSharedLayers), maskFraction(maskFraction),
          learningRate(lr), rng(seed)
    {
        // ---- Validate mask coverage ----
        float totalCoverage = numSharedLayers * maskFraction;
        if (totalCoverage > 1.0f + 1e-6f) {
            std::cerr << "ERROR: numSharedLayers(" << numSharedLayers
                      << ") * maskFraction(" << maskFraction
                      << ") = " << totalCoverage
                      << " > 1.0. Total mask coverage cannot exceed 100% of weights."
                      << std::endl;
            exit(1);
        }

        // ---- Xavier initialization ----
        auto xavier = [&](int fanIn, int fanOut) -> Matrix {
            float scale = std::sqrt(2.0f / (fanIn + fanOut));
            std::normal_distribution<float> dist(0.0f, scale);
            Matrix W = makeMatrix(fanIn, fanOut);
            for (auto& row : W) for (auto& v : row) v = dist(rng);
            return W;
        };

        W_input  = xavier(inputSize, hiddenSize);
        W_shared = xavier(hiddenSize, hiddenSize);
        W_out    = xavier(hiddenSize, outputSize);

        b_input.assign(hiddenSize, 0.0f);
        b_shared.resize(numSharedLayers);
        for (int l = 0; l < numSharedLayers; l++)
            b_shared[l].assign(hiddenSize, 0.0f);
        b_out.assign(outputSize, 0.0f);

        // ---- Build mutually exclusive masks ----
        // Flatten all weight positions, shuffle, then assign fractions to each layer
        int totalWeights = hiddenSize * hiddenSize;
        int weightsPerMask = (int)(totalWeights * maskFraction);

        // Create shuffled indices of all weight positions
        std::vector<int> allPositions(totalWeights);
        std::iota(allPositions.begin(), allPositions.end(), 0);
        std::shuffle(allPositions.begin(), allPositions.end(), rng);

        // Initialize all masks to zero
        masks.resize(numSharedLayers);
        for (int l = 0; l < numSharedLayers; l++)
            masks[l] = makeMatrix(hiddenSize, hiddenSize, 0.0f);

        // Assign non-overlapping slices to each mask
        for (int l = 0; l < numSharedLayers; l++) {
            int start = l * weightsPerMask;
            int end   = std::min(start + weightsPerMask, totalWeights);
            for (int k = start; k < end; k++) {
                int pos = allPositions[k];
                int row = pos / hiddenSize;
                int col = pos % hiddenSize;
                masks[l][row][col] = 1.0f;
            }
        }

        // ---- Print summary ----
        std::cout << "Network initialized:" << std::endl;
        std::cout << "  Input:          " << inputSize << std::endl;
        std::cout << "  Hidden:         " << hiddenSize << std::endl;
        std::cout << "  Shared layers:  " << numSharedLayers << std::endl;
        std::cout << "  Mask fraction:  " << (maskFraction * 100.0f) << "% per layer" << std::endl;
        std::cout << "  Total coverage: " << (totalCoverage * 100.0f) << "% of W_shared" << std::endl;
        std::cout << "  Output:         " << outputSize << std::endl;

        for (int l = 0; l < numSharedLayers; l++) {
            int count = 0;
            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    count += (int)masks[l][i][j];
            std::cout << "  Mask " << l << " active weights: " << count
                      << " / " << totalWeights << std::endl;
        }

        // Verify no overlap
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                float sum = 0.0f;
                for (int l = 0; l < numSharedLayers; l++)
                    sum += masks[l][i][j];
                if (sum > 1.0f) {
                    std::cerr << "ERROR: Mask overlap detected at (" << i << "," << j << ")!" << std::endl;
                    exit(1);
                }
            }
        }
        std::cout << "  Mask overlap check: PASSED (all mutually exclusive)" << std::endl;
    }

    // ---- Forward pass ----
    // Stores all intermediates for backprop
    struct ForwardResult {
        // z[i] = pre-activation, a[i] = post-ReLU for each layer
        // Index 0 = input projection, 1..N = shared layers
        std::vector<Matrix> z;  // [numSharedLayers + 1] pre-activations
        std::vector<Matrix> a;  // [numSharedLayers + 1] activations
        Matrix z_out;           // pre-softmax logits
        Matrix probs;           // softmax output
    };

    ForwardResult forward(const Matrix& X) {
        ForwardResult r;
        int N = numSharedLayers;
        r.z.resize(N + 1);
        r.a.resize(N + 1);

        // Input projection: z[0] = X * W_input + b_input, a[0] = ReLU(z[0])
        r.z[0] = addBias(matmul(X, W_input), b_input);
        r.a[0] = relu(r.z[0]);

        // Shared layers: for layer l (1..N):
        //   z[l] = a[l-1] * (W_shared ⊙ mask[l-1]) + b_shared[l-1]
        //   a[l] = ReLU(z[l])
        for (int l = 0; l < N; l++) {
            Matrix W_masked = hadamard(W_shared, masks[l]);
            r.z[l + 1] = addBias(matmul(r.a[l], W_masked), b_shared[l]);
            r.a[l + 1] = relu(r.z[l + 1]);
        }

        // Output: z_out = a[N] * W_out + b_out
        r.z_out = addBias(matmul(r.a[N], W_out), b_out);
        r.probs = softmax(r.z_out);

        return r;
    }

    float crossEntropyLoss(const Matrix& probs, const std::vector<int>& labels) {
        float loss = 0.0f;
        int batchSize = (int)probs.size();
        for (int i = 0; i < batchSize; i++) {
            float p = std::max(probs[i][labels[i]], 1e-7f);
            loss -= std::log(p);
        }
        return loss / batchSize;
    }

    // ---- Backward pass + SGD update ----
    void backward(const Matrix& X, const std::vector<int>& labels, const ForwardResult& fwd) {
        int batchSize = (int)X.size();
        float scale = 1.0f / batchSize;
        int N = numSharedLayers;

        // ---- Output layer ----
        // dL/dz_out = probs - one_hot(labels)
        Matrix dz_out = fwd.probs;
        for (int i = 0; i < batchSize; i++)
            dz_out[i][labels[i]] -= 1.0f;

        Matrix dW_out = matmul(transpose(fwd.a[N]), dz_out);
        Vector db_out_grad(outputSize, 0.0f);
        for (int i = 0; i < batchSize; i++)
            for (int j = 0; j < outputSize; j++)
                db_out_grad[j] += dz_out[i][j];

        // da[N] = dz_out * W_out^T
        Matrix da_current = matmul(dz_out, transpose(W_out));

        // ---- Backprop through shared layers in reverse ----
        // Accumulate masked gradients for W_shared
        Matrix dW_shared_total = makeMatrix(hiddenSize, hiddenSize, 0.0f);
        std::vector<Vector> db_shared_grad(N, Vector(hiddenSize, 0.0f));

        for (int l = N - 1; l >= 0; l--) {
            // dz[l+1] = da[l+1] ⊙ relu'(z[l+1])
            Matrix dz = hadamard(da_current, reluDerivative(fwd.z[l + 1]));

            // Gradient for W_shared from this layer, masked
            // dW = a[l]^T * dz, then mask
            Matrix dW_layer = matmul(transpose(fwd.a[l]), dz);
            Matrix dW_masked = hadamard(dW_layer, masks[l]);

            // Accumulate into total W_shared gradient
            for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    dW_shared_total[i][j] += dW_masked[i][j];

            // Bias gradient for this shared layer
            for (int i = 0; i < batchSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    db_shared_grad[l][j] += dz[i][j];

            // Propagate gradient to previous activation
            // da[l] = dz * (W_shared ⊙ mask[l])^T
            Matrix W_masked = hadamard(W_shared, masks[l]);
            da_current = matmul(dz, transpose(W_masked));
        }

        // ---- Backprop through input layer ----
        // dz[0] = da[0] ⊙ relu'(z[0])
        Matrix dz_input = hadamard(da_current, reluDerivative(fwd.z[0]));
        Matrix dW_input = matmul(transpose(X), dz_input);
        Vector db_input_grad(hiddenSize, 0.0f);
        for (int i = 0; i < batchSize; i++)
            for (int j = 0; j < hiddenSize; j++)
                db_input_grad[j] += dz_input[i][j];

        // ---- SGD updates ----
        auto updateMatrix = [&](Matrix& W, const Matrix& dW) {
            for (int i = 0; i < (int)W.size(); i++)
                for (int j = 0; j < (int)W[0].size(); j++)
                    W[i][j] -= learningRate * scale * dW[i][j];
        };
        auto updateVector = [&](Vector& b, const Vector& db) {
            for (int i = 0; i < (int)b.size(); i++)
                b[i] -= learningRate * scale * db[i];
        };

        updateMatrix(W_input, dW_input);
        updateVector(b_input, db_input_grad);

        updateMatrix(W_shared, dW_shared_total);
        for (int l = 0; l < N; l++)
            updateVector(b_shared[l], db_shared_grad[l]);

        updateMatrix(W_out, dW_out);
        updateVector(b_out, db_out_grad);
    }

    std::vector<int> predict(const Matrix& X) {
        ForwardResult fwd = forward(X);
        std::vector<int> preds(X.size());
        for (int i = 0; i < (int)X.size(); i++)
            preds[i] = (int)(std::max_element(fwd.probs[i].begin(), fwd.probs[i].end()) - fwd.probs[i].begin());
        return preds;
    }

    float accuracy(const Matrix& X, const std::vector<int>& labels) {
        std::vector<int> preds = predict(X);
        int correct = 0;
        for (int i = 0; i < (int)labels.size(); i++)
            if (preds[i] == labels[i]) correct++;
        return (float)correct / labels.size();
    }
};

// ============================================================================
// Main
// ============================================================================

int main() {
    // ---- Configuration ----
    std::string labelsFilename = "C:/Users/cshep/Downloads/t10k-labels.idx1-ubyte";
    std::string imagesFilename = "C:/Users/cshep/Downloads/t10k-images.idx3-ubyte/t10k-images.idx3-ubyte";

    int    hiddenSize      = 128;
    int    numSharedLayers  = 2;      // number of times W_shared is reused
    float  maskFraction    = 0.4f;    // each layer updates 30% of W_shared
    float  lr              = 0.05f;
    int    epochs          = 1000;
    int    sampleSize      = 16;      // random sample per SGD step

    // ---- Load MNIST ----
    auto images = loadMNISTImages(imagesFilename);
    auto labels = loadMNISTLabels(labelsFilename);
    int totalSamples = (int)images.size();
    std::cout << "Total samples loaded: " << totalSamples << std::endl;

    // ---- Train/test split (80/20) ----
    int trainSize = (int)(totalSamples * 0.8);
    int testSize  = totalSamples - trainSize;

    std::vector<int> indices(totalSamples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 shuffleRng(123);
    std::shuffle(indices.begin(), indices.end(), shuffleRng);

    Matrix trainImages(trainSize), testImages(testSize);
    std::vector<int> trainLabels(trainSize), testLabels(testSize);
    for (int i = 0; i < trainSize; i++) {
        trainImages[i] = images[indices[i]];
        trainLabels[i] = labels[indices[i]];
    }
    for (int i = 0; i < testSize; i++) {
        testImages[i] = images[indices[trainSize + i]];
        testLabels[i] = labels[indices[trainSize + i]];
    }
    std::cout << "Train: " << trainSize << ", Test: " << testSize << std::endl;

    // ---- Create network ----
    SharedMaskNN net(784, hiddenSize, 10, numSharedLayers, maskFraction, lr);

    // ---- Training (SGD with random samples) ----
    std::cout << "\n=== Training (SGD, sample=" << sampleSize
              << ", epochs=" << epochs << ") ===" << std::endl;
    std::uniform_int_distribution<int> sampleDist(0, trainSize - 1);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Randomly sample images
        Matrix batchX(sampleSize);
        std::vector<int> batchY(sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            int idx = sampleDist(shuffleRng);
            batchX[i] = trainImages[idx];
            batchY[i] = trainLabels[idx];
        }

        auto fwd = net.forward(batchX);
        float loss = net.crossEntropyLoss(fwd.probs, batchY);
        net.backward(batchX, batchY, fwd);

        if ((epoch + 1) % 50 == 0 || epoch == 0 || epoch == epochs - 1) {
            float trainAcc = net.accuracy(trainImages, trainLabels);
            float testAcc  = net.accuracy(testImages, testLabels);
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " | Loss: " << loss
                      << " | Train Acc: " << (trainAcc * 100.0f) << "%"
                      << " | Test Acc: "  << (testAcc * 100.0f) << "%"
                      << std::endl;
        }
    }

    // ---- Final evaluation ----
    std::cout << "\n=== Final Evaluation ===" << std::endl;
    float finalTrainAcc = net.accuracy(trainImages, trainLabels);
    float finalTestAcc  = net.accuracy(testImages, testLabels);
    std::cout << "Train Accuracy: " << (finalTrainAcc * 100.0f) << "%" << std::endl;
    std::cout << "Test Accuracy:  " << (finalTestAcc * 100.0f) << "%" << std::endl;

    // ---- Sample predictions ----
    std::cout << "\n=== Sample Predictions (first 20 test images) ===" << std::endl;
    Matrix sampleX(testImages.begin(), testImages.begin() + 20);
    std::vector<int> sampleY(testLabels.begin(), testLabels.begin() + 20);
    auto samplePreds = net.predict(sampleX);
    for (int i = 0; i < 20; i++) {
        std::cout << "  True: " << sampleY[i] << " | Predicted: " << samplePreds[i]
                  << (sampleY[i] == samplePreds[i] ? " ok" : " WRONG") << std::endl;
    }

    return 0;
}
