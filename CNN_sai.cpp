#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <bits/stdc++.h>
#include <random>

using namespace std;

// ===== MATRIX CLASS ===== //
class Matrix {
public:
    size_t rows;
    size_t cols;
    std::vector<std::vector<float>> data;

    Matrix() : rows(0), cols(0) {}

    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data = std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));
    }

    // Return reference for modification
    std::vector<float>& operator[](size_t index) {
        return data[index];
    }
    const std::vector<float>& operator[](size_t index) const {
        return data[index];
    }

    Matrix operator+(const Matrix& other) const {
        if(rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrix shapes do not match for addition");
        Matrix sum(rows, cols);
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                sum[i][j] = data[i][j] + other[i][j];
            }
        }
        return sum;
    }

    Matrix operator-(const Matrix& other) const {
        if(rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrix shapes do not match for subtraction");
        Matrix diff(rows, cols);
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                diff[i][j] = data[i][j] - other[i][j];
            }
        }
        return diff;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("Matrices do not match shape for multiplication");
        Matrix prod(rows, other.cols);
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < other.cols; j++){
                float sum = 0;
                for (size_t k = 0; k < cols; k++){
                    sum += data[i][k] * other[k][j];
                }
                prod[i][j] = sum;
            }
        }
        return prod;
    }

    Matrix operator*(float other) const {
        Matrix prod(rows, cols);
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                prod[i][j] = data[i][j] * other;
            }
        }
        return prod;
    }

    Matrix T() const {
        Matrix transposed(cols, rows);
        for (size_t i = 0; i < rows; ++i){
            for (size_t j = 0; j < cols; ++j){
                transposed[j][i] = data[i][j];
            }
        }
        return transposed;
    }
};

// ===== FULLY CONNECTED LAYER ===== //
class FullyConnected {
public:
    int input_shape;
    int output_shape;
    Matrix weight;
    Matrix bias;
    Matrix last_input;

    FullyConnected(int input_shape, int output_shape)
        : input_shape(input_shape), output_shape(output_shape),
          weight(output_shape, input_shape), bias(output_shape, 1) {
        // Initialize weights and biases (for example, with small random values)
        std::default_random_engine generator;
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (size_t i = 0; i < weight.rows; i++){
            for (size_t j = 0; j < weight.cols; j++){
                weight[i][j] = dist(generator);
            }
        }
        for (size_t i = 0; i < bias.rows; i++){
            bias[i][0] = dist(generator);
        }
    }

    Matrix forward(const Matrix& input) {
        if(input.rows != (size_t)input_shape)
            throw std::invalid_argument("Input does not match shape in FullyConnected");
        Matrix output = weight * input;
        output = output + bias;
        last_input = input;
        return output;
    }

    Matrix backward(const Matrix& output_gradient, float learning_rate) {
        if (output_gradient.rows != (size_t)output_shape)
            throw std::invalid_argument("Output gradient does not match shape in FullyConnected");
        Matrix input_gradient = weight.T() * output_gradient;
        Matrix weight_gradient = output_gradient * last_input.T();
        Matrix bias_gradient = output_gradient;
        weight = weight - weight_gradient * learning_rate;
        bias = bias - bias_gradient * learning_rate;
        return input_gradient;
    }
};

// ===== ACTIVATION: LEAKY RELU ===== //
class ReLU_Leaky {
private:
    float coeff = 0.01f;
    float coeffp = 1.0f / coeff;
public:
    int input_shape;
    int output_shape;
    Matrix last_input;

    ReLU_Leaky (int input_shape) : input_shape(input_shape), output_shape(input_shape) {}

    Matrix forward(const Matrix& input) {
        Matrix output(input.rows, input.cols);
        for(size_t i = 0; i < input.rows; i++){
            for(size_t j = 0; j < input.cols; j++){
                output[i][j] = input[i][j] > 0 ? input[i][j] : coeff * input[i][j];
            }
        }
        last_input = input;
        return output;
    }

    Matrix backward(const Matrix& output_gradient, float learning_rate) {
        if (output_gradient.rows != (size_t)output_shape)
            throw std::invalid_argument("Output gradient does not match shape in ReLU_Leaky");
        Matrix input_gradient(last_input.rows, last_input.cols);
        for(size_t i = 0; i < last_input.rows; i++){
            for(size_t j = 0; j < last_input.cols; j++){
                input_gradient[i][j] = last_input[i][j] > 0 ? output_gradient[i][j] : coeffp * output_gradient[i][j];
            }
        }
        return input_gradient;
    }
};

// ===== LOSS: BINARY CROSS-ENTROPY ===== //
class BinaryCrossEntropyLoss {
public:
    BinaryCrossEntropyLoss() {}

    float forward(const Matrix& predicted, const Matrix& target) {
        if (predicted.rows != target.rows || predicted.cols != target.cols)
            throw std::invalid_argument("Predicted and target matrices must have the same dimensions.");
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.rows; i++){
            float predicted_prob = std::max(std::min(predicted[i][0], 1.0f), 1e-15f);
            loss -= target[i][0] * std::log(predicted_prob) + (1 - target[i][0]) * std::log(1 - predicted_prob);
        }
        return loss / predicted.rows;
    }

    Matrix backward(const Matrix& predicted, const Matrix& target) {
        if (predicted.rows != target.rows || predicted.cols != target.cols)
            throw std::invalid_argument("Predicted and target matrices must have the same dimensions.");
        Matrix gradient(predicted.rows, predicted.cols);
        for (size_t i = 0; i < predicted.rows; i++){
            float predicted_prob = std::max(std::min(predicted[i][0], 1.0f), 1e-15f);
            gradient[i][0] = (predicted_prob - target[i][0]) / (predicted_prob * (1 - predicted_prob));
        }
        return gradient * (1.0f / predicted.rows);
    }
};

// ===== ACTIVATION: SOFTMAX ===== //
class Softmax {
public:
    int input_shape;
    int output_shape;
    Matrix last_output;

    Softmax(int input_shape) : input_shape(input_shape), output_shape(input_shape) {}

    Matrix forward(const Matrix& input) {
        if (input.rows != (size_t)input_shape || input.cols != 1)
            throw std::invalid_argument("Input must be column vector of size input_shape x 1 in Softmax");
        Matrix output(input.rows, 1);
        float max_val = input[0][0];
        for (size_t i = 0; i < input.rows; ++i) {
            if (input[i][0] > max_val)
                max_val = input[i][0];
        }
        float sum_exp = 0.0f;
        for (size_t i = 0; i < input.rows; ++i) {
            output[i][0] = exp(input[i][0] - max_val);
            sum_exp += output[i][0];
        }
        for (size_t i = 0; i < input.rows; ++i)
            output[i][0] /= sum_exp;
        last_output = output;
        return output;
    }

    Matrix backward(const Matrix& output_gradient, float learning_rate) {
        Matrix input_gradient(input_shape, 1);
        float sum = 0.0f;
        for (size_t i = 0; i < input_shape; ++i)
            sum += last_output[i][0] * output_gradient[i][0];
        for (size_t i = 0; i < input_shape; ++i)
            input_gradient[i][0] = last_output[i][0] * (output_gradient[i][0] - sum);
        return input_gradient;
    }
};

// ===== CONVOLUTION LAYER ===== //
class Conv2D {
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<Matrix> kernels;
    std::vector<float> biases;
    Matrix last_input;
    Matrix last_input_padded;

    Matrix pad_matrix(const Matrix& input, int pad) const {
        Matrix padded(input.rows + 2 * pad, input.cols + 2 * pad);
        for (size_t i = 0; i < input.rows; ++i) {
            for (size_t j = 0; j < input.cols; ++j) {
                padded[i + pad][j + pad] = input[i][j];
            }
        }
        return padded;
    }

public:
    // For simplicity, assume single-channel input (e.g., grayscale)
    Conv2D(int in_ch, int out_ch, int k_size, int stride = 1, int pad = 0)
        : in_channels(in_ch), out_channels(out_ch),
          kernel_size(k_size), stride(stride), padding(pad) {
        float stddev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
        std::default_random_engine generator;
        std::normal_distribution<float> dist(0.0f, stddev);
        kernels.reserve(out_channels);
        for (int i = 0; i < out_channels; ++i) {
            Matrix kernel(kernel_size, kernel_size);
            for (int h = 0; h < kernel_size; ++h)
                for (int w = 0; w < kernel_size; ++w)
                    kernel[h][w] = dist(generator);
            kernels.push_back(kernel);
        }
        biases = std::vector<float>(out_channels, 0.1f);
    }

    // Forward: returns a vector of feature maps (one per output channel) with 2D shape.
    std::vector<Matrix> forward(const Matrix& input) {
        last_input = input;
        last_input_padded = pad_matrix(input, padding);
        int out_h = (input.rows - kernel_size + 2 * padding) / stride + 1;
        int out_w = (input.cols - kernel_size + 2 * padding) / stride + 1;
        std::vector<Matrix> output;
        // For each output channel, create an output feature map.
        for (int k = 0; k < out_channels; ++k) {
            Matrix featureMap(out_h, out_w);
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    float sum = 0.0f;
                    int h_start = i * stride;
                    int w_start = j * stride;
                    for (int x = 0; x < kernel_size; ++x) {
                        for (int y = 0; y < kernel_size; ++y) {
                            sum += last_input_padded[h_start + x][w_start + y] * kernels[k][x][y];
                        }
                    }
                    featureMap[i][j] = sum + biases[k];
                }
            }
            output.push_back(featureMap);
        }
        return output;
    }

    // Backward pass omitted for brevity.
};

// ===== POOLING LAYER ===== //
// Implements both max and average pooling on each 2D feature map.
class PoolingLayer {
private:
    int poolSize;
    int stride;
    std::string poolingType; // "max" or "average"
public:
    PoolingLayer(int poolSize, int stride, const std::string& poolingType = "max")
        : poolSize(poolSize), stride(stride), poolingType(poolingType) {}

    // Forward: Applies pooling on each feature map in the input vector.
    std::vector<Matrix> forward(const std::vector<Matrix>& input_feature_maps) {
        std::vector<Matrix> pooled_maps;
        for (const auto& featureMap : input_feature_maps) {
            int inputRows = featureMap.rows;
            int inputCols = featureMap.cols;
            int outputRows = ((inputRows - poolSize) / stride) + 1;
            int outputCols = ((inputCols - poolSize) / stride) + 1;
            Matrix pooled(outputRows, outputCols);
            for (int i = 0; i < outputRows; ++i) {
                for (int j = 0; j < outputCols; ++j) {
                    float poolResult = (poolingType == "max") ? -std::numeric_limits<float>::infinity() : 0.0f;
                    for (int m = 0; m < poolSize; ++m) {
                        for (int n = 0; n < poolSize; ++n) {
                            int row = i * stride + m;
                            int col = j * stride + n;
                            if (row < inputRows && col < inputCols) {
                                if (poolingType == "max") {
                                    poolResult = std::max(poolResult, featureMap.data[row][col]);
                                } else if (poolingType == "average") {
                                    poolResult += featureMap.data[row][col];
                                }
                            }
                        }
                    }
                    if (poolingType == "average")
                        poolResult /= (poolSize * poolSize);
                    pooled[i][j] = poolResult;
                }
            }
            pooled_maps.push_back(pooled);
        }
        return pooled_maps;
    }
    // Backward pass can be implemented similarly if needed.
};

// ===== MAIN FUNCTION: DEMONSTRATION & TRAINING TEMPLATE ===== //
int main() {
    // -------------------------------
    // Dummy parameters (for demonstration)
    // Assume EMNIST images are 28x28 grayscale.
    const int imageHeight = 28;
    const int imageWidth = 28;
    const int numClasses = 10; // e.g., using a subset; EMNIST can have more classes.
    
    // Dummy single image data (normally load from EMNIST)
    Matrix image(imageHeight, imageWidth);
    // Fill with dummy values
    for (size_t i = 0; i < image.rows; i++){
        for (size_t j = 0; j < image.cols; j++){
            image[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // -------------------------------
    // Build CNN pipeline:
    // Conv2D -> Pooling -> Flatten -> FullyConnected -> Softmax
    Conv2D convLayer(1, 8, 3, 1, 1);  // 1 input channel, 8 output channels, 3x3 kernels, padding=1 to preserve dimensions.
    PoolingLayer poolLayer(2, 2, "max"); // 2x2 max pooling.

    // Forward pass through convolution layer (output is vector<Matrix> of feature maps)
    std::vector<Matrix> convOutput = convLayer.forward(image);
    // Apply pooling on each feature map
    std::vector<Matrix> pooledOutput = poolLayer.forward(convOutput);

    // Flatten pooled feature maps into a single column vector for fully connected layer.
    // Calculate total features from pooled maps.
    int totalFeatures = 0;
    for (const auto& fmap : pooledOutput)
        totalFeatures += fmap.rows * fmap.cols;

    Matrix flattened(totalFeatures, 1);
    int index = 0;
    for (const auto& fmap : pooledOutput) {
        for (size_t i = 0; i < fmap.rows; i++){
            for (size_t j = 0; j < fmap.cols; j++){
                flattened[index++][0] = fmap.data[i][j];
            }
        }
    }

    // Create fully connected layer for classification (CNN solution)
    FullyConnected fcCNN(totalFeatures, numClasses);
    // Softmax activation for probability distribution
    Softmax softmaxCNN(numClasses);

    // Forward pass for CNN pipeline
    Matrix fcOutputCNN = fcCNN.forward(flattened);
    Matrix probabilitiesCNN = softmaxCNN.forward(fcOutputCNN);

    cout << "CNN Prediction (probabilities):" << endl;
    for (size_t i = 0; i < probabilitiesCNN.rows; i++){
        cout << probabilitiesCNN[i][0] << " ";
    }
    cout << "\n" << endl;

    // -------------------------------
    // Build non-CNN (fully connected) pipeline:
    // Flatten image directly -> FullyConnected -> Softmax
    Matrix flattenedImage(imageHeight * imageWidth, 1);
    index = 0;
    for (size_t i = 0; i < image.rows; i++){
        for (size_t j = 0; j < image.cols; j++){
            flattenedImage[index++][0] = image[i][j];
        }
    }
    FullyConnected fcNonCNN(imageHeight * imageWidth, numClasses);
    Softmax softmaxNonCNN(numClasses);
    Matrix fcOutputNonCNN = fcNonCNN.forward(flattenedImage);
    Matrix probabilitiesNonCNN = softmaxNonCNN.forward(fcOutputNonCNN);

    cout << "Non-CNN (Fully Connected) Prediction (probabilities):" << endl;
    for (size_t i = 0; i < probabilitiesNonCNN.rows; i++){
        cout << probabilitiesNonCNN[i][0] << " ";
    }
    cout << "\n" << endl;

    // -------------------------------
    // Note: To complete the project, you will need to implement:
    //   - Proper data loading for the EMNIST dataset.
    //   - A training loop that performs forward and backward passes.
    //   - Evaluation code to compare accuracy between the CNN and non-CNN solutions.
    //   - A detailed report covering theoretical aspects and analysis of your results.
    //
    // The code below is a template to guide you in integrating your modules.
    // Ensure you test and modify the code to meet your project's requirements.

    return 0;
}
