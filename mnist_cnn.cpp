#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <random>
#include <cstdint>
#include <string>
#include <sstream>
#include <utility>

using namespace std;

// =====================================================
// ================ MATRIX CLASS =======================
// =====================================================
class Matrix {
public:
    size_t rows, cols;
    vector<vector<float>> data;
    
    Matrix() : rows(0), cols(0) {}
    
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data = vector<vector<float>>(rows, vector<float>(cols, 0.0f));
    }
    
    vector<float>& operator[](size_t index) {
        return data[index];
    }
    
    const vector<float>& operator[](size_t index) const {
        return data[index];
    }
    
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw invalid_argument("Matrix shapes do not match for addition");
        Matrix sum(rows, cols);
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                sum[i][j] = data[i][j] + other[i][j];
        return sum;
    }
    
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw invalid_argument("Matrix shapes do not match for subtraction");
        Matrix diff(rows, cols);
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                diff[i][j] = data[i][j] - other[i][j];
        return diff;
    }
    
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw invalid_argument("Matrices do not match shape for multiplication");
        Matrix prod(rows, other.cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
                float sum = 0;
                for (size_t k = 0; k < cols; k++)
                    sum += data[i][k] * other[k][j];
                prod[i][j] = sum;
            }
        }
        return prod;
    }
    
    Matrix operator*(float scalar) const {
        Matrix prod(rows, cols);
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                prod[i][j] = data[i][j] * scalar;
        return prod;
    }
    
    Matrix T() const {
        Matrix transposed(cols, rows);
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                transposed[j][i] = data[i][j];
        return transposed;
    }
};

// =====================================================
// =========== FULLY CONNECTED LAYER ===================
// =====================================================
class FullyConnected {
public:
    int input_shape, output_shape;
    Matrix weight;
    Matrix bias;
    Matrix last_input;
    
    FullyConnected(int input_shape, int output_shape)
        : input_shape(input_shape), output_shape(output_shape),
          weight(output_shape, input_shape), bias(output_shape, 1) {
        default_random_engine generator;
        uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (size_t i = 0; i < weight.rows; i++)
            for (size_t j = 0; j < weight.cols; j++)
                weight[i][j] = dist(generator);
        for (size_t i = 0; i < bias.rows; i++)
            bias[i][0] = dist(generator);
    }
    
    Matrix forward(const Matrix& input) {
        if (input.rows != (size_t)input_shape)
            throw invalid_argument("Input does not match shape in FullyConnected");
        Matrix output = weight * input;
        output = output + bias;
        last_input = input;
        return output;
    }
    
    Matrix backward(const Matrix& output_gradient, float learning_rate) {
        if (output_gradient.rows != (size_t)output_shape)
            throw invalid_argument("Output gradient does not match shape in FullyConnected");
        Matrix input_gradient = weight.T() * output_gradient;
        Matrix weight_gradient = output_gradient * last_input.T();
        Matrix bias_gradient = output_gradient;
        weight = weight - weight_gradient * learning_rate;
        bias = bias - bias_gradient * learning_rate;
        return input_gradient;
    }
};

// =====================================================
// =============== ACTIVATIONS =========================
// =====================================================

// ---- Leaky ReLU Activation ----
class ReLU_Leaky {
private:
    float coeff = 0.01f;
    float coeffp = 1.0f / coeff;
public:
    int input_shape, output_shape;
    Matrix last_input;
    
    ReLU_Leaky(int input_shape) : input_shape(input_shape), output_shape(input_shape) {}
    
    Matrix forward(const Matrix& input) {
        Matrix output(input.rows, input.cols);
        for (size_t i = 0; i < input.rows; i++)
            for (size_t j = 0; j < input.cols; j++)
                output[i][j] = (input[i][j] > 0) ? input[i][j] : coeff * input[i][j];
        last_input = input;
        return output;
    }
    
    Matrix backward(const Matrix& output_gradient, float learning_rate) {
        if (output_gradient.rows != last_input.rows || output_gradient.cols != last_input.cols)
            throw invalid_argument("Gradient shape mismatch in ReLU_Leaky backward");
        Matrix input_gradient(last_input.rows, last_input.cols);
        for (size_t i = 0; i < last_input.rows; i++)
            for (size_t j = 0; j < last_input.cols; j++)
                input_gradient[i][j] = (last_input[i][j] > 0) ? output_gradient[i][j] : coeffp * output_gradient[i][j];
        return input_gradient;
    }
};

// ---- Softmax Activation ----
class Softmax {
public:
    int input_shape, output_shape;
    Matrix last_output;
    
    Softmax(int input_shape) : input_shape(input_shape), output_shape(input_shape) {}
    
    Matrix forward(const Matrix& input) {
        if (input.rows != (size_t)input_shape || input.cols != 1)
            throw invalid_argument("Input must be column vector in Softmax");
        Matrix output(input.rows, 1);
        float max_val = input[0][0];
        for (size_t i = 0; i < input.rows; i++)
            if (input[i][0] > max_val)
                max_val = input[i][0];
        float sum_exp = 0.0f;
        for (size_t i = 0; i < input.rows; i++){
            output[i][0] = exp(input[i][0] - max_val);
            sum_exp += output[i][0];
        }
        for (size_t i = 0; i < input.rows; i++)
            output[i][0] /= sum_exp;
        last_output = output;
        return output;
    }
    
    Matrix backward(const Matrix& output_gradient, float learning_rate) {
        Matrix input_gradient(input_shape, 1);
        float sum = 0.0f;
        for (size_t i = 0; i < input_shape; i++)
            sum += last_output[i][0] * output_gradient[i][0];
        for (size_t i = 0; i < input_shape; i++)
            input_gradient[i][0] = last_output[i][0] * (output_gradient[i][0] - sum);
        return input_gradient;
    }
};

// =====================================================
// ================ LOSS FUNCTION ======================
// =====================================================
class CategoricalCrossEntropyLoss {
public:
    CategoricalCrossEntropyLoss() {}

    float forward(const Matrix& predicted, const Matrix& target) {
        if (predicted.rows != target.rows || predicted.cols != target.cols)
            throw invalid_argument("Predicted and target matrices must have the same dimensions.");
        float loss = 0.0f;
        for (size_t i = 0; i < predicted.rows; i++) {
            float p = max(min(predicted[i][0], 1.0f - 1e-15f), 1e-15f);
            loss -= target[i][0] * log(p);
        }
        return loss;
    }
    
    Matrix backward(const Matrix& predicted, const Matrix& target) {
        if (predicted.rows != target.rows || predicted.cols != target.cols)
            throw invalid_argument("Predicted and target matrices must have the same dimensions.");
        Matrix grad(predicted.rows, predicted.cols);
        for (size_t i = 0; i < predicted.rows; i++) {
            float p = max(min(predicted[i][0], 1.0f - 1e-15f), 1e-15f);
            grad[i][0] = - target[i][0] / p;
        }
        return grad;
    }
};

// =====================================================
// ================ CONVOLUTION LAYER ==================
// =====================================================
class Conv2D {
private:
    int in_channels, out_channels, kernel_size, stride, padding;
    vector<Matrix> kernels;
    vector<float> biases;
    Matrix last_input;
    Matrix last_input_padded;
    
    Matrix pad_matrix(const Matrix& input, int pad) const {
        Matrix padded(input.rows + 2 * pad, input.cols + 2 * pad);
        for (size_t i = 0; i < input.rows; i++)
            for (size_t j = 0; j < input.cols; j++)
                padded[i + pad][j + pad] = input[i][j];
        return padded;
    }
    
public:
    Conv2D(int in_ch, int out_ch, int k_size, int stride = 1, int pad = 0)
        : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), stride(stride), padding(pad) {
        float stddev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
        default_random_engine generator;
        normal_distribution<float> dist(0.0f, stddev);
        for (int i = 0; i < out_channels; i++) {
            Matrix kernel(kernel_size, kernel_size);
            for (int h = 0; h < kernel_size; h++)
                for (int w = 0; w < kernel_size; w++)
                    kernel[h][w] = dist(generator);
            kernels.push_back(kernel);
        }
        biases = vector<float>(out_channels, 0.1f);
    }
    
    vector<Matrix> forward(const Matrix& input) {
        last_input = input;
        last_input_padded = pad_matrix(input, padding);
        int out_h = (input.rows - kernel_size + 2 * padding) / stride + 1;
        int out_w = (input.cols - kernel_size + 2 * padding) / stride + 1;
        vector<Matrix> output;
        for (int k = 0; k < out_channels; k++) {
            Matrix featureMap(out_h, out_w);
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    float sum = 0.0f;
                    int h_start = i * stride;
                    int w_start = j * stride;
                    for (int x = 0; x < kernel_size; x++)
                        for (int y = 0; y < kernel_size; y++)
                            sum += last_input_padded[h_start + x][w_start + y] * kernels[k][x][y];
                    featureMap[i][j] = sum + biases[k];
                }
            }
            output.push_back(featureMap);
        }
        return output;
    }
    
    // Backward pass for Conv2D.
    // grad_output: gradients for each output channel (same shape as forward output).
    // Returns: gradient with respect to the input.
    vector<Matrix> backward(const vector<Matrix>& grad_output, float learning_rate) {
        int out_h = grad_output[0].rows;
        int out_w = grad_output[0].cols;
        vector<Matrix> kernel_gradients;
        for (int k = 0; k < out_channels; k++)
            kernel_gradients.push_back(Matrix(kernel_size, kernel_size));
        vector<float> bias_gradients(out_channels, 0.0f);
        
        Matrix d_input_padded(last_input_padded.rows, last_input_padded.cols);
        
        for (int k = 0; k < out_channels; k++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    float grad_val = grad_output[k][i][j];
                    bias_gradients[k] += grad_val;
                    int h_start = i * stride;
                    int w_start = j * stride;
                    for (int x = 0; x < kernel_size; x++) {
                        for (int y = 0; y < kernel_size; y++) {
                            kernel_gradients[k][x][y] += last_input_padded[h_start + x][w_start + y] * grad_val;
                            d_input_padded[h_start + x][w_start + y] += kernels[k][x][y] * grad_val;
                        }
                    }
                }
            }
        }
        
        for (int k = 0; k < out_channels; k++) {
            for (size_t i = 0; i < kernels[k].rows; i++)
                for (size_t j = 0; j < kernels[k].cols; j++)
                    kernels[k][i][j] -= learning_rate * kernel_gradients[k][i][j];
            biases[k] -= learning_rate * bias_gradients[k];
        }
        
        int pad = padding;
        Matrix d_input(last_input.rows, last_input.cols);
        for (size_t i = 0; i < d_input.rows; i++)
            for (size_t j = 0; j < d_input.cols; j++)
                d_input[i][j] = d_input_padded[i + pad][j + pad];
        
        return { d_input };
    }
};

// =====================================================
// ================ POOLING LAYER ======================
// =====================================================
// For max pooling, we store the indices of the maximum value.
class PoolingLayer {
private:
    int poolSize, stride;
    string poolingType; // "max" or "average"
    // For max pooling, store indices for each feature map.
    vector<vector<vector<pair<int, int>>>> maxIndices;
public:
    PoolingLayer(int poolSize, int stride, const string& poolingType = "max")
        : poolSize(poolSize), stride(stride), poolingType(poolingType) {}
    
    vector<Matrix> forward(const vector<Matrix>& input_feature_maps) {
        maxIndices.clear();
        vector<Matrix> pooled_maps;
        for (const auto& fmap : input_feature_maps) {
            int inRows = fmap.rows, inCols = fmap.cols;
            int outRows = ((inRows - poolSize) / stride) + 1;
            int outCols = ((inCols - poolSize) / stride) + 1;
            Matrix pooled(outRows, outCols);
            vector<vector<pair<int, int>>> fmapMaxIndices(outRows, vector<pair<int, int>>(outCols, {0,0}));
            for (int i = 0; i < outRows; i++) {
                for (int j = 0; j < outCols; j++) {
                    float poolResult = (poolingType == "max") ? -numeric_limits<float>::infinity() : 0.0f;
                    pair<int, int> maxIndex = {i * stride, j * stride};
                    for (int m = 0; m < poolSize; m++) {
                        for (int n = 0; n < poolSize; n++) {
                            int row = i * stride + m;
                            int col = j * stride + n;
                            if (row < inRows && col < inCols) {
                                if (poolingType == "max") {
                                    if (fmap.data[row][col] > poolResult) {
                                        poolResult = fmap.data[row][col];
                                        maxIndex = {row, col};
                                    }
                                } else if (poolingType == "average") {
                                    poolResult += fmap.data[row][col];
                                }
                            }
                        }
                    }
                    if (poolingType == "average")
                        poolResult /= (poolSize * poolSize);
                    pooled[i][j] = poolResult;
                    fmapMaxIndices[i][j] = maxIndex;
                }
            }
            pooled_maps.push_back(pooled);
            maxIndices.push_back(fmapMaxIndices);
        }
        return pooled_maps;
    }
    
    // Backward pass for max pooling.
    vector<Matrix> backward(const vector<Matrix>& grad_pooled, const vector<Matrix>& input_feature_maps) {
        vector<Matrix> grad_input;
        for (size_t f = 0; f < input_feature_maps.size(); f++) {
            Matrix grad(input_feature_maps[f].rows, input_feature_maps[f].cols);
            int outRows = grad_pooled[f].rows;
            int outCols = grad_pooled[f].cols;
            for (int i = 0; i < outRows; i++) {
                for (int j = 0; j < outCols; j++) {
                    pair<int, int> maxIndex = maxIndices[f][i][j];
                    grad[maxIndex.first][maxIndex.second] += grad_pooled[f][i][j];
                }
            }
            grad_input.push_back(grad);
        }
        return grad_input;
    }
};

// =====================================================
// ================ MNIST DATA LOADING =================
// =====================================================
uint32_t reverseInt(uint32_t i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) | ((uint32_t)c2 << 16) | ((uint32_t)c3 << 8) | c4;
}

vector<Matrix> loadMNISTImages(const string& filename, int numImages) {
    ifstream file(filename, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open image file: " + filename);
    
    uint32_t magic, num, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);
    magic = reverseInt(magic);
    num = reverseInt(num);
    rows = reverseInt(rows);
    cols = reverseInt(cols);
    
    vector<Matrix> images;
    for (int i = 0; i < numImages; i++) {
        Matrix img(rows, cols);
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                img[r][c] = pixel / 255.0f;
            }
        images.push_back(img);
    }
    return images;
}

vector<int> loadMNISTLabels(const string& filename, int numLabels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open())
        throw runtime_error("Could not open label file: " + filename);
    
    uint32_t magic, num;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num), 4);
    magic = reverseInt(magic);
    num = reverseInt(num);
    
    vector<int> labels;
    for (int i = 0; i < numLabels; i++) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels.push_back(static_cast<int>(label));
    }
    return labels;
}

// =====================================================
// =============== UTILITY FUNCTIONS ===================
// =====================================================
Matrix oneHotEncode(int label, int numClasses = 10) {
    Matrix encoded(numClasses, 1);
    encoded[label][0] = 1.0f;
    return encoded;
}

Matrix flattenMatrix(const Matrix& m) {
    Matrix flat(m.rows * m.cols, 1);
    int index = 0;
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            flat[index++][0] = m.data[i][j];
    return flat;
}

Matrix flattenFeatureMaps(const vector<Matrix>& maps) {
    int totalFeatures = 0;
    for (const auto& map : maps)
        totalFeatures += map.rows * map.cols;
    Matrix flat(totalFeatures, 1);
    int index = 0;
    for (const auto& map : maps)
        for (size_t i = 0; i < map.rows; i++)
            for (size_t j = 0; j < map.cols; j++)
                flat[index++][0] = map.data[i][j];
    return flat;
}

// =====================================================
// ============= TRAINING & EVALUATION =================
// =====================================================
struct ModelResults {
    float cnnAccuracy;
    float nonCnnAccuracy;
    float cnnLoss;
    float nonCnnLoss;
};

void trainEpoch(const vector<Matrix>& trainImages, const vector<int>& trainLabels,
                int numClasses, float learning_rate,
                Conv2D& convLayer, PoolingLayer& poolLayer,
                FullyConnected& fcCNN, Softmax& softmaxCNN,
                FullyConnected& fcNonCNN, Softmax& softmaxNonCNN,
                CategoricalCrossEntropyLoss& lossFunc) {
    for (size_t idx = 0; idx < trainImages.size(); idx++) {
        // ----- CNN Pipeline -----
        vector<Matrix> convOut = convLayer.forward(trainImages[idx]);
        vector<Matrix> pooledOut = poolLayer.forward(convOut);
        Matrix flatCNN = flattenFeatureMaps(pooledOut);
        Matrix fcOutCNN = fcCNN.forward(flatCNN);
        Matrix predCNN = softmaxCNN.forward(fcOutCNN);
        
        Matrix targetCNN = oneHotEncode(trainLabels[idx], numClasses);
        float lossCNN = lossFunc.forward(predCNN, targetCNN);
        
        // Backward pass for CNN:
        Matrix gradLossCNN = lossFunc.backward(predCNN, targetCNN);
        Matrix gradSoftmaxCNN = softmaxCNN.backward(gradLossCNN, learning_rate);
        Matrix gradFlatCNN = fcCNN.backward(gradSoftmaxCNN, learning_rate);
        
        // Reshape the gradient to match pooled outputs.
        // Here we assume each pooled map has same size.
        int pooledFeatureCount = pooledOut[0].rows * pooledOut[0].cols;
        vector<Matrix> gradPooled;
        int idx_flat = 0;
        for (size_t k = 0; k < pooledOut.size(); k++) {
            Matrix gradMap(pooledOut[k].rows, pooledOut[k].cols);
            for (size_t i = 0; i < pooledOut[k].rows; i++)
                for (size_t j = 0; j < pooledOut[k].cols; j++)
                    gradMap[i][j] = gradFlatCNN[idx_flat++][0];
            gradPooled.push_back(gradMap);
        }
        
        // Backprop through pooling layer.
        vector<Matrix> gradConv = poolLayer.backward(gradPooled, convOut);
        // Backprop through convolution layer.
        convLayer.backward(gradConv, learning_rate);
        
        // ----- Non-CNN Pipeline -----
        Matrix flatNonCNN = flattenMatrix(trainImages[idx]);
        Matrix fcOutNonCNN = fcNonCNN.forward(flatNonCNN);
        Matrix predNonCNN = softmaxNonCNN.forward(fcOutNonCNN);
        
        Matrix targetNonCNN = oneHotEncode(trainLabels[idx], numClasses);
        float lossNonCNN = lossFunc.forward(predNonCNN, targetNonCNN);
        
        Matrix gradLossNonCNN = lossFunc.backward(predNonCNN, targetNonCNN);
        Matrix gradSoftmaxNonCNN = softmaxNonCNN.backward(gradLossNonCNN, learning_rate);
        fcNonCNN.backward(gradSoftmaxNonCNN, learning_rate);
        
        // Optionally, print per-sample loss if needed.
    }
}

ModelResults evaluateModel(const vector<Matrix>& testImages, const vector<int>& testLabels,
                           int numClasses,
                           Conv2D& convLayer, PoolingLayer& poolLayer,
                           FullyConnected& fcCNN, Softmax& softmaxCNN,
                           FullyConnected& fcNonCNN, Softmax& softmaxNonCNN,
                           CategoricalCrossEntropyLoss& lossFunc) {
    int correctCNN = 0, correctNonCNN = 0;
    float totalLossCNN = 0.0f, totalLossNonCNN = 0.0f;
    for (size_t idx = 0; idx < testImages.size(); idx++) {
        // ----- CNN Pipeline -----
        vector<Matrix> convOut = convLayer.forward(testImages[idx]);
        vector<Matrix> pooledOut = poolLayer.forward(convOut);
        Matrix flatCNN = flattenFeatureMaps(pooledOut);
        Matrix fcOutCNN = fcCNN.forward(flatCNN);
        Matrix predCNN = softmaxCNN.forward(fcOutCNN);
        Matrix targetCNN = oneHotEncode(testLabels[idx], numClasses);
        totalLossCNN += lossFunc.forward(predCNN, targetCNN);
        int predictedLabelCNN = 0;
        float maxValCNN = predCNN[0][0];
        for (size_t i = 1; i < predCNN.rows; i++) {
            if (predCNN[i][0] > maxValCNN) {
                maxValCNN = predCNN[i][0];
                predictedLabelCNN = i;
            }
        }
        if (predictedLabelCNN == testLabels[idx])
            correctCNN++;
        
        // ----- Non-CNN Pipeline -----
        Matrix flatNonCNN = flattenMatrix(testImages[idx]);
        Matrix fcOutNonCNN = fcNonCNN.forward(flatNonCNN);
        Matrix predNonCNN = softmaxNonCNN.forward(fcOutNonCNN);
        Matrix targetNonCNN = oneHotEncode(testLabels[idx], numClasses);
        totalLossNonCNN += lossFunc.forward(predNonCNN, targetNonCNN);
        int predictedLabelNonCNN = 0;
        float maxValNonCNN = predNonCNN[0][0];
        for (size_t i = 1; i < predNonCNN.rows; i++) {
            if (predNonCNN[i][0] > maxValNonCNN) {
                maxValNonCNN = predNonCNN[i][0];
                predictedLabelNonCNN = i;
            }
        }
        if (predictedLabelNonCNN == testLabels[idx])
            correctNonCNN++;
    }
    
    int total = testImages.size();
    ModelResults results;
    results.cnnAccuracy = (float)correctCNN / total * 100.0f;
    results.nonCnnAccuracy = (float)correctNonCNN / total * 100.0f;
    results.cnnLoss = totalLossCNN / total;
    results.nonCnnLoss = totalLossNonCNN / total;
    return results;
}

// =====================================================
// ==================== MAIN FUNCTION ==================
// =====================================================
int main() {
    // -------------------- PARAMETERS --------------------
    const int imageHeight = 28, imageWidth = 28;
    const int numClasses = 10;
    
    // Use a larger subset.
    const int numTrain = 30000;  // 10,000 training images
    const int numTest  = 5000;   // 2,000 test images
    
    // File names (assumed to be unzipped in the working directory)
    string trainImagesFile = "train-images-idx3-ubyte";
    string trainLabelsFile = "train-labels-idx1-ubyte";
    string testImagesFile  = "t10k-images-idx3-ubyte";
    string testLabelsFile  = "t10k-labels-idx1-ubyte";
    
    cout << "Loading MNIST training data..." << endl;
    vector<Matrix> trainImages = loadMNISTImages(trainImagesFile, numTrain);
    vector<int> trainLabels = loadMNISTLabels(trainLabelsFile, numTrain);
    cout << "Loading MNIST test data..." << endl;
    vector<Matrix> testImages = loadMNISTImages(testImagesFile, numTest);
    vector<int> testLabels = loadMNISTLabels(testLabelsFile, numTest);
    
    // ---------------- MODEL INITIALIZATION ----------------
    // CNN pipeline: Conv2D -> Pooling -> Flatten -> FullyConnected -> Softmax
    // Using one convolutional layer with 8 kernels of size 3x3 and padding 1.
    Conv2D convLayer(1, 8, 3, 1, 1);
    PoolingLayer poolLayer(2, 2, "max"); // 2x2 max pooling
    // After pooling, each feature map becomes 14x14. Total features = 8*14*14.
    FullyConnected fcCNN(8 * 14 * 14, numClasses);
    Softmax softmaxCNN(numClasses);
    
    // Non-CNN pipeline: Flatten image directly -> FullyConnected -> Softmax
    FullyConnected fcNonCNN(imageHeight * imageWidth, numClasses);
    Softmax softmaxNonCNN(numClasses);
    
    // Loss function
    CategoricalCrossEntropyLoss lossFunc;
    
    // ---------------- TRAINING LOOP ----------------
    const int epochs = 20;  // Increase epochs for better accuracy
    const float learning_rate = 0.01f;
    
    cout << "Starting training for " << epochs << " epochs..." << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
        trainEpoch(trainImages, trainLabels, numClasses, learning_rate,
                   convLayer, poolLayer, fcCNN, softmaxCNN,
                   fcNonCNN, softmaxNonCNN, lossFunc);
        
        ModelResults results = evaluateModel(testImages, testLabels, numClasses,
                                             convLayer, poolLayer, fcCNN, softmaxCNN,
                                             fcNonCNN, softmaxNonCNN, lossFunc);
        cout << "After Epoch " << epoch + 1 << ":\n"
             << "  CNN -> Loss: " << results.cnnLoss << ", Accuracy: " << results.cnnAccuracy << "%\n"
             << "  Non-CNN -> Loss: " << results.nonCnnLoss << ", Accuracy: " << results.nonCnnAccuracy << "%" << endl;
    }
    
    // ---------------- FINAL EVALUATION ----------------
    cout << "Final evaluation on test set:" << endl;
    ModelResults finalResults = evaluateModel(testImages, testLabels, numClasses,
                                              convLayer, poolLayer, fcCNN, softmaxCNN,
                                              fcNonCNN, softmaxNonCNN, lossFunc);
    cout << "CNN -> Loss: " << finalResults.cnnLoss << ", Accuracy: " << finalResults.cnnAccuracy << "%" << endl;
    cout << "Non-CNN -> Loss: " << finalResults.nonCnnLoss << ", Accuracy: " << finalResults.nonCnnAccuracy << "%" << endl;
    
    // ---------------- DIGIT RECOGNITION DEMO ----------------
    cout << "\nDigit Recognition (first 20 test images):" << endl;
    for (int i = 0; i < 20; i++) {
        vector<Matrix> convOut = convLayer.forward(testImages[i]);
        vector<Matrix> pooledOut = poolLayer.forward(convOut);
        Matrix flatCNN = flattenFeatureMaps(pooledOut);
        Matrix fcOutCNN = fcCNN.forward(flatCNN);
        Matrix predCNN = softmaxCNN.forward(fcOutCNN);
        int predictedLabel = 0;
        float maxVal = predCNN[0][0];
        for (size_t j = 1; j < predCNN.rows; j++) {
            if (predCNN[j][0] > maxVal) {
                maxVal = predCNN[j][0];
                predictedLabel = j;
            }
        }
        cout << "Test Image " << i+1 << ": Predicted = " << predictedLabel 
             << ", True Label = " << testLabels[i] << endl;
    }
    
    return 0;
}


