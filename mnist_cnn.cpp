/*TODO
>address other TODOs
>clean up evaluate loop
>some way to save weights and load??
>check the below imports
>use underscores instead of camel case everywhere
>cleanup utility functions, check where each needed
>Check activation functions being applied over CNN layers nodes.
*/
#include <iostream>
#include <fstream>
#include <vector> //use
#include <stdexcept>
#include <cmath>
#include <limits>
#include <random>
#include <cstdint>
#include <string>
#include <utility> //unique ptr

using namespace std;

/*
Matrix class to handle some common operations
*/
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

/*
NeuralLayer interface
ALl other layers inherit from this
This allows us to define the NN and the forward and backward passes without knowing specific implementation of any.
*/
class NeuralLayer {
    public:
        virtual vector<Matrix> forward(const vector<Matrix>& input)=0;
        virtual vector<Matrix> backward(const vector<Matrix>& output_gradient, float learning_rate)=0;
        void save_weights(ofstream& file){};
        void load_weights(ifstream& file){};
};



/*
Fully connected layer
Input : Vector<Matrix> with dimension (1,input_shape,1)
Output : Vector<Matrix> with dimension (1,output_shape,1)
*/
class FullyConnected : public NeuralLayer {
public:
    int input_shape, output_shape;
    Matrix weight;
    Matrix bias;
    Matrix last_input;
    
    FullyConnected(int input_shape, int output_shape)
        : input_shape(input_shape), output_shape(output_shape),
          weight(output_shape, input_shape), bias(output_shape, 1) {

        //Random initialize weight and bias matrix
        default_random_engine generator;
        uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (size_t i = 0; i < weight.rows; i++)
            for (size_t j = 0; j < weight.cols; j++)
                weight[i][j] = dist(generator);
        for (size_t i = 0; i < bias.rows; i++)
            bias[i][0] = dist(generator);
    }
    
    vector<Matrix> forward(const vector<Matrix>& input) {
        Matrix input_matrix = input[0];
        if (input_matrix.rows != (size_t)input_shape)
            throw invalid_argument("Input does not match shape in FullyConnected");
        Matrix output = weight * input_matrix;
        output = output + bias;
        last_input = input_matrix;
        return vector<Matrix>{1,output};
    }
    
    vector<Matrix> backward(const vector<Matrix>& output_gradient, float learning_rate) {
        Matrix output_gradient_matrix = output_gradient[0];
        if (output_gradient_matrix.rows != (size_t)output_shape)
            throw invalid_argument("Output gradient does not match shape in FullyConnected");
        Matrix input_gradient = weight.T() * output_gradient_matrix;
        Matrix weight_gradient = output_gradient_matrix * last_input.T();
        Matrix bias_gradient = output_gradient_matrix;
        weight = weight - weight_gradient * learning_rate;
        bias = bias - bias_gradient * learning_rate;
        return vector<Matrix>(1,input_gradient);
    }

    void save_weights(ofstream& file) {
        if (!file.is_open()) {
            cerr << "Error: File not open for saving weights." << endl;
            return;
        }

        // Save weight matrix dimensions and data
        file.write(reinterpret_cast<char*>(&weight.rows), sizeof(weight.rows));
        file.write(reinterpret_cast<char*>(&weight.cols), sizeof(weight.cols));
        for (size_t i = 0; i < weight.rows; i++) {
            for (size_t j = 0; j < weight.cols; j++) {
                file.write(reinterpret_cast<char*>(&weight[i][j]), sizeof(weight[i][j]));
            }
        }

        // Save bias matrix dimensions and data
        file.write(reinterpret_cast<char*>(&bias.rows), sizeof(bias.rows));
        file.write(reinterpret_cast<char*>(&bias.cols), sizeof(bias.cols));
        for (size_t i = 0; i < bias.rows; i++) {
                file.write(reinterpret_cast<char*>(&bias[i][0]), sizeof(bias[i][0]));
        }
        file << endl;
    }

    void load_weights(ifstream& file) {
        if (!file.is_open()) {
            cerr << "Error: File not open for loading weights." << endl;
            return;
        }

        // Load weights
        size_t weight_rows, weight_cols;
        file.read(reinterpret_cast<char*>(&weight_rows), sizeof(weight_rows));
        file.read(reinterpret_cast<char*>(&weight_cols), sizeof(weight_cols));
        weight = Matrix(weight_rows, weight_cols);
        for (size_t i = 0; i < weight_rows; i++) {
            for (size_t j = 0; j < weight_cols; j++) {
                file.read(reinterpret_cast<char*>(&weight[i][j]), sizeof(weight[i][j]));
            }
        }

        // Load bias
        size_t bias_rows, bias_cols;
        file.read(reinterpret_cast<char*>(&bias_rows), sizeof(bias_rows));
        file.read(reinterpret_cast<char*>(&bias_cols), sizeof(bias_cols));

        if(bias_cols!=1) {
            cerr << "Error: Bias columns is not 1." <<endl;
            return;
        }

        bias = Matrix(bias_rows, bias_cols);
        for (size_t i = 0; i < bias_rows; i++) {
                file.read(reinterpret_cast<char*>(&bias[i][0]), sizeof(bias[i][0]));
        }
        file.ignore(numeric_limits<streamsize>::max(), '\n');
    }

};



/*
ReLU_Leaky - Leaky ReLU Acitvation for hidden layers
*/
class ReLU_Leaky : public NeuralLayer {
private:
    float coeff = 0.01f;
    float coeffp = 1.0f / coeff;
public:
    int rows, cols, depth;
    vector<Matrix> last_input;
    
    ReLU_Leaky(int rows, int cols, int depth) : rows(rows), cols(cols), depth(depth) {}
    
    vector<Matrix> forward(const vector<Matrix>& input) {
        // Matrix input_matrix = input[0];
        vector<Matrix> output(depth, Matrix(rows, cols));
        for (size_t n = 0; n < depth; n++)
            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < cols; j++)
                    output[n][i][j] = (input[n][i][j] > 0) ? input[n][i][j] : coeff * input[n][i][j];
        last_input = input;
        return vector<Matrix>(output);
    }
    
    vector<Matrix> backward(const vector<Matrix>& output_gradient, float learning_rate) {
        vector<Matrix> input_gradient(depth, Matrix(rows, cols));
        for (size_t n = 0; n < depth; n++)
            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < cols; j++)
                    input_gradient[n][i][j] = (last_input[n][i][j] > 0) ? output_gradient[n][i][j] : coeffp * output_gradient[n][i][j];
        return input_gradient;
    }
};

/*
Softmax - Acitvation for Catergorization task
*/
class Softmax {
    public:
        int input_shape, output_shape;
        vector<Matrix> last_output;
        
        Softmax(int input_shape) : input_shape(input_shape), output_shape(input_shape) {}
        
        vector<Matrix> forward(vector< Matrix>& input) {
            if (input[0].rows != (size_t)input_shape || input[0].cols != 1)
                throw invalid_argument("Input must be column vector in Softmax");
            vector<Matrix> output(1, Matrix(input[0].rows, 1));
            float max_val = input[0][0][0];
            for (size_t i = 0; i < input[0].rows; i++)
                if (input[0][i][0] > max_val)
                    max_val = input[0][i][0];
            float sum_exp = 0.0f;
            for (size_t i = 0; i < input[0].rows; i++){
                output[0][i][0] = exp(input[0][i][0] - max_val);
                sum_exp += output[0][i][0];
            }
            for (size_t i = 0; i < input[0].rows; i++)
                output[0][i][0] /= sum_exp;
            last_output = output;
            return output;
        }
        
        vector<Matrix> backward(vector< Matrix>& output_gradient, float learning_rate) {
            vector<Matrix> input_gradient(1, Matrix(input_shape, 1));
            float sum = 0.0f;
            for (size_t i = 0; i < input_shape; i++)
                sum += last_output[0][i][0] * output_gradient[0][i][0];
            for (size_t i = 0; i < input_shape; i++)
                input_gradient[0][i][0] = last_output[0][i][0] * (output_gradient[0][i][0] - sum);
            return input_gradient;
        }
};

/*
<Insert Here>
*/
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

/*
Converts input feature maps from one set of channels to another 
using 2D convolution with specified kernels, stride, and padding.
*/
class Conv2D {
    private:
        int in_channels, out_channels;
        int kernel_size, stride, padding;
        vector<vector<Matrix>> kernels; // [out_ch][in_ch][kernel]
        vector<float> biases;
        vector<Matrix> last_input;
        vector<Matrix> last_input_padded;
    
        vector<Matrix> pad_input(const vector<Matrix>& input) const {
            vector<Matrix> padded;
            for(const auto& channel : input) {
                Matrix p_channel(channel.rows + 2*padding, 
                                channel.cols + 2*padding);
                for(size_t i=0; i<channel.rows; ++i) {
                    for(size_t j=0; j<channel.cols; ++j) {
                        p_channel[i+padding][j+padding] = channel[i][j];
                    }
                }
                padded.push_back(p_channel);
            }
            return padded;
        }
    
    public:
        Conv2D(int in_ch, int out_ch, int k_size, 
              int stride=1, int pad=0) :
            in_channels(in_ch), out_channels(out_ch),
            kernel_size(k_size), stride(stride), padding(pad) {
            
            // He initialization with fan-in = in_channels * kernel_size^2
            float stddev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
            default_random_engine generator;
            normal_distribution<float> dist(0.0f, stddev);
    
            // Initialize kernels: out_channels x in_channels x kernel_size x kernel_size
            kernels.resize(out_channels);
            for(int o=0; o<out_channels; ++o) {
                kernels[o].resize(in_channels);
                for(int i=0; i<in_channels; ++i) {
                    kernels[o][i] = Matrix(kernel_size, kernel_size);
                    for(int x=0; x<kernel_size; ++x) {
                        for(int y=0; y<kernel_size; ++y) {
                            kernels[o][i][x][y] = dist(generator);
                        }
                    }
                }
            }
            
            // Initialize biases
            biases.resize(out_channels, 0.1f);
        }
    
        vector<Matrix> forward(const vector<Matrix>& input) {
            if(input.size() != (size_t)in_channels)
                throw invalid_argument("Input channel count mismatch");
                
            last_input = input;
            last_input_padded = pad_input(input);
    
            const int out_h = (input[0].rows - kernel_size + 2*padding)/stride + 1;
            const int out_w = (input[0].cols - kernel_size + 2*padding)/stride + 1;
            
            vector<Matrix> output(out_channels, Matrix(out_h, out_w));
    
            // For each output channel
            for(int o=0; o<out_channels; ++o) {
                // For each input channel
                for(int i=0; i<in_channels; ++i) {
                    // Perform cross-correlation
                    for(int h=0; h<out_h; ++h) {
                        for(int w=0; w<out_w; ++w) {
                            const int h_start = h*stride;
                            const int w_start = w*stride;
                            
                            float sum = 0.0f;
                            for(int x=0; x<kernel_size; ++x) {
                                for(int y=0; y<kernel_size; ++y) {
                                    sum += last_input_padded[i][h_start+x][w_start+y] * 
                                          kernels[o][i][x][y];
                                }
                            }
                            output[o][h][w] += sum;
                        }
                    }
                }
                // Add bias
                for(int h=0; h<out_h; ++h) {
                    for(int w=0; w<out_w; ++w) {
                        output[o][h][w] += biases[o];
                    }
                }
            }
            return output;
        }
    
        vector<Matrix> backward(const vector<Matrix>& grad_output, float lr) {
            vector<Matrix> input_grad(in_channels, 
                Matrix(last_input[0].rows, last_input[0].cols));
                
            vector<vector<Matrix>> kernel_grads(out_channels,
                vector<Matrix>(in_channels, Matrix(kernel_size, kernel_size)));
                
            vector<float> bias_grads(out_channels, 0.0f);
    
            // For each output channel
            for(int o=0; o<out_channels; ++o) {
                // For each input channel
                for(int i=0; i<in_channels; ++i) {
                    // Calculate kernel gradients
                    for(int h=0; h<grad_output[o].rows; ++h) {
                        for(int w=0; w<grad_output[o].cols; ++w) {
                            const float grad = grad_output[o][h][w];
                            const int h_start = h*stride;
                            const int w_start = w*stride;
                            
                            for(int x=0; x<kernel_size; ++x) {
                                for(int y=0; y<kernel_size; ++y) {
                                    // Update kernel gradient
                                    kernel_grads[o][i][x][y] += 
                                        last_input_padded[i][h_start+x][w_start+y] * grad;
                                    
                                    // Update input gradient
                                    input_grad[i][h_start+x-padding][w_start+y-padding] += 
                                        kernels[o][i][x][y] * grad;
                                }
                            }
                        }
                    }
                }
                // Calculate bias gradients
                bias_grads[o] = accumulate(
                    grad_output[o].data.begin(), grad_output[o].data.end(), 0.0f,
                    [](float sum, const vector<float>& row) {
                        return sum + accumulate(row.begin(), row.end(), 0.0f);
                    });
            }
    
            // Update parameters
            for(int o=0; o<out_channels; ++o) {
                for(int i=0; i<in_channels; ++i) {
                    for(int x=0; x<kernel_size; ++x) {
                        for(int y=0; y<kernel_size; ++y) {
                            kernels[o][i][x][y] -= lr * kernel_grads[o][i][x][y];
                        }
                    }
                }
                biases[o] -= lr * bias_grads[o];
            }
    
            return input_grad;
        }
    };

/*
<Insert Here>
*/
// For max pooling, we store the indices of the maximum value.
class PoolingLayer : NeuralLayer{
private:
    int poolSize, stride;
    string poolingType; // "max" or "average"
    // For max pooling, store indices for each feature map.
    vector<vector<vector<pair<int, int>>>> maxIndices;
    vector<Matrix> input_feature_maps;
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
        this->input_feature_maps = input_feature_maps;
        return pooled_maps;
    }
    
    // Backward pass for max pooling.
    vector<Matrix> backward(const vector<Matrix>& grad_pooled, float learning_rate) {//Learning rate unused, just to match interface
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

/*
Convert the data from Big Endian.
*/
uint32_t reverseInt(uint32_t i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8) & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) | ((uint32_t)c2 << 16) | ((uint32_t)c3 << 8) | c4;
}

/*
MNIST uses the idx3 format, where magic encodes the dimensions and data type, followed bu the dimensions of the data, then the data itself
*/
vector<Matrix> loadMNISTImages(const string& filename, int numImages) { // TODO comment this function
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

/*
<Insert Here>
*/
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

/*
<Insert Here>
*/
Matrix oneHotEncode(int label, int numClasses = 10) {
    Matrix encoded(numClasses, 1);
    encoded[label][0] = 1.0f;
    return encoded;
}

//TODO is this safe to remove?
// Matrix flattenMatrix(const Matrix& m) {
//     Matrix flat(m.rows * m.cols, 1);
//     int index = 0;
//     for (size_t i = 0; i < m.rows; i++)
//         for (size_t j = 0; j < m.cols; j++)
//             flat[index++][0] = m.data[i][j];
//     return flat;
// }


/*
<Insert Here>
*/
class FlattenFeatureMaps  {//TODO make sure working
    private:
        vector<pair<size_t, size_t>> original_shapes;
    
    public:
        FlattenFeatureMaps() = default;
    
        // Forward pass: Convert vector<Matrix> to column vector
        vector<Matrix> forward(const vector<Matrix>& maps) {
            original_shapes.clear();
            int totalFeatures = 0;
            
            // Store original shapes and calculate total features
            for (const auto& map : maps) {
                original_shapes.emplace_back(map.rows, map.cols);
                totalFeatures += map.rows * map.cols;
            }
    
            Matrix flat(totalFeatures, 1);
            int index = 0;
            
            // Flatten all feature maps
            for (const auto& map : maps) {
                for (size_t i = 0; i < map.rows; i++) {
                    for (size_t j = 0; j < map.cols; j++) {
                        flat[index++][0] = map[i][j];
                    }
                }
            }
            return vector<Matrix>(1,flat);
        }
    
        // Backward pass: Convert column vector back to original feature maps
        vector<Matrix> backward(const vector<Matrix>& gradient) {
            vector<Matrix> reconstructed;
            int index = 0;
            Matrix gradient_matrix = gradient[0];
            // Reconstruct each feature map using stored shapes
            for (const auto& shape : original_shapes) {
                Matrix map(shape.first, shape.second);
                
                for (size_t i = 0; i < shape.first; i++) {
                    for (size_t j = 0; j < shape.second; j++) {
                        map[i][j] = gradient_matrix[index++][0];
                    }
                }
                reconstructed.push_back(map);
            }
            return reconstructed;
        }
    
        // For compatibility with existing training interface
        vector<Matrix> backward(const Matrix& gradient, float learning_rate) {
            return backward(vector<Matrix>(1,gradient)); // learning_rate not used
        }
    };

/*
<Insert Here>
*/
struct ModelResults {
    float Accuracy;
    float Loss;
};

// TODO replace this with a func that takes MODEL class as input, and test  images+lables
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
        //predCNN= model.predict()
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

/*
Defines a classification model as a vector of Layer interfaces
This allows us to calculate forward() and backward() looping through the layers.
Ensure num_classes matches last layer output
*/
class ClassificationModel{
    public:
        vector<unique_ptr<NeuralLayer>> layers;
        int num_classes;
        CategoricalCrossEntropyLoss loss_calc  = CategoricalCrossEntropyLoss();

        ClassificationModel(vector<unique_ptr<NeuralLayer>> layers, int num_classes){
            this->layers = layers;
            this->num_classes =  num_classes;
        }

        float train(const vector<Matrix>& trainImages, const vector<int>& trainLabels, float learning_rate){
            for (size_t idx = 0; idx < trainImages.size(); idx++) {
                // FORWARD PASS___
                vector<Matrix> intermediate;
                intermediate = vector<Matrix>(1,trainImages[idx]);
                for(size_t i = 0; i<layers.size(); i++)
                {
                    intermediate  = layers[i]->forward(intermediate);
                }
                Matrix output = intermediate[0];
                
                Matrix target = oneHotEncode(trainLabels[idx], num_classes);
                float loss = loss_calc.forward(output, target); //TODO, why is there loss forward and loss backward both
                
                // BACKWARD PASS___
                intermediate = vector<Matrix>(1,loss_calc.backward(output, target));
                for(size_t i = layers.size(); i>=0; i--)
                {
                    intermediate  = layers[i]->backward(intermediate, learning_rate);
                }

                // Print per sample single loss value??
                return loss;
            }
        }

        vector<float> train_epochs(const vector<Matrix>& trainImages, const vector<int>& trainLabels, float learning_rate, int epochs){
            vector<float> loss_vector;
            for(int i=0; i<epochs; i++){
                loss_vector.push_back(this->train(trainImages, trainLabels, learning_rate));
                cout << "Epoch: " << i << "\tLoss: " << loss_vector.back() << endl;
            }
            return loss_vector;
        }

        vector<float> predict(Matrix input_image){} //check correct return

        
};

int main() {
    const int imageHeight = 28, imageWidth = 28;
    const int numClasses = 10;
    
    // Use a larger subset lataer
    const int numTrain = 10000;
    const int numTest  = 2000;  
    
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
    vector<unique_ptr<NeuralLayer>> CNNLayers;
    CNNLayers.emplace_back(new Conv2D(1, 8, 3, 1, 1));
    CNNLayers.emplace_back(new PoolingLayer(2, 2, "max"));
    CNNLayers.emplace_back(new FullyConnected(8 * 14 * 14, numClasses));
    CNNLayers.emplace_back(new Softmax(numClasses));
    ClassificationModel CNN = ClassificationModel(CNNLayers,numClasses);
    CNN; 



    // Non-CNN pipeline: Flatten image directly -> FullyConnected -> Softmax TODO change this more complex.. Use RELU activation IMPORTANT
    vector<unique_ptr<NeuralLayer>> LinearLayers;
    int hidden_layer_size = 20;
    LinearLayers.emplace_back(new FullyConnected(imageHeight * imageWidth, hidden_layer_size));
    LinearLayers.emplace_back(new ReLU_Leaky(hidden_layer_size));
    LinearLayers.emplace_back(new FullyConnected(hidden_layer_size, numClasses));
    LinearLayers.emplace_back(new Softmax(numClasses));
    ClassificationModel Linear = ClassificationModel(LinearLayers, numClasses);
    // Hyperparams-tune if necessary (dont touch lr)
    const int epochs = 10;  // Increase epochs for better accuracy
    const float learning_rate = 0.001f;
    
    // ---------------- TRAINING LOOP ----------------
    cout << "Starting training for CNN model for " << epochs << " epochs" << endl;
    CNN.train_epochs(trainImages, trainLabels, learning_rate, epochs);

    cout << "Starting training for Linear model for " << epochs << " epochs" << endl;
    Linear.train_epochs(trainImages, trainLabels, learning_rate, epochs);


    // TODO, cleanup  below
    // // ---------------- FINAL EVALUATION ----------------
    // cout << "Final evaluation on test set:" << endl;
    // ModelResults finalResults = evaluateModel(testImages, testLabels, numClasses,
    //                                           convLayer, poolLayer, fcCNN, softmaxCNN,
    //                                           fcNonCNN, softmaxNonCNN, lossFunc);
    // cout << "CNN -> Loss: " << finalResults.cnnLoss << ", Accuracy: " << finalResults.cnnAccuracy << "%" << endl;
    // cout << "Non-CNN -> Loss: " << finalResults.nonCnnLoss << ", Accuracy: " << finalResults.nonCnnAccuracy << "%" << endl;
    
    // // ---------------- DIGIT RECOGNITION DEMO ----------------
    // cout << "\nDigit Recognition Demo (first 10 test images):" << endl;
    // for (int i = 0; i < 10; i++) {
    //     vector<Matrix> convOut = convLayer.forward(testImages[i]);
    //     vector<Matrix> pooledOut = poolLayer.forward(convOut);
    //     Matrix flatCNN = flattenFeatureMaps(pooledOut);
    //     Matrix fcOutCNN = fcCNN.forward(flatCNN);
    //     Matrix predCNN = softmaxCNN.forward(fcOutCNN);
    //     int predictedLabel = 0;
    //     float maxVal = predCNN[0][0];
    //     for (size_t j = 1; j < predCNN.rows; j++) {
    //         if (predCNN[j][0] > maxVal) {
    //             maxVal = predCNN[j][0];
    //             predictedLabel = j;
    //         }
    //     }
    //     cout << "Test Image " << i+1 << ": Predicted = " << predictedLabel 
    //          << ", True Label = " << testLabels[i] << endl;
    // }
    
    return 0;
}


