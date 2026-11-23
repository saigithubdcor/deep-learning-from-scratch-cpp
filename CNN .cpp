#include <vector>
#include <stdexcept>
#include <cmath>
#include<bits/stdc++.h>
#include<random>

class Matrix{
    public:
        size_t cols;
        size_t rows;
        std::vector<std::vector<float>> data;

        Matrix(){
            data = std::vector<std::vector<float>>(0,std::vector<float>(0));
            this->cols = 0;
            this->rows = 0;
        }

        Matrix(size_t rows, size_t cols){
            data = std::vector<std::vector<float>>(rows,std::vector<float>(cols));
            this->cols = cols;
            this->rows = rows;
        }

        std::vector<float> operator [](size_t index){
            return data[index];
        }
        std::vector<float> operator [](size_t index) const{
            return data[index];
        }

        Matrix operator+(const Matrix& other) const{
            Matrix sum(rows, cols);
            unsigned i,j;
            for (i = 0; i < rows; i++)
            {
                for (j = 0; j < cols; j++)
                {
                    sum[i][j] = this->data[i][j] + other[i][j];
                }
            }
            
            return sum;
        }
        
        // Subtraction of Two Matrices
        Matrix operator-(const Matrix& other) const{
            Matrix diff(rows, cols);

            size_t i,j;
            for (i = 0; i < rows; i++)
            {
                for (j = 0; j < cols; j++)
                {
                    diff[i][j] = this->data[i][j] - other[i][j];
                }
            }
            
            return diff;
        }

        Matrix operator*(const Matrix& other) const{
            if(this->cols!=other.rows)
                throw std::invalid_argument( "Matrices do not match shape" );
            
            int rows = this->rows;
            int cols = other.cols;
            int n = this->cols;
            Matrix prod(rows, cols);

            size_t i,j,k;
            for (i = 0; i < rows; i++)
            {
                for (j = 0; j < cols; j++)
                {
                    float sum = 0;
                    for (k = 0; k < n; k++)
                        sum+=(*this)[i][k]*other[k][j];
                    prod[i][j] = sum;
                }
            }
            return prod;
        }

        Matrix operator*(float other) const{
            Matrix prod(rows, cols);
            size_t i,j;
            for (i = 0; i < rows; i++)
            {
                for (j = 0; j < cols; j++)
                {
                    prod[i][j] = this->data[i][j]*other;
                }
            }
            return prod;
        }

        Matrix T(){
            Matrix transposed(cols, rows);
    
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    transposed[j][i] = (*this)[i][j];
                }
            }
    
            return transposed;
        }
};

class FullyConnected {
        int input_shape;
        int output_shape;
        Matrix weight;
        Matrix bias;
        Matrix last_input;

        FullyConnected(int input_shape, int output_shape)
        {
            this->input_shape = input_shape;
            this->output_shape = output_shape;
            weight = Matrix(output_shape, input_shape);
            bias = Matrix(output_shape,1);
        }

        Matrix forward(Matrix& input){
            if(input.rows!=input_shape)
                throw std::invalid_argument( "Input does not match shape" );
            Matrix output = weight*input;

            output=output+bias;

            last_input = input;
            return output;
        }

        Matrix backward(const Matrix& output_gradient, float learning_rate) {
            if (output_gradient.rows != output_shape) {
                throw std::invalid_argument("Output gradient does not match shape");
            }

            Matrix input_gradient = weight.T() * output_gradient;
            Matrix weight_gradient = output_gradient * last_input.T();
            Matrix bias_gradient = output_gradient;

            weight=weight-weight_gradient*learning_rate;
            bias=bias-bias_gradient*learning_rate;
            return input_gradient;
        }

};

class ReLU_Leaky {
    private:
        float coeff = 0.01f;
        float coeffp = 1.0f/coeff;
    public:
        int input_shape;
        int output_shape;
        Matrix last_input;
    
    ReLU_Leaky (int input_shape){
        this->input_shape = input_shape;
        output_shape = input_shape;
    }

    Matrix forward(const Matrix& input){
        Matrix output(input.rows,1);
        for(size_t i=0; i<input.rows; i++){
            output[i][0] = input[i][0]>0 ? input[i][0] : coeff*input[i][0];
        }
        last_input = input;
        return output;
    }

    Matrix backward(const Matrix& output_gradient, float learning_rate){
        if (output_gradient.rows != output_shape) {
            throw std::invalid_argument("Output gradient does not match shape");
        }
        
        Matrix input_gradient(this->input_shape,1);

        for(size_t i=0; i<input_gradient.rows; i++){
            input_gradient[i][0] = last_input[i][0]>0 ? output_gradient[i][0] : coeffp*output_gradient[i][0];
        }
    }
};


class BinaryCrossEntropyLoss {
public:
    BinaryCrossEntropyLoss() {}

    // Forward pass: Calculate Binary Cross-Entropy Loss
    float forward(const Matrix& predicted, const Matrix& target) {
        if (predicted.rows != target.rows || predicted.cols != target.cols) {
            throw std::invalid_argument("Predicted and target matrices must have the same dimensions.");
        }

        float loss = 0.0f;
        for (size_t i = 0; i < predicted.rows; i++) {
            // Ensure predicted values are within [0, 1] range
            float predicted_prob = std::max(std::min(predicted[i][0], 1.0f), 1e-15f); // Clamp to avoid log(0)

            // Binary Cross-Entropy formula
            loss -= target[i][0] * std::log(predicted_prob) + (1 - target[i][0]) * std::log(1 - predicted_prob);
        }
        return loss / predicted.rows; // Average loss over the batch
    }

    // Backward pass: Compute the gradient of the loss with respect to predicted values
    Matrix backward(const Matrix& predicted, const Matrix& target) {
        if (predicted.rows != target.rows || predicted.cols != target.cols) {
            throw std::invalid_argument("Predicted and target matrices must have the same dimensions.");
        }

        Matrix gradient(predicted.rows, predicted.cols);

        for (size_t i = 0; i < predicted.rows; i++) {
            // Clamp predicted values to avoid division by zero in the gradient calculation
            float predicted_prob = std::max(std::min(predicted[i][0], 1.0f), 1e-15f);

            // Gradient of Binary Cross-Entropy Loss w.r.t predicted probabilities
            gradient[i][0] = (predicted_prob - target[i][0]) / (predicted_prob * (1 - predicted_prob));
        }

        return (gradient*(1/predicted.rows)); // Average gradient over the batch
    }
};

class Softmax {
    public:
        int input_shape;
        int output_shape;
        Matrix last_output;
    
        Softmax(int input_shape) {
            this->input_shape = input_shape;
            output_shape = input_shape;
        }
    
        Matrix forward(const Matrix& input) {
            if (input.rows != input_shape || input.cols != 1) {
                throw std::invalid_argument("Input must be column vector of size input_shape x 1");
            }
    
            Matrix output(input.rows, 1);
            float max_val = input[0][0];
            
            // Find maximum value for numerical stability
            for (size_t i = 0; i < input.rows; ++i) {
                if (input[i][0] > max_val) {
                    max_val = input[i][0];
                }
            }
    
            float sum_exp = 0.0f;
            for (size_t i = 0; i < input.rows; ++i) {
                output[i][0] = exp(input[i][0] - max_val);
                sum_exp += output[i][0];
            }
    
            // Normalize to create probability distribution
            for (size_t i = 0; i < input.rows; ++i) {
                output[i][0] /= sum_exp;
            }
    
            last_output = output;
            return output;
        }
    
        Matrix backward(const Matrix& output_gradient, float learning_rate) {
            Matrix input_gradient(input_shape, 1);
            float sum = 0.0f;
    
            // Compute sum(s_i * dy_i)
            for (size_t i = 0; i < input_shape; ++i) {
                sum += last_output[i][0] * output_gradient[i][0];
            }
    
            // Calculate gradient using softmax derivative property
            for (size_t i = 0; i < input_shape; ++i) {
                input_gradient[i][0] = last_output[i][0] * (output_gradient[i][0] - sum);
            }
    
            return input_gradient;
        }
    };
    
//     // ======== FIRST: CRITICAL MATRIX CLASS FIXES ======== //
// // Fix operator[] to return reference for proper modification
// std::vector<float>& operator[](size_t index) {
//     return data[index];
// }
// const std::vector<float>& operator[](size_t index) const {
//     return data[index];
// }

// // Fix matrix multiplication (corrected from previous version)
// Matrix operator*(const Matrix& other) const {
//     if(cols != other.rows)
//         throw std::invalid_argument("Matrices do not match shape");
    
//     Matrix prod(rows, other.cols);
//     for(size_t i=0; i<rows; ++i) {
//         for(size_t j=0; j<other.cols; ++j) {
//             float sum = 0.0f;
//             for(size_t k=0; k<cols; ++k) {
//                 sum += data[i][k] * other[k][j];  // CORRECTED: Multiplication not addition
//             }
//             prod[i][j] = sum;
//         }
//     }
//     return prod;
// }

// ======== CONV2D IMPLEMENTATION ======== //
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
        Matrix padded(input.rows + 2*pad, input.cols + 2*pad);
        for(size_t i=0; i<input.rows; ++i) {
            for(size_t j=0; j<input.cols; ++j) {
                padded[i+pad][j+pad] = input[i][j];
            }
        }
        return padded;
    }

    Matrix rotate180(const Matrix& m) const {
        Matrix rotated(m.rows, m.cols);
        for(size_t i=0; i<m.rows; ++i) {
            for(size_t j=0; j<m.cols; ++j) {
                rotated[i][j] = m[m.rows-1-i][m.cols-1-j];
            }
        }
        return rotated;
    }

public:
    Conv2D(int in_ch, int out_ch, int k_size, int stride=1, int pad=0)
        : in_channels(in_ch), out_channels(out_ch),
          kernel_size(k_size), stride(stride), padding(pad) {
        
        // He initialization with correct fan-in
        float stddev = sqrt(2.0f / (in_channels * kernel_size * kernel_size));
        std::default_random_engine generator;
        std::normal_distribution<float> dist(0.0f, stddev);

        // Initialize kernels and biases
        kernels.reserve(out_channels);
        for(int i=0; i<out_channels; ++i) {
            kernels.emplace_back(kernel_size, kernel_size);
            for(int h=0; h<kernel_size; ++h) {
                for(int w=0; w<kernel_size; ++w) {
                    kernels[i][h][w] = dist(generator);
                }
            }
        }
        biases = std::vector<float>(out_channels, 0.1f);
    }

    Matrix forward(const Matrix& input) {
        // Store original input and create padded version
        last_input = input;
        last_input_padded = pad_matrix(input, padding);

        const int out_h = (input.rows - kernel_size + 2*padding)/stride + 1;
        const int out_w = (input.cols - kernel_size + 2*padding)/stride + 1;
        
        Matrix output(out_channels, out_h * out_w);

        // Convolution operation
        for(int k=0; k<out_channels; ++k) {
            for(int i=0; i<out_h; ++i) {
                for(int j=0; j<out_w; ++j) {
                    float sum = 0.0f;
                    const int h_start = i*stride;
                    const int w_start = j*stride;
                    
                    for(int x=0; x<kernel_size; ++x) {
                        for(int y=0; y<kernel_size; ++y) {
                            sum += last_input_padded[h_start+x][w_start+y] * 
                                   kernels[k][x][y];
                        }
                    }
                    output[k][i*out_w + j] = sum + biases[k];
                }
            }
        }
        return output;
    }

    Matrix backward(const Matrix& output_grad, float lr) {
        const int out_h = (last_input.rows - kernel_size + 2*padding)/stride + 1;
        const int out_w = (last_input.cols - kernel_size + 2*padding)/stride + 1;
        
        Matrix input_grad(last_input.rows, last_input.cols);
        std::vector<Matrix> kernel_grads(out_channels);
        std::vector<float> bias_grads(out_channels, 0.0f);

        // Calculate gradients
        for(int k=0; k<out_channels; ++k) {
            kernel_grads[k] = Matrix(kernel_size, kernel_size);
            
            for(int i=0; i<out_h; ++i) {
                for(int j=0; j<out_w; ++j) {
                    const float grad = output_grad[k][i*out_w + j];
                    const int h_start = i*stride;
                    const int w_start = j*stride;
                    
                    // Kernel gradients
                    for(int x=0; x<kernel_size; ++x) {
                        for(int y=0; y<kernel_size; ++y) {
                            kernel_grads[k][x][y] += 
                                last_input_padded[h_start+x][w_start+y] * grad;
                        }
                    }
                    
                    // Bias gradients
                    bias_grads[k] += grad;
                    
                    // Input gradients
                    for(int x=0; x<kernel_size; ++x) {
                        for(int y=0; y<kernel_size; ++y) {
                            input_grad[h_start+x-padding][w_start+y-padding] += 
                                kernels[k][x][y] * grad;
                        }
                    }
                }
            }
        }

        // Update parameters
        for(int k=0; k<out_channels; ++k) {
            // Update kernel
            for(int x=0; x<kernel_size; ++x) {
                for(int y=0; y<kernel_size; ++y) {
                    kernels[k][x][y] -= lr * kernel_grads[k][x][y];
                }
            }
            // Update bias
            biases[k] -= lr * bias_grads[k];
        }

        return input_grad;
    }
};
