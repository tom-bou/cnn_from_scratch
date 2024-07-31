import numpy as np
from tqdm import tqdm

class ConvolutionalLayer:
    def __init__(self, num_channels, num_filters, filter_size, stride, padding):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size, num_channels) * np.sqrt(2.0 / (filter_size * filter_size * num_channels))

    def forward(self, image):
        self.last_input = image
        self.input_padded = np.pad(image, [(0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)], mode='constant')

        batch_size, h, w, _ = image.shape
        self.batch_size = batch_size
        self.h = h
        self.w = w
        output_height = (h + 2 * self.padding - self.filter_size) // self.stride + 1
        output_width = (w + 2 * self.padding - self.filter_size) // self.stride + 1
        output = np.zeros((self.batch_size, output_height, output_width, self.num_filters))

        for i in range(output_height):
            for j in range(output_width):
                region = self.input_padded[:, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                output[:, i, j, :] = np.tensordot(region, self.filters, axes=([1, 2, 3], [1, 2, 3]))
        
        return output
    
    def backward(self, d_output, learning_rate):
        d_filters = np.zeros_like(self.filters)
        d_input_padded = np.zeros_like(self.input_padded)

        batch_size, output_height, output_width, _ = d_output.shape

        for i in range(output_height):
            for j in range(output_width):
                region = self.input_padded[:, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                for k in range(self.num_filters):
                    d_filters[k] += np.tensordot(region, d_output[:, i, j, k], axes=([0], [0]))
                for n in range(batch_size):
                    d_input_padded[n, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :] += np.sum(self.filters * (d_output[n, i, j, :, None, None, None]), axis=0)
        
        self.filters -= learning_rate * d_filters / batch_size

        if self.padding == 0:
            return d_input_padded
        else:
            return d_input_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]

class MaxPoolingLayer:
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride
    
    def forward(self, images):
        self.last_input = images
        batch_size, h, w, num_filters = images.shape
        h_out = (h - self.filter_size) // self.stride + 1
        w_out = (w - self.filter_size) // self.stride + 1
        
        self.output = np.zeros((batch_size, h_out, w_out, num_filters))
        
        for i in range(h_out):
            for j in range(w_out):
                region = images[:, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                self.output[:, i, j, :] = np.max(region, axis=(1, 2))
        
        return self.output

    def backward(self, d_output):
        d_input = np.zeros_like(self.last_input)
        
        batch_size, h_out, w_out, num_filters = d_output.shape
        
        for i in range(h_out):
            for j in range(w_out):
                region = self.last_input[:, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                max_region = np.max(region, axis=(1, 2))
                for n in range(batch_size):
                    for k in range(num_filters):
                        mask = (region[n, :, :, k] == max_region[n, k])
                        d_input[n, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, k] += mask * d_output[n, i, j, k]
        
        return d_input

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
    
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def ReLU_derivative(self, x):
        return (x > 0).astype(x.dtype)
    
    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        self.output = self.ReLU(self.z)
        return self.output
    
    def backward(self, d_output, learning_rate):
        d_z = d_output * self.ReLU_derivative(self.z)
        d_input = np.dot(d_z, self.weights.T)
        d_weights = np.dot(self.input.T, d_z)
        d_biases = np.sum(d_z, axis=0, keepdims=True)
        
        self.weights -= learning_rate * d_weights / self.input.shape[0]
        self.biases -= learning_rate * d_biases / self.input.shape[0]
        
        return d_input

class BatchNormalizationLayer:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones((1, 1, 1, num_features))
        self.beta = np.zeros((1, 1, 1, num_features))
        self.running_mean = np.zeros((1, 1, 1, num_features))
        self.running_var = np.ones((1, 1, 1, num_features))

    def forward(self, x, training=True):
        if training:
            batch_mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
            batch_var = np.var(x, axis=(0, 1, 2), keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.batch_mean = batch_mean
            self.batch_var = batch_var
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        self.x_centered = x - batch_mean
        self.stddev_inv = 1. / np.sqrt(batch_var + self.epsilon)
        self.x_normalized = self.x_centered * self.stddev_inv
        out = self.gamma * self.x_normalized + self.beta
        return out

    def backward(self, d_out, learning_rate):
        d_x_normalized = d_out * self.gamma
        d_var = np.sum(d_x_normalized * self.x_centered, axis=(0, 1, 2)) * -0.5 * np.power(self.stddev_inv, 3)
        d_mean = np.sum(d_x_normalized * -self.stddev_inv, axis=(0, 1, 2)) + d_var * np.mean(-2. * self.x_centered, axis=(0, 1, 2))
        
        d_x = (d_x_normalized * self.stddev_inv) + (d_var * 2 * self.x_centered / d_out.shape[0]) + (d_mean / d_out.shape[0])
        d_gamma = np.sum(d_out * self.x_normalized, axis=(0, 1, 2))
        d_beta = np.sum(d_out, axis=(0, 1, 2))

        self.gamma -= learning_rate * d_gamma
        self.beta -= learning_rate * d_beta

        return d_x


class CNN:
    def __init__(self):
        self.conv = ConvolutionalLayer(num_channels=1, num_filters=8, filter_size=3, stride=1, padding=1)
        self.bn1 = BatchNormalizationLayer(num_features=8)
        self.pool = MaxPoolingLayer(filter_size=2, stride=2)
        #self.conv2 = ConvolutionalLayer(num_channels=8, num_filters=16, filter_size=3, stride=1, padding=1)
        #self.bn2 = BatchNormalizationLayer(num_features=16)
        #self.pool2 = MaxPoolingLayer(filter_size=2, stride=2)
        self.fc1 = DenseLayer(14*14*8, 128)
        self.fc2 = DenseLayer(128, 10)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, images):
        conv_out = self.conv.forward(images)
        bn1_out = self.bn1.forward(conv_out)
        pool_out = self.pool.forward(bn1_out)
        
        #conv2_out = self.conv2.forward(pool_out)
        #bn2_out = self.bn2.forward(conv2_out)
        #pool2_out = self.pool2.forward(bn2_out)
        flattened = pool_out.reshape(pool_out.shape[0], -1)
        fc1_out = self.fc1.forward(flattened)
        fc2_out = self.fc2.forward(fc1_out)
        return self.softmax(fc2_out)
    
    def backward(self, d_output, learning_rate):
        d_output = self.fc2.backward(d_output, learning_rate)
        d_output = self.fc1.backward(d_output, learning_rate)
        d_output = d_output.reshape(self.pool.output.shape)
        #d_output = self.pool2.backward(d_output)
        #d_output = self.bn2.backward(d_output, learning_rate)
        #d_output = self.conv2.backward(d_output, learning_rate)
        d_output = self.pool.backward(d_output)
        d_output = self.bn1.backward(d_output, learning_rate)
        d_output = self.conv.backward(d_output, learning_rate)
        
        return d_output