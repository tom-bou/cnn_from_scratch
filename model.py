import numpy as np

class ConvolutionalLayer:
    def __init__(self, num_channels,num_filters, filter_size, stride, padding):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size, num_channels) / (filter_size * filter_size)

    def forward(self, image):
        self.last_input = image
        self.input_padded = np.pad(image, [(0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)], mode='constant')

        batch_size, h, w, _ = image.shape
        self.batch_size = batch_size
        self.h = h
        self.w = w
        output_height = (h + 2 * self.padding - self.filter_size) // self.stride + 1
        output_width = (w + 2 * self.padding - self.filter_size) // self.stride + 1
        output = np.zeros((self.batch_size, output_height, output_width, self.num_filters))

        for i in range(output_height):
            for j in range(output_width):
                for k in range(self.num_filters):
                    region = self.input_padded[:, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                    output[:, i, j, k] = np.sum(region * self.filters[k, :, :, :], axis=(1,2,3))

        return output
    
    def backward(self, d_output, learning_rate):
        d_filters = np.zeros_like(self.filters)
        d_input_padded = np.zeros_like(self.input_padded)

        batch_size, output_height, output_width, _ = d_output.shape

        for i in range(output_height):
            for j in range(output_width):
                for k in range(self.num_filters):
                    region = self.input_padded[:, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :]
                    d_filters[k, :, :, :] += np.sum(region * d_output[:, i, j, k][:, None, None, None], axis=0)
                    d_input_padded[:, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, :] += self.filters[k, :, :, :] * d_output[:, i, j, k][:, None, None, None]

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
        
        if h_out <= 0 or w_out <= 0:
            raise ValueError("Invalid output dimensions. Adjust filter size or stride.")
        
        self.output = np.zeros((batch_size, h_out, w_out, num_filters))
        
        for n in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for k in range(num_filters):
                        region = images[n, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, k]
                        if region.size == 0:
                            raise ValueError("Region size is zero. Adjust filter size or stride.")
                        self.output[n, i, j, k] = np.max(region)
        
        return self.output

    def backward(self, d_output):
        d_input = np.zeros_like(self.last_input)
        
        batch_size, h_out, w_out, num_filters = d_output.shape
        
        for n in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for k in range(num_filters):
                        region = self.last_input[n, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size, k]
                        max_val = np.max(region)
                        for m in range(self.filter_size):
                            for p in range(self.filter_size):
                                if region.size > 0 and region[m, p] == max_val:
                                    d_input[n, i*self.stride+m, j*self.stride+p, k] = d_output[n, i, j, k]
        
        return d_input

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
    
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
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


class CNN:
    def __init__(self):
        self.conv = ConvolutionalLayer(num_channels=1, num_filters=8, filter_size=3, stride=1, padding=1)
        self.pool = MaxPoolingLayer(filter_size=2, stride=2)
        self.conv2 = ConvolutionalLayer(num_channels=8, num_filters=16, filter_size=3, stride=1, padding=1)
        self.pool2 = MaxPoolingLayer(filter_size=2, stride=2)
        self.fc1 = DenseLayer(7*7*16, output_size=128)
        self.fc2 = DenseLayer(128, output_size=10)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, images):
        conv_out = self.conv.forward(images)
        pool_out = self.pool.forward(conv_out)
        conv2_out = self.conv2.forward(pool_out)
        pool2_out = self.pool2.forward(conv2_out)
        flattened = pool2_out.reshape(pool2_out.shape[0], -1)
        fc1_out = self.fc1.forward(flattened)
        fc2_out = self.fc2.forward(fc1_out)
        return self.softmax(fc2_out)
    
    def backward(self, d_output, learning_rate):
        print("D_output", d_output.shape)
        d_output = self.fc2.backward(d_output, learning_rate)
        print("D_output_FC2", d_output.shape)
        d_output = self.fc1.backward(d_output, learning_rate)
        print("D_output_FC1", d_output.shape)
        d_output = d_output.reshape(self.pool2.output.shape)
        print("D_output_FC1_Reshape", d_output.shape)
        d_output = self.pool2.backward(d_output)
        print("D_output_Pool2", d_output.shape)
        d_output = self.conv2.backward(d_output, learning_rate)
        print("D_output_Conv2", d_output.shape)
        d_output = self.pool.backward(d_output)
        print("D_output_Pool", d_output.shape)
        d_output = self.conv.backward(d_output, learning_rate)
        print("D_output_Conv", d_output.shape)
        
        return d_output

        
        
 
