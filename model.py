import numpy as np

class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size, stride, padding):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

    def forward(self, image):
        self.last_input = image
        self.input_padded = np.pad(image, [(0,0), (self.padding, self.padding), (self.padding, self.padding)], mode='constant')
        h, w = image.shape
        self.h = h
        self.w = w
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                for k in range(self.num_filters):
                    self.input_padded = image[i:i+self.filter_size, j:j+self.filter_size]
                    output[i, j, k] = np.sum(self.input_padded * self.filters[k])

        return output
    
    def backward(self, d_output, learning_rate):
        d_filters = np.zeros(self.filters.shape)
        d_input_padded = np.zeros((self.h + 2 * self.padding, self.w + 2 * self.padding))
        
        h_out, w_out = d_output.shape
        
        for i in range(self.num_filters):
            for j in range(0, h_out, self.stride):
                for k in range(0, w_out, self.stride):
                    region = self.input_padded[j:j+self.filter_size, k:k+self.filter_size]
                    d_filters[i] += region * d_output[j, k, i]
                    d_input_padded[j:j+self.filter_size, k:k+self.filter_size] += self.filters[i] * d_output[j, k, i]
        
        self.filters -= learning_rate * d_filters
        
        if self.padding == 0:
            return d_input_padded
        else:
            return d_input_padded[self.padding:-self.padding, self.padding:-self.padding]
    
class MaxPoolingLayer:
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride
    
    def forward(self, image):
        self.last_input = image
        h, w = input.shape
        h_out = (h - self.filter_size) // self.stride + 1
        w_out = (w - self.filter_size) // self.stride + 1
        
        self.output = np.zeros((h_out, w_out))
        
        for i in range(0, h_out*self.stride, self.stride):
            for j in range(0, w_out*self.stride, self.stride):
                region = image[i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size]
                self.output[i//self.stride, j//self.stride] = np.max(region)
        
        return self.output

    def backward(self, d_output):
        d_input = np.zeros(self.last_input.shape)
        
        h_out, w_out = d_output.shape
        
        for i in range(h_out):
            for j in range(w_out):
                region = d_input[i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size]
                max_val = np.max(region)
                for k in range(self.filter_size):
                    for l in range(self.filter_size):
                        d_input[i + k, j + l] = d_output[i//self.stride, j //self.stride]
        
        

