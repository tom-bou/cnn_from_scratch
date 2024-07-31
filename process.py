import numpy as np

def load_dataset():
    # Load the training data
    train_data = np.loadtxt('mnist/mnist_train.csv', delimiter=',', skiprows=1)
    train_y = train_data[:, 0].astype(int)
    train_x = train_data[:, 1:].astype(float) / 255.0
    train_x = train_x.reshape(-1, 28, 28, 1)
    print(train_x.shape)

    # Load the test data
    test_data = np.loadtxt('mnist/mnist_test.csv', delimiter=',', skiprows=1)
    test_y = test_data[:, 0].astype(int)
    test_x = test_data[:, 1:].astype(float) / 255.0
    test_x = test_x.reshape(-1, 28, 28, 1)
    

 
    
    return (train_x, train_y), (test_x, test_y)
