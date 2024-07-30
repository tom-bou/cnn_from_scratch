import numpy as np
from process import load_dataset
from model import CNN
from tqdm import tqdm

def cross_entropy_loss(output, target):
    m = target.shape[0]
    log_likelihood = -np.log(output[range(m), target])
    loss = np.sum(log_likelihood) / m
    return loss

def backward_cross_entropy(output, target):
    m = target.shape[0]
    grad = output.copy()
    grad[range(m), target] -= 1
    grad = grad / m
    return grad


def train(model, train_x, train_y, epochs, learning_rate, batch_size):
    num_batches = len(train_x) // batch_size

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx in tqdm(range(num_batches)):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            x_batch = train_x[batch_start:batch_end]
            y_batch = train_y[batch_start:batch_end]

            output = model.forward(x_batch)
            loss = cross_entropy_loss(output, y_batch)
            epoch_loss += loss

            d_output = backward_cross_entropy(output, y_batch)

            model.backward(d_output, learning_rate)

        print(f'Epoch {epoch + 1}, Loss: {loss}')

def evaluate(model, test_X, test_y):
    output = model.forward(test_X)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions == test_y)
    return accuracy

(train_x, train_y), (test_x, test_y) = load_dataset()
model = CNN()
print("Training...")
train(model, train_x, train_y, epochs=10, learning_rate=0.01, batch_size=64)
print("Evaluating...")
accuracy = evaluate(model, test_x, test_y)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
