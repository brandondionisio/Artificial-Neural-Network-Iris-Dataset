import numpy as np
import pandas as pd
import re
from globals import WEIGHT_RATE, BIAS_RATE, NUM_EPOCHS, EPOCH_FREQ, DROPOUT_PROBABILITY

# sigmoid function
# changes the range of the values in the input array to (0,1)
# when deriv == True, returns the derivative of the sigmoid function (for backprop)
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    # prevent overflow by limiting min and max range
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

# cross entropy error function
# loss function for the neural network
# target is the hot encoded vector of targets
def cross_entropy(output, target):
    return - np.sum(np.log(output) * (target))

# cost function
# calculates the mean cross entropy across data points
# target is the hot encoded vector of targets
def cost(output, target):
    return np.mean(cross_entropy(output, target))

# soft max function
# normalizes the given array using exponentialization
def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()

# dropout function
# prevents overfitting by randomly setting a fraction
# of input units to zero during training
def dropout(a, prob):
    shape = a.shape[0]
    vec = np.random.choice([0,1], size = (shape,1), p = [prob, 1-prob])
    return vec * a

# determines if the given string is a valid input for the query loop
# the string must fit the form: "[number1] [number2] [number3] [number4]"
def is_valid_string(s):
    # Define a regex pattern for four numbers separated by spaces
    pattern = re.compile(r'^\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?$')

    # Check if the string matches the pattern
    return bool(pattern.match(s))

# trains the given iris data
# returns the first and second layer weights and biases after training
def train(iris_data, iris_targets):
    # initialize weights and biases
    # divide by sqrt(4) to prevent explosive gradients
    weight0 = np.random.randn(4,4) / 2
    weight1 =  np.random.randn(3,4) / 2
    bias0 =  np.random.randn(4,1) / 2
    bias1 = np.random.randn(3,1) /2

    print("Training the dataset...")
    for epoch in range(NUM_EPOCHS):
        for i in range(150):
            # forward prop
            l0 = iris_data[i][None].T
            l0 = dropout(l0, DROPOUT_PROBABILITY)
            l1 = sigmoid(np.dot(weight0, l0) + bias0)
            l1 = dropout(l1, DROPOUT_PROBABILITY)
            l2 = softmax(np.dot(weight1, l1) + bias1) 

            # setup target one hot encoded vector
            target = np.zeros([3,1])
            target[iris_targets[i]][0] = 1

            # back propagation
            weight1_delta = np.outer((l2-target), l1.T)
            bias1_delta = -(l2-target)
            error_hidden = np.dot(weight1.T, (l2 - target)) * sigmoid(l1, True)
            weight0_delta = np.outer(error_hidden,l0.T)
            bias0_delta = -error_hidden
            
            # adjust weights with learning rate
            weight0 -= WEIGHT_RATE * weight0_delta
            weight1 -= WEIGHT_RATE * weight1_delta
            bias0 += BIAS_RATE * bias0_delta
            bias1 += BIAS_RATE * bias1_delta
            
            if i == 149 and (epoch) % EPOCH_FREQ == 0:
                loss = cost(l2, target)
                print("Epoch: ", epoch)
                print("Loss: ", loss, end="\n")
    
    return weight0, weight1, bias0, bias1

def query(weight0, weight1, bias0, bias1):
    user = input()
    # input query loop
    valid_input = False
    # accept input only if is valid iris information (four numbers separated by spaces)
    while not valid_input:
        if user != "q" and not is_valid_string(user):
            print("Invalid input. Please try again: ", end="")
            user = input()
        elif user != "q":
            unknown_iris = user.split()
            unknown_iris = [float(substring) for substring in unknown_iris]

            # calculates layers with weights and biases
            layer0 = np.array(unknown_iris)
            layer0 = np.reshape(layer0, (len(layer0), 1))
            layer1 = sigmoid(np.dot(weight0, layer0) + bias0)
            layer2 = softmax(np.dot(weight1, layer1) + bias1)

            # prints the identified iris
            max_dims = np.argmax(layer2)
            row_index, _ = np.unravel_index(max_dims, layer2.shape)
            if row_index == 0:
                print("Iris type: Iris Setosa")
            elif row_index == 1:
                print("Iris type: Iris Versicolor")
            else:
                print("Iris type: Iris Virginica")

            print("Enter information for another iris: ", end="")
            user = input()
        else:
            print("Bye!")
            valid_input = True
    return

def main():
    # Load the iris data into a pandas DataFrame
    file_path = 'ANN - Iris data.txt'
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv(file_path, header=None, names=columns)

    # Map the class names to numeric values
    class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['class'] = df['class'].map(class_mapping)

    # Extract features and labels as numpy arrays
    iris_data = df.iloc[:, :-1].to_numpy()
    iris_targets = df['class'].to_numpy()

    # train iris data
    weight0, weight1, bias0, bias1 = train(iris_data, iris_targets)

    print("\nHello gardener!")
    print("Enter the iris information with four numbers separated by spaces.")
    print("Use the following form: \"[sepal_length] [sepal_width] [petal_length] [petal_width]\"")
    print("Decimals under 1 must have a preceding 0.")
    print("Type \"q\" to quit.\n\nInformation: ", end="")

    # user input query loop for testing
    query(weight0, weight1, bias0, bias1)

if __name__ == "__main__":
    main()