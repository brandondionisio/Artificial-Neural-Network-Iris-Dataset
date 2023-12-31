# Artificial-Neural-Network-Iris-Dataset

Implementation of a multi-layered neural network that classifies iris flowers based on sepal length, sepal width, petal length, and petal width. The ANN is trained using the Iris dataset and the program prompts user input.

## Title

CS 131 HW 06 - Gardens of Heaven

## Author

Brandon Dionisio

## Acknowledgements

stackoverflow  
CS 131 Canvas Slides  
numpy.org  
pandas.pydata.org  
docs.python.org

## Running The Program

To run the Iris classifier, use "python iris.py"

## User Inputs

The user will be prompted with inputting the information for an iris.  
The iris information is in the following format:

"[sepal_length] [sepal_width] [petal_length] [petal_width]"

Note: Decimals under 1 must have a preceding 0 for each number.

The user can also input "q" to quit the program.

The user will continue to be requested input until one of the above inputs are given.

## Hidden layer

The program has a hidden layer consisting of 4 neurons.

## Layer calculations

The program utilizes the sigmoid function to calculate the first layer and
the softmax function for the second.

## Flexibility

The globals.py file allows the changing of the following:

rate of change of weights during training  
rate of change of biases during training  
total number of epochs for training  
printing frequency of epochs during training  
probability of setting an input to zero during the dropout function

## Additional notes

To prevent overflow in the sigmoid function due to exponentialization, the values are clipped to (-500, 500).

Initial weights and biases are chosen randomly with a numpy randn distribution
