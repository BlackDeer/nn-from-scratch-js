# nn-from-scratch-js

Just trying to build everything from scratch in JS for self educational purposes.

## Usage
See main-test.js for usage details.

```js
// Sample data for AND Function
  const trainData = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];

  const trainLabels = [
    [0],
    [0],
    [0],
    [1]
  ];

  // Hyperparameters
  const epochs = 5000;
  const learning_rate = 0.1;

  // Train the neural network
  console.log("Training the neural network...");
  const { weights, bias } = train(trainData, trainLabels, epochs, learning_rate);
  console.log(weights, bias)

  // Test the neural network
  console.log("Testing the neural network...");
  for (const input of trainData) {
    const prediction = predict([input], weights, bias, sigmoid);  // Wrap 'input' in an array to make it 2D
    console.log(`For input [${input}], prediction is ${prediction}`);
  }
```

## Includes
**Matrix Multiplication Function**

`matrixMultiply(A, B)` that takes two matrices (2D arrays) and returns their product.

**Random Weights and Bias Generation**

`initializeWeights(rows, cols)` that returns a matrix with random weights.
`initializeBias(rows)` that returns a vector with random biases.

**Activation Function**

Implements the Sigmoid activation function `sigmoid(x)`.
Implements its derivative `sigmoidDerivative(x)`.

**Loss Function**

Implement a Mean Squared Error (MSE) loss function `mse(y_true, y_pred)`.

**Forward Propagation**

`forwardPropagation(input, weights, bias)` that performs one forward pass.

**Backpropagation**

`backPropagation(y_true, y_pred, weights, learning_rate)` that updates the weights and biases.

**Neural Network Trainer**

`train(data, labels, epochs, learning_rate)` that trains the neural network.

**Prediction Function**

`predict(input)` that takes an input and returns a prediction.

**Testing and Validation**

Use some sample data to train and test your neural network.

## Limitations
Currnet implmentation is limited to:
- 2D data
- single layer perceptrons
- binary prediction (0 or 1)

A single-layer perceptron is essentially a linear classifier. It can only deal with problems that are linearly separable. 
