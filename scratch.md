Switching from a single-layer to a multi-layer perceptron (MLP) involves adding one or more hidden layers between the input and output layers. 

Initialize Additional Weights and Biases: For each hidden layer, you'll need a set of weights and biases.

Forward Propagation: You'll need to extend your forwardPropagation function to handle multiple layers. Each layer's output becomes the next layer's input.

Backpropagation: This gets more complex with additional layers. You'll need to calculate gradients for each layer and update the weights and biases accordingly.

Example of one hidden layer with 3 neurons:

javascript
Copy code
// Initialize weights and biases for input layer and one hidden layer
let input_weights = initializeWeights(inputSize, 3);  // 3 neurons in hidden layer
let hidden_weights = initializeWeights(3, outputSize);  // 3 neurons in hidden layer to output

let input_bias = initializeBias(3);  // Bias for hidden layer
let output_bias = initializeBias(outputSize);  // Bias for output layer

// Forward Propagation
function forwardPropagation(input, input_weights, hidden_weights, input_bias, output_bias, activationFunction) {
  const hidden_layer_input = matrixMultiply(input, input_weights);
  addBias(hidden_layer_input, input_bias);
  const hidden_layer_output = applyActivation(hidden_layer_input, activationFunction);

  const output_layer_input = matrixMultiply(hidden_layer_output, hidden_weights);
  addBias(output_layer_input, output_bias);
  const output = applyActivation(output_layer_input, activationFunction);

  return output;
}

// ... You'll need to extend your existing backPropagation function to handle the hidden layer.