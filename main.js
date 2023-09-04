function matrixMultiply(A, B) {
  // Validate dimensions
  if (A[0].length !== B.length) {
    throw new Error(`Invalid matrix dimensions. ${A[0].length} does not equal ${B.length}`);
  }

  // Initialize result matrix with zeros
  const result = Array.from({ length: A.length }, () => Array(B[0].length).fill(0));

  // Perform multiplication
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < A[0].length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

function initializeWeights(rows, cols){
  // Initialize an empty matrix
  const matrix = [];

  // Populate the matrix with random values between -1 and 1
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(Math.random() * 2 - 1);
    }
    matrix.push(row);
  }

  return matrix;
}

function initializeBias(rows) {
  // Initialize an empty array
  const bias = [];

  // Populate the array with random values between -1 and 1
  for (let i = 0; i < rows; i++) {
    bias.push(Math.random() * 2 - 1);
  }

  return bias;
}

// activation functions
// derivatives are used in backpropagation
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function sigmoidDerivative(x) {
  const sigmoidValue = sigmoid(x);
  return sigmoidValue * (1 - sigmoidValue);
}

function relu(x) {
  return Math.max(0, x);
}
function reluDerivative(x) {
  return x > 0 ? 1 : 0;
}

// loss function
function mse(y_true, y_pred) {
  // Validate array lengths
  if (y_true.length !== y_pred.length) {
    throw new Error('Outer array lengths must match');
  }

  let sum = 0;
  let count = 0;

  for (let i = 0; i < y_true.length; i++) {
    if (Array.isArray(y_true[i]) && Array.isArray(y_pred[i])) {
      if (y_true[i].length !== y_pred[i].length) {
        throw new Error('Inner array lengths must match');
      }
      for (let j = 0; j < y_true[i].length; j++) {
        sum += Math.pow(y_true[i][j] - y_pred[i][j], 2);
        count++;
      }
    } else {
      sum += Math.pow(y_true[i] - y_pred[i], 2);
      count++;
    }
  }

  return sum / count;
}

function forwardPropagation(input, weights, bias, activationFunction) {
  // Step 1: Matrix multiplication
  const weightedSum = matrixMultiply(input, weights);

  // Step 2: Add bias
  for (let i = 0; i < weightedSum.length; i++) {
    for (let j = 0; j < weightedSum[0].length; j++) {
      weightedSum[i][j] += bias[j];
    }
  }

  // Step 3: Apply activation function
  const output = weightedSum.map(row => row.map(col => activationFunction(col)));

  return output;
}

// Backpropagation
function backPropagation(y_true, y_pred, input, weights, bias, learning_rate, activationDerivative) {
  // Step 1: Compute the error
  const error = y_true.map((row, i) => row.map((val, j) => val - y_pred[i][j]));

  // Step 2: Compute the derivative of the error with respect to the output (dE/dO)
  const dError_dOutput = error.map(row => row.map(val => -2 * val));

  // Step 3: Compute the derivative of the output with respect to the weighted sum (dO/dZ)
  const dOutput_dWeightedSum = y_pred.map(row => row.map(col => activationDerivative(col)));

  // Step 4: Compute the derivative of the weighted sum with respect to the weights (dZ/dW)
  const dWeightedSum_dWeights = input;

  // Step 5: Compute the gradient for weights
  let gradient = matrixMultiply(transpose(dWeightedSum_dWeights), dError_dOutput.map((row, i) => row.map((val, j) => val * dOutput_dWeightedSum[i][j])));

  // Step 6: Update weights and bias
  const newWeights = [...weights];
  const newBias = [...bias];

  if (gradient.length === weights.length && gradient[0].length === weights[0].length) {
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights[0].length; j++) {
        newWeights[i][j] -= learning_rate * gradient[i][j];
      }
    }
  } else {
    console.log("Dimension mismatch between gradient and weights.");
  }

  for (let i = 0; i < bias.length; i++) {
    newBias[i] -= learning_rate * dError_dOutput.reduce((sum, row) => sum + row[i], 0) * dOutput_dWeightedSum.reduce((sum, row) => sum + row[i], 0);
  }

  return { newWeights, newBias };
}

// Helper function to transpose a matrix
function transpose(matrix) {
  return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

function train(data, labels, epochs, learning_rate) {
  // Initialize weights and biases
  const inputSize = data[0].length;
  const outputSize = labels[0].length;
  let weights = initializeWeights(inputSize, outputSize);
  let bias = initializeBias(outputSize);

  // Training loop
  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < data.length; i++) {
      // Forward propagation
      const input = [data[i]];  // Make sure this is compatible with your other functions
      const y_true = [labels[i]];  // Wrapped in an array to match the shape of y_pred
      const y_pred = forwardPropagation(input, weights, bias, sigmoid);

      // Compute loss
      totalLoss += mse(y_true, y_pred);

      // Backpropagation and update weights and biases
      const { newWeights, newBias } = backPropagation(y_true, y_pred, input, weights, bias, learning_rate, sigmoidDerivative);
      weights = newWeights;
      bias = newBias;
    }

    // Log the average loss for this epoch
    console.log(`Epoch ${epoch + 1} / ${epochs} - Loss: ${totalLoss / data.length}`);
  }

  return { weights, bias };
}

function predict(input, weights, bias, activationFunction) {
  // Perform forward propagation to get the output
  const output = forwardPropagation(input, weights, bias, activationFunction);

  // For simplicity, let's assume the output is binary and apply a threshold
  return output[0].map(val => (val > 0.5 ? 1 : 0));
}