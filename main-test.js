// import * as main from './main.js';
import { matrixMultiply, initializeWeights, initializeBias, sigmoid, sigmoidDerivative, relu, reluDerivative, mse, forwardPropagation, backPropagation, train, predict } from './main.js';

function testMatrixMultiply(){
  const A = [
    [1, 2],
    [3, 4]
  ];
  
  const B = [
    [5, 6],
    [7, 8]
  ];
  
  const C = matrixMultiply(A, B);
  const expect = [[19, 22], [43, 50]];

  if (JSON.stringify(C) == JSON.stringify(expect)){
    console.log('Test passed');
  } else {
    console.log('Test failed');
    console.log('expect: ', expect);
    console.log('got: ', C);
  }
}

function testInitializeWeights(){
  // Test the function
  const weights = initializeWeights(2, 3);
  const expectRow = 2;
  const expectCol = 3;

  if(weights.length === expectRow && weights[0].length === expectCol){
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

function testInitializeBias(){
  // Test the function
  const bias = initializeBias(3);
  const expect = 3;

  if(bias.length === expect){
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

function testSigmoid(){
  // Test the function
  const x = 0;
  const y = sigmoid(x);
  const expect = 0.5;

  if(y === expect){
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

function testSigmoidDerivative(){
  // Test the function
  const x = 0;
  const y = sigmoidDerivative(x);
  const expect = 0.25;

  if(y === expect){
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

function testRelu(){
  // Test the function
  const x = 0;
  const y = relu(x);
  const expect = 0;

  if(y === expect){
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

function testReluDerivative(){
  // Test the function
  const x = 0;
  const y = reluDerivative(x);
  const expect = 0;

  if(y === expect){
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

function testMse(){
  // Test the function
  const y_true = [1, 2, 3];
  const y_pred = [1, 2, 3];
  const y = mse(y_true, y_pred);
  const expect = 0;

  if(y === expect){
    console.log('Test passed');
  } else {
    console.log('Test failed');
  }
}

function testForwardPropagation(){
  // Test the function
  const X = [
    [1, 2],
    [3, 4]
  ];
  const W = [
    [5, 6],
    [7, 8]
  ];
  const b = [9, 10];

  // Using identity function as activation function for testing
  const identity = x => x;

  const y = forwardPropagation(X, W, b, identity);
  const expect = [[28, 32], [52, 60]];

  if(JSON.stringify(y) === JSON.stringify(expect)){
    console.log('Test passed');
  } else {
    console.log('Test failed');
    console.log('expect: ', expect);
    console.log('got: ', y);
  }
}

function testBackPropagation(){
  // Test the function
  const y_true = [[1, 1], [1, 1]];  // 2x2 to match X's number of rows and W's number of columns
  const y_pred = [[1, 1], [1, 1]];  // 2x2 to match y_true
  const X = [
    [1, 2],
    [3, 4]
  ];  // 2x2
  const W = [
    [0.5, 0.5],
    [0.5, 0.5]
  ];  // 2x2 to match X's number of columns and y_true's number of columns
  const b = [0, 0];  // Length 2 to match the number of columns in y_true and y_pred
  const learning_rate = 0.01;
  const identityDerivative = x => 1;

  const result = backPropagation(y_true, y_pred, X, W, b, learning_rate, identityDerivative);
  
  // Your expected new weights would depend on your backPropagation function's logic
  // For this example, let's assume they should be the same as the initial weights
  const expect = [
    [0.5, 0.5],
    [0.5, 0.5]
  ];

  if(JSON.stringify(result.newWeights) === JSON.stringify(expect)){
    console.log('Test passed');
  } else {
    console.log('Test failed');
    console.log('Expected new weights: ', expect);
    console.log('Got new weights: ', result.newWeights);
  }
}

function testTrain(){
  // Test the function
  const X = [
    [1, 2],
    [3, 4]
  ];
  const y_true = [[1, 1], [1, 1]];
  const epochs = 100;
  const learning_rate = 0.01;

  // Run the training
  const { weights, bias } = train(X, y_true, epochs, learning_rate);

  // Run a forward pass with the final weights and biases
  const y_pred_after_training = forwardPropagation(X, weights, bias, sigmoid);

  // Calculate the loss after training
  const loss_after_training = mse(y_true, y_pred_after_training);

  // Your expected output would depend on your train function's logic
  // For this example, let's assume the loss should be less than a certain threshold
  const loss_threshold = 0.1;

  if(loss_after_training < loss_threshold){
    console.log('Test passed');
  } else {
    console.log('Test failed');
    console.log('Final loss: ', loss_after_training);
  }
}

function testPredict(){
  // Test the function
  const X = [
    [1, 2],
    [3, 4]
  ];
  const W = [
    [5, 6],
    [7, 8]
  ];
  const b = [9, 10];

  const sigmoid = x => 1 / (1 + Math.exp(-x));
  
  // Run the prediction
  const y = predict(X, W, b, sigmoid);

  // Expected output
  const expect = [1, 1];

  // Check if the prediction matches the expected output
  if(JSON.stringify(y) === JSON.stringify(expect)){
    console.log('Test passed');
  } else {
    console.log('Test failed');
    console.log('Expected: ', expect);
    console.log('Got: ', y);
  }
}

function testMain(){
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
}

function testMain2(){
  const trainData = [
    [0, 0],  // Below the line y = x
    [1, 0],  // Below
    [0, 1],  // Above
    [1, 1],  // On the line
    [2, 1],  // Below
    [1, 2],  // Above
    [2, 2],  // On the line
    [3, 2],  // Below
    [2, 3],  // Above
  ];
  
  // Labels: 0 for below the line, 1 for above or on the line
  const trainLabels = [
    [0],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
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
}

function testAll(){
  testMatrixMultiply();
  testInitializeWeights();
  testInitializeBias();
  testSigmoid();
  testSigmoidDerivative();
  testRelu();
  testReluDerivative();
  testMse();
  testForwardPropagation();
  testBackPropagation();
  testTrain();
  testPredict();
  testMain();
}