// Define a simple sequential model
const model = tf.sequential();

// Add an embedding layer
model.add(tf.layers.embedding({
  inputDim: vocabularySize, // Size of the vocabulary
  outputDim: embeddingSize, // Size of the embedding vectors
  inputLength: maxSequenceLength // Length of input sequences
}));

// Add a flattening layer
model.add(tf.layers.flatten());

// Add a dense layer with sigmoid activation for binary classification
model.add(tf.layers.dense({
  units: 1, // Output units (1 for binary classification)
  activation: 'sigmoid' // Sigmoid activation function
}));

// Compile the model
model.compile({
  optimizer: 'adam', // Adam optimizer
  loss: 'binaryCrossentropy', // Binary cross-entropy loss function
  metrics: ['accuracy'] // Accuracy metric
});

// Function to make predictions using the TensorFlow.js model
async function predictSpamOrHam(message) {
  // Convert the message to tensor
  const inputTensor = convertMessageToTensor(message);

  // Make prediction using the model
  const prediction = model.predict(inputTensor);

  // Extract the prediction value
  const predictionValue = prediction.dataSync()[0];

  // Define a threshold for classification
  const threshold = 0.5;

  // Classify as spam if prediction value is above threshold, otherwise classify as ham
  const result = predictionValue > threshold ? 'spam' : 'ham';

  return result;
}

// Event listener for the submit button
document.getElementById('submitBtn').addEventListener('click', async () => {
  const userInput = document.getElementById('userInput').value;
  const prediction = await predictSpamOrHam(userInput);
  document.getElementById('predictionResult').innerText = `Prediction: ${prediction}`;
});

