import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {

    Neuron[] inputLayer;
    Neuron[] hiddenLayer;
    Neuron[] outputLayer;

    // Store the last activations for backpropagation
    private double[] lastInputs;
    private double[] lastHiddenOutputs;
    private double[] lastOutputs;

    public NeuralNetwork(int numInputs, int numHidden, int numOutputs) {
        inputLayer = new Neuron[numInputs];
        hiddenLayer = new Neuron[numHidden];
        outputLayer = new Neuron[numOutputs];

        for (int i = 0; i < numInputs; i++) {
            inputLayer[i] = new Neuron(numInputs); // or new Neuron(1) if input neurons don't need weights
        }
        for (int i = 0; i < numHidden; i++) {
            hiddenLayer[i] = new Neuron(numInputs);
        }
        for (int i = 0; i < numOutputs; i++) {
            outputLayer[i] = new Neuron(numHidden);
        }
    }

	public double[] forward(double[] data) {
        // Store inputs for backpropagation
        this.lastInputs = Arrays.copyOf(data, data.length);

        // Step 1: Compute hidden layer activations
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activate(data);
        }

        // Store hidden outputs for backpropagation
        this.lastHiddenOutputs = Arrays.copyOf(hiddenOutputs, hiddenOutputs.length);

        // Step 2: Compute output layer activations
        double[] outputOutputs = new double[outputLayer.length];
        for (int i = 0; i < outputLayer.length; i++) {
            outputOutputs[i] = outputLayer[i].activate(hiddenOutputs);
        }

        // Store outputs for backpropagation
        this.lastOutputs = Arrays.copyOf(outputOutputs, outputOutputs.length);

        return outputOutputs;
	}

    public void setWeights(double[][] hiddenWeights, double[][] outputWeights) {
        // Set weights for hidden layer
        for (int i = 0; i < hiddenLayer.length; i++) {
            if (i < hiddenWeights.length) {
                for (int j = 0; j < hiddenLayer[i].weights.length && j < hiddenWeights[i].length; j++) {
                    hiddenLayer[i].weights[j] = hiddenWeights[i][j];
                }
            }
        }
        // Set weights for output layer
        for (int i = 0; i < outputLayer.length; i++) {
            if (i < outputWeights.length) {
                for (int j = 0; j < outputLayer[i].weights.length && j < outputWeights[i].length; j++) {
                    outputLayer[i].weights[j] = outputWeights[i][j];
                }
            }
        }
    }

    /**
     * Get the current weights of the network
     * @return Array containing hidden and output layer weights
     */
    public double[][][] getWeights() {
        double[][] hiddenWeights = new double[hiddenLayer.length][];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenWeights[i] = Arrays.copyOf(hiddenLayer[i].weights, hiddenLayer[i].weights.length);
        }

        double[][] outputWeights = new double[outputLayer.length][];
        for (int i = 0; i < outputLayer.length; i++) {
            outputWeights[i] = Arrays.copyOf(outputLayer[i].weights, outputLayer[i].weights.length);
        }

        return new double[][][] { hiddenWeights, outputWeights };
    }

    /**
     * Train the neural network using backpropagation
     * @param trainingData List of training data
     * @param learningRate Learning rate for weight updates (how fast the network learns)
     * @param epochs Number of training epochs (how many times to go through all examples)
     */
    public void train(List<TrainingData> trainingData, double learningRate, int epochs) {
        System.out.println("Starting training for " + epochs + " epochs...");

        // Repeat the training process 'epochs' times
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;

            // Go through each training example
            for (TrainingData data : trainingData) {
                // Step 1: Forward pass - get the network's prediction
                double[] inputs = data.getInputs();                // RGB values
                double[] expectedOutputs = data.getExpectedOutputs(); // Correct traffic light class
                double[] actualOutputs = forward(inputs);          // Network's prediction

                // Step 2: Calculate how wrong the prediction was (error)
                double error = 0;
                for (int i = 0; i < actualOutputs.length; i++) {
                    // Square the difference between expected and actual
                    error += Math.pow(expectedOutputs[i] - actualOutputs[i], 2);
                }
                totalError += error;

                // Step 3: Update the weights to make the network better
                backpropagate(expectedOutputs, learningRate);
            }

            // Print progress every 100 epochs
            if (epoch % 100 == 0 || epoch == epochs - 1) {
                System.out.println("Epoch " + epoch + ", Error: " + totalError);
            }
        }

        System.out.println("Training completed.");
    }

    /**
     * Backpropagation algorithm to update weights
     * This is how the network learns from its mistakes
     * @param expectedOutputs Expected output values
     * @param learningRate Learning rate for weight updates
     */
    private void backpropagate(double[] expectedOutputs, double learningRate) {
        // STEP 1: Calculate errors in the output layer
        // ---------------------------------------------
        double[] outputErrors = new double[outputLayer.length];
        double[] outputDeltas = new double[outputLayer.length];

        for (int i = 0; i < outputLayer.length; i++) {
            // How wrong was the prediction? (expected - actual)
            outputErrors[i] = expectedOutputs[i] - lastOutputs[i];

            // Calculate delta (error * derivative) for weight updates
            outputDeltas[i] = outputErrors[i] * outputLayer[i].sigmoidDerivative(lastOutputs[i]);
        }

        // STEP 2: Calculate errors in the hidden layer
        // -------------------------------------------
        double[] hiddenErrors = new double[hiddenLayer.length];
        double[] hiddenDeltas = new double[hiddenLayer.length];

        for (int i = 0; i < hiddenLayer.length; i++) {
            // Each hidden neuron contributed to all output errors
            for (int j = 0; j < outputLayer.length; j++) {
                hiddenErrors[i] += outputDeltas[j] * outputLayer[j].weights[i];
            }
            hiddenDeltas[i] = hiddenErrors[i] * hiddenLayer[i].sigmoidDerivative(lastHiddenOutputs[i]);
        }

        // STEP 3: Update weights in the output layer
        // -----------------------------------------
        for (int i = 0; i < outputLayer.length; i++) {
            for (int j = 0; j < outputLayer[i].weights.length - 1; j++) {
                // Update each weight: weight += learning_rate * delta * input
                outputLayer[i].weights[j] += learningRate * outputDeltas[i] * lastHiddenOutputs[j];
            }
            // Update bias weight (the extra weight)
            outputLayer[i].weights[outputLayer[i].weights.length - 1] += learningRate * outputDeltas[i];
        }

        // STEP 4: Update weights in the hidden layer
        // -----------------------------------------
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].weights.length - 1; j++) {
                // Update each weight: weight += learning_rate * delta * input
                hiddenLayer[i].weights[j] += learningRate * hiddenDeltas[i] * lastInputs[j];
            }
            // Update bias weight (the extra weight)
            hiddenLayer[i].weights[hiddenLayer[i].weights.length - 1] += learningRate * hiddenDeltas[i];
        }
    }

    /**
     * Test the neural network on the training data and print results
     * @param testData The data to test on
     * @return The accuracy percentage (0-100)
     */
    public double testNetwork(List<TrainingData> testData) {
        // Counter for correct predictions
        int correct = 0;

        // Test each example
        for (TrainingData data : testData) {
            // Get the inputs (RGB values)
            double[] inputs = data.getInputs();

            // Get the expected outputs (correct traffic light class)
            double[] expectedOutputs = data.getExpectedOutputs();

            // Get the network's prediction
            double[] actualOutputs = forward(inputs);

            // Find which class has the highest value (the network's prediction)
            int expectedClass = findMaxIndex(expectedOutputs);
            int actualClass = findMaxIndex(actualOutputs);

            // Print the result
            System.out.print("Input (RGB): ");
            for (double input : inputs) {
                System.out.printf("%.2f ", input);
            }
            System.out.print("| Expected: " + expectedClass + " | Actual: " + actualClass);

            // Check if the prediction was correct
            if (expectedClass == actualClass) {
                System.out.println(" ✓");  // Correct
                correct++;
            } else {
                System.out.println(" ✗");  // Wrong
            }
        }

        // Calculate and print the accuracy
        double accuracy = (double) correct / testData.size() * 100;
        System.out.printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, testData.size());

        return accuracy;
    }

    /**
     * Find the index of the maximum value in an array
     * This helps us determine which class the network predicted
     * @param array The array to search
     * @return The index of the maximum value
     */
    public int findMaxIndex(double[] array) {
        // Start with the first element
        int maxIndex = 0;
        double maxValue = array[0];

        // Check each element
        for (int i = 1; i < array.length; i++) {
            // If this element is bigger, remember it
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        // Return the index of the biggest value
        return maxIndex;
    }

    @Override
    public String toString() {
        return "NeuralNetwork [inputLayer=" + Arrays.toString(inputLayer) + ", hiddenLayer="
                + Arrays.toString(hiddenLayer) + ", outputLayer=" + Arrays.toString(outputLayer) + "]";
    }
}