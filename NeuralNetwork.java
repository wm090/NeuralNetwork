import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {

    Neuron[] inputLayer;
    Neuron[] hiddenLayer;
    Neuron[] outputLayer;

    // for backpropagation
    private double[] lastInputs;
    private double[] lastHiddenOutputs;
    private double[] lastOutputs;

    public NeuralNetwork(int numInputs, Integer numHidden, int numOutputs) {
        // Check for null values first
        if (numHidden == null) {
            throw new IllegalArgumentException("Number of hidden neurons cannot be null");
        }

        // Validations
        if (numInputs <= 0) {
            throw new IllegalArgumentException("Network must have at least one input neuron");
        }
        if (numHidden <= 0) {
            throw new IllegalArgumentException("Network must have at least one hidden neuron");
        }
        if (numOutputs <= 0) {
            throw new IllegalArgumentException("Network must have at least one output neuron");
        }

        inputLayer = new Neuron[numInputs];
        hiddenLayer = new Neuron[numHidden];
        outputLayer = new Neuron[numOutputs];

        // Create neurons
        for (int i = 0; i < numInputs; i++) {
            inputLayer[i] = new Neuron();
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

        // Set values for input neurons
        for (int i = 0; i < inputLayer.length; i++) {
            inputLayer[i].setValue(data[i]);
        }

        // Get values from input neurons
        double[] inputValues = new double[inputLayer.length];
        for (int i = 0; i < inputLayer.length; i++) {
            inputValues[i] = inputLayer[i].getValue();
        }

        // hidden layer activations
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activate(inputValues);
        }

        // Store hidden outputs for backpropagation
        this.lastHiddenOutputs = Arrays.copyOf(hiddenOutputs, hiddenOutputs.length);

        // output layer activations
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

    // Get weightS of the network
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

    public void train(List<Main.TrainingData> trainingData, double learningRate, int epochs) {
        System.out.println("Starting training for " + epochs + " epochs...");

        // epochs
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;

            for (Main.TrainingData data : trainingData) {
                
                double[] inputs = data.getInputs(); 
                double[] expectedOutputs = data.getExpectedOutputs(); 
                double[] actualOutputs = forward(inputs); 

                // error calculation
                double error = 0;
                for (int i = 0; i < actualOutputs.length; i++) {
                    // Square the difference between expected and actual
                    error += Math.pow(expectedOutputs[i] - actualOutputs[i], 2);
                }
                totalError += error;

                // Update the weights to make the network better
                backpropagate(expectedOutputs, learningRate);
            }
        }

        System.out.println("Training completed.");
    }

    private void backpropagate(double[] expectedOutputs, double learningRate) {
        
        // ---------------------------------------------
        double[] outputErrors = new double[outputLayer.length];
        double[] outputDeltas = new double[outputLayer.length];

        for (int i = 0; i < outputLayer.length; i++) {
            // How wrong was the prediction? (expected - actual)
            outputErrors[i] = expectedOutputs[i] - lastOutputs[i];

            // Calculate delta (error * derivative) for weight updates
            outputDeltas[i] = outputErrors[i] * outputLayer[i].sigmoidDerivative(lastOutputs[i]);
        }

        // Calculate errors in the hidden layer
        double[] hiddenErrors = new double[hiddenLayer.length];
        double[] hiddenDeltas = new double[hiddenLayer.length];

        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                hiddenErrors[i] += outputDeltas[j] * outputLayer[j].weights[i];
            }
            hiddenDeltas[i] = hiddenErrors[i] * hiddenLayer[i].sigmoidDerivative(lastHiddenOutputs[i]);
        }

        // Update weights in the output layer
        for (int i = 0; i < outputLayer.length; i++) {
            for (int j = 0; j < outputLayer[i].weights.length - 1; j++) {
                outputLayer[i].weights[j] += learningRate * outputDeltas[i] * lastHiddenOutputs[j];
            }
            // Update bias weight
            outputLayer[i].weights[outputLayer[i].weights.length - 1] += learningRate * outputDeltas[i];
        }

        // Update weights in the hidden layer
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].weights.length - 1; j++) {
                hiddenLayer[i].weights[j] += learningRate * hiddenDeltas[i] * lastInputs[j];
            }
            hiddenLayer[i].weights[hiddenLayer[i].weights.length - 1] += learningRate * hiddenDeltas[i];
        }
    }

    public void testNetwork(List<Main.TrainingData> testData) {
        // Counter for correct predictions
        int correct = 0;

        // Test each example
        for (Main.TrainingData data : testData) {
            double[] inputs = data.getInputs();
            double[] expectedOutputs = data.getExpectedOutputs();
            double[] actualOutputs = forward(inputs);

            int expectedClass = findMaxIndex(expectedOutputs);
            int actualClass = findMaxIndex(actualOutputs);

            System.out.print("Input: ");
            for (double input : inputs) {
                System.out.printf("%.2f ", input);
            }
            System.out.print("| Expected: " + expectedClass + " | Actual: " + actualClass);

            if (expectedClass == actualClass) {
                System.out.println(" correct");
                correct++;
            } else {
                System.out.println(" wrong");
            }
        }
    }

    public int findMaxIndex(double[] array) {
        int maxIndex = 0;
        double maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
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