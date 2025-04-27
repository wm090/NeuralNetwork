import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    Neuron[] inputLayer;
    Neuron inputBiasNeuron;
    Neuron[] hiddenLayer;
    Neuron hiddenBiasNeuron;
    Neuron[] outputLayer;

    // Cache for backpropagation
    private double[] lastInputs;
    private double[] lastHiddenOutputs;
    private double[] lastOutputs;

    public NeuralNetwork(int numInputs, Integer numHidden, int numOutputs) {
        // Validate network architecture
        if (numHidden == null || numInputs <= 0 || numHidden <= 0 || numOutputs <= 0) {
            throw new IllegalArgumentException("Invalid network architecture");
        }

        // Initialize layers
        inputLayer = new Neuron[numInputs];
        inputBiasNeuron = new Neuron(numHidden, true);
        hiddenLayer = new Neuron[numHidden];
        hiddenBiasNeuron = new Neuron(numOutputs, true);
        outputLayer = new Neuron[numOutputs];

        // neurons for each layer
        for (int i = 0; i < numInputs; i++){
            inputLayer[i] = new Neuron();
        }
        for (int i = 0; i < numHidden; i++) {
            hiddenLayer[i] = new Neuron(numInputs + 1);
        }
        for (int i = 0; i < numOutputs; i++) {
            outputLayer[i] = new Neuron(numHidden + 1);
        }
    }

    public double[] forward(double[] data) {
        // Store inputs for backpropagation
        lastInputs = Arrays.copyOf(data, data.length);

        // Set input values
        for (int i = 0; i < inputLayer.length; i++) {
            inputLayer[i].setValue(data[i]);
        }

        // Collect input values (including bias)
        double[] inputValues = new double[inputLayer.length + 1];
        for (int i = 0; i < inputLayer.length; i++) {
            inputValues[i] = inputLayer[i].getValue();
        }
        inputValues[inputLayer.length] = inputBiasNeuron.getValue();

        // Process hidden layer
        double[] hiddenOutputs = new double[hiddenLayer.length + 1];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activate(inputValues);
        }
        hiddenOutputs[hiddenLayer.length] = hiddenBiasNeuron.getValue();
        lastHiddenOutputs = Arrays.copyOf(hiddenOutputs, hiddenOutputs.length);

        // Process output layer
        double[] outputs = new double[outputLayer.length];
        for (int i = 0; i < outputLayer.length; i++) {
            outputs[i] = outputLayer[i].activate(hiddenOutputs);
        }
        lastOutputs = Arrays.copyOf(outputs, outputs.length);

        return outputs;
    }

    public void train(List<Main.TrainingData> trainingData, double learningRate, int epochs) {
        System.out.println("Starting training for " + epochs + " epochs...");

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Main.TrainingData data : trainingData) {
                forward(data.getInputs());
                backpropagate(data.getExpectedOutputs(), learningRate);
            }
        }

        System.out.println("Training completed.");
    }

    private void backpropagate(double[] expectedOutputs, double learningRate) {
        // Calculate output layer deltas
        double[] outputDeltas = new double[outputLayer.length];
        for (int i = 0; i < outputLayer.length; i++) {
            outputDeltas[i] = (expectedOutputs[i] - lastOutputs[i]) *
                              outputLayer[i].sigmoidDerivative(lastOutputs[i]);
        }

        // Calculate hidden layer deltas
        double[] hiddenDeltas = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            double error = 0;
            for (int j = 0; j < outputLayer.length; j++) {
                error += outputDeltas[j] * outputLayer[j].weights[i];
            }
            hiddenDeltas[i] = error * hiddenLayer[i].sigmoidDerivative(lastHiddenOutputs[i]);
        }

        // Update output layer weights
        for (int i = 0; i < outputLayer.length; i++) {
            for (int j = 0; j < outputLayer[i].weights.length; j++) {
                outputLayer[i].weights[j] += learningRate * outputDeltas[i] * lastHiddenOutputs[j];
            }
        }

        // Update hidden layer weights
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < hiddenLayer[i].weights.length; j++) {
                double input = j < lastInputs.length ? lastInputs[j] : 1.0; // 1.0 for bias
                hiddenLayer[i].weights[j] += learningRate * hiddenDeltas[i] * input;
            }
        }
    }

    public void testNetwork(List<Main.TrainingData> testData) {
        int correct = 0;

        for (Main.TrainingData data : testData) {
            double[] inputs = data.getInputs();
            double[] expectedOutputs = data.getExpectedOutputs();
            double[] actualOutputs = forward(inputs);

            int expectedClass = findMaxIndex(expectedOutputs);
            int actualClass = findMaxIndex(actualOutputs);

            System.out.print("Input: ");
            for (double input : inputs) System.out.printf("%.2f ", input);

            System.out.print("| Expected: " + expectedClass + " | Actual: " + actualClass);
            System.out.println(expectedClass == actualClass ? " correct" : " wrong");

            if (expectedClass == actualClass) correct++;
        }
    }

    public int findMaxIndex(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }

    public void setWeights(double[][] hiddenWeights, double[][] outputWeights) {
        // Set hidden layer weights
        for (int i = 0; i < hiddenLayer.length && i < hiddenWeights.length; i++) {
            for (int j = 0; j < hiddenLayer[i].weights.length && j < hiddenWeights[i].length; j++) {
                hiddenLayer[i].weights[j] = hiddenWeights[i][j];
            }
        }

        // Set output layer weights
        for (int i = 0; i < outputLayer.length && i < outputWeights.length; i++) {
            for (int j = 0; j < outputLayer[i].weights.length && j < outputWeights[i].length; j++) {
                outputLayer[i].weights[j] = outputWeights[i][j];
            }
        }
    }

    public double[][][] getWeights() {
        double[][] hiddenWeights = new double[hiddenLayer.length][];
        double[][] outputWeights = new double[outputLayer.length][];

        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenWeights[i] = Arrays.copyOf(hiddenLayer[i].weights, hiddenLayer[i].weights.length);
        }
        for (int i = 0; i < outputLayer.length; i++) {
            outputWeights[i] = Arrays.copyOf(outputLayer[i].weights, outputLayer[i].weights.length);
        }
        return new double[][][] { hiddenWeights, outputWeights };
    }
}