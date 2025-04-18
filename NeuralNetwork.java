import java.util.Arrays;

public class NeuralNetwork {

    Neuron[] inputLayer;
    Neuron[] hiddenLayer;
    Neuron[] outputLayer;
    
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
        // Step 1: Compute hidden layer activations
        double[] hiddenOutputs = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenOutputs[i] = hiddenLayer[i].activate(data);
        }

        // Step 2: Compute output layer activations
        double[] outputOutputs = new double[outputLayer.length];
        for (int i = 0; i < outputLayer.length; i++) {
            outputOutputs[i] = outputLayer[i].activate(hiddenOutputs);
        }

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

    @Override
    public String toString() {
        return "NeuralNetwork [inputLayer=" + Arrays.toString(inputLayer) + ", hiddenLayer="
                + Arrays.toString(hiddenLayer) + ", outputLayer=" + Arrays.toString(outputLayer) + "]";
    }
    
    
}