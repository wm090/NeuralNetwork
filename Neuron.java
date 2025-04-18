/**
 * A single neuron in the neural network
 * Each neuron has weights for its connections to the previous layer
 */
public class Neuron {

    // Array to store the weights (including one extra weight for the bias)
    double weights[];

    /**
     * Create a new neuron with random weights
     * @param numInputs Number of inputs to this neuron
     */
    public Neuron(int numInputs) {
        // Create weights array (add 1 for bias weight)
        weights = new double[numInputs + 1];

        // Initialize with random weights between -1 and 1
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2) - 1;
        }
    }

    /**
     * Calculate the neuron's output for given inputs
     * @param inputs Input values
     * @return Output value (between 0 and 1)
     */
    public double activate(double[] inputs) {
        // Calculate the weighted sum of inputs
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }

        // Add the bias (extra weight)
        sum += weights[inputs.length];

        // Apply the activation function (sigmoid)
        return sigmoid(sum);
    }

    /**
     * Sigmoid activation function
     * Converts any number into a value between 0 and 1
     * @param x Input value
     * @return Output value between 0 and 1
     */
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Calculate the derivative of the sigmoid function
     * Used for backpropagation during training
     * @param output The output value (sigmoid(x))
     * @return The derivative value
     */
    public double sigmoidDerivative(double output) {
        // Derivative of sigmoid is: sigmoid(x) * (1 - sigmoid(x))
        return output * (1 - output);
    }

    /**
     * Convert the neuron's weights to a string
     */
    public String toString() {
        String s = "";
        for (int i = 0; i < weights.length; i++) {
            s += weights[i] + " ";
        }
        return s;
    }
}
