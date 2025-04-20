/**
 * A single neuron in the neural network
 * Each neuron has weights for its connections to the previous layer
 * Input neurons are special and just store a value
 */
public class Neuron {

    // Array to store the weights (including one extra weight for the bias)
    double weights[];

    // For input neurons: store the input value
    private double value;

    // Flag to identify if this is an input neuron
    private boolean isInputNeuron;

    /**
     * Create a new neuron with random weights (for hidden and output neurons)
     * @param numInputs Number of inputs to this neuron
     */
    public Neuron(int numInputs) {
        this.isInputNeuron = false;

        // Create weights array (add 1 for bias weight)
        weights = new double[numInputs + 1];

        // Initialize with random weights between -1 and 1
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2) - 1;
        }
    }

    /**
     * Create an input neuron (no weights, just stores a value)
     */
    public Neuron() {
        this.isInputNeuron = true;
        this.value = 0.0;
        this.weights = null; // Input neurons don't have weights
    }

    /**
     * Calculate the neuron's output for given inputs
     * @param inputs Input values
     * @return Output value (between 0 and 1)
     */
    public double activate(double[] inputs) {
        // For input neurons, just return the stored value
        if (isInputNeuron) {
            return value;
        }

        // For hidden and output neurons, calculate weighted sum
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
     * Set the value for an input neuron
     * @param value The input value to store
     */
    public void setValue(double value) {
        this.value = value;
    }

    /**
     * Get the current value of the neuron
     * @return The current value
     */
    public double getValue() {
        return this.value;
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
