import java.util.Arrays;

public class Neuron {

    double weights[];

    // For input neurons
    private double value;

    private boolean isInputNeuron;
    private boolean isBiasNeuron;

    // Regular neurons (hidden or output)
    public Neuron(int numInputs) {
        this.isInputNeuron = false;
        this.isBiasNeuron = false;
        weights = new double[numInputs];

        for (int i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2) - 1;
        }
    }

    // Input neurons
    public Neuron() {
        this.isInputNeuron = true;
        this.isBiasNeuron = false;
        this.value = 0.0;
        this.weights = null; // Input neurons don't have weights
    }

    // Bias neurons
    public Neuron(int numInputs, boolean isBias) {
        this.isInputNeuron = false;
        this.isBiasNeuron = isBias;
        this.value = 1.0;
        weights = new double[numInputs];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2) - 1;
        }
    }

    public double activate(double[] inputs) {
        if (isInputNeuron) {
            return value;
        }

        if (isBiasNeuron) {
            return 1.0;
        }

        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sigmoid(sum);
    }

    public void setValue(double value) {
        if (!isBiasNeuron) { // Don't change bias neuron value
            this.value = value;
        }
    }

    public double getValue() {
        if (isBiasNeuron) {
            return 1.0;
        }
        return this.value;
    }

    public boolean isBiasNeuron() {
        return this.isBiasNeuron;
    }

    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoidDerivative(double output) {
        // sigmoid Ableitung: sigmoid(x) * (1 - sigmoid(x))
        return output * (1 - output);
    }

    @Override
    public String toString() {
        return "Neuron [weights=" + Arrays.toString(weights) + ", value=" + value + ", isInputNeuron=" + isInputNeuron
                + ", isBiasNeuron=" + isBiasNeuron + "]";
    }
}
