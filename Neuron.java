public class Neuron {

    double weights[];

    // For input neurons
    private double value;

    private boolean isInputNeuron;

    public Neuron(int numInputs) {
        this.isInputNeuron = false;
        weights = new double[numInputs + 1];

        for (int i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2) - 1;
        }
    }

    // Input neurons
    public Neuron() {
        this.isInputNeuron = true;
        this.value = 0.0;
        this.weights = null; // Input neurons don't have weights
    }

    public double activate(double[] inputs) {
        if (isInputNeuron) {
            return value;
        }
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        sum += weights[inputs.length];
        return sigmoid(sum);
        
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return this.value;
    }

    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoidDerivative(double output) {
        // Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        return output * (1 - output);
    }

    public String toString() {
        String s = "";
        for (int i = 0; i < weights.length; i++) {
            s += weights[i] + " ";
        }
        return s;
    }
}
