public class Neuron {
    
    double weights[];

    public Neuron(int numInputs) {
        weights = new double[numInputs + 1];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2) - 1;
        }
    }

    public double activate(double[] inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        sum += weights[inputs.length];
        return sigmoid(sum);
    }

    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public String toString() {
        String s = "";
        for (int i = 0; i < weights.length; i++) {
            s += weights[i] + " ";
        }
        return s;
    }
}
