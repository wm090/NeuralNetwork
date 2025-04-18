public class Main {
    public static void main(String[] args) {
        //NeuralNetwork nn = new NeuralNetwork(3,2,1);
        double[] data = {1.0, 0.5,0.9, 0.2};

        NeuralNetwork nn = new NeuralNetwork(data.length,3,1);

        // Example weights for hidden layer (2 neurons, each with 3 weights: 2 inputs + 1 bias)
        double[][] hiddenWeights = {
            {0.5, 0.3, 0.2},
            {0.3, 0.2, -0.5}
        };
        // Example weights for output layer (1 neuron, with 3 weights: 2 hidden outputs + 1 bias)
        double[][] outputWeights = {
            {0.9, 0.2, -0.8}
        };
        //nn.setWeights(hiddenWeights, outputWeights);
        double[] output = nn.forward(data);
        System.out.println(output[0]);
    }

}
