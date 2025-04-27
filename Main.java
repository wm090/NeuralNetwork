import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        // File paths and network parameters
        String trainingDataFile = "KW17_traindata_trafficlights_classification.csv";
        String weightsFile = "KW17_weights_trafficlights_classification_simplified.csv";
        String outputWeightsFile = "trained_weights.csv";
        int numHidden = 3;  // Hidden neurons

        // Load training data and create network
        List<TrainingData> trainingData = loadTrainingData(trainingDataFile);
        int numInputs = trainingData.get(0).getInputs().length;
        int numOutputs = trainingData.get(0).getExpectedOutputs().length;

        System.out.println("Neural network: " + numInputs + " inputs, "
                          + numHidden + " hidden, " + numOutputs + " outputs");
        NeuralNetwork nn = new NeuralNetwork(numInputs, numHidden, numOutputs);

        // Load weights, test, train, and save
        double[][][] weights = loadWeights(weightsFile);
        if (weights != null) {
            nn.setWeights(weights[0], weights[1]);
        }

        System.out.println("\nBefore training - test:");
        nn.testNetwork(trainingData);

        System.out.println("\nTraining...");
        nn.train(trainingData, 0.1, 1000);

        System.out.println("\nAfter training:");
        nn.testNetwork(trainingData);

        saveWeights(nn.getWeights(), outputWeightsFile);

        System.out.println("\nDone!");

        System.out.println("--Simple example with manual weights--");
        testSimpleNetwork();

    }

    // Load training data from CSV file
    private static List<TrainingData> loadTrainingData(String filename) throws Exception {
        System.out.println("Loading training data from " + filename);
        List<TrainingData> trainingData = new ArrayList<>();

        // Open file and read line by line
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;

        while ((line = reader.readLine()) != null) {
            // Split line into input and output parts
            String[] parts = line.split("\\s+");
            if (parts.length < 2) continue;

            // Convert input string to double array
            String[] inputStrings = parts[0].split(";");
            double[] inputs = new double[inputStrings.length];
            for (int i = 0; i < inputs.length; i++) {
                inputs[i] = Double.parseDouble(inputStrings[i].trim());
            }

            // Convert output string to double array
            String[] outputStrings = parts[1].split(";");
            double[] outputs = new double[outputStrings.length];
            for (int i = 0; i < outputs.length; i++) {
                outputs[i] = Double.parseDouble(outputStrings[i].trim());
            }

            // Add to training data list
            trainingData.add(new TrainingData(inputs, outputs));
        }
        reader.close();

        System.out.println("Trainingsdata loaded");
        return trainingData;
    }

    // Load weights from CSV file
    private static double[][][] loadWeights(String filename) throws Exception {
        System.out.println("Loading weights from " + filename);

        BufferedReader reader = new BufferedReader(new FileReader(filename));

        // Read network size
        String[] dimensions = reader.readLine().split(";");
        int numInputs = Integer.parseInt(dimensions[1]);
        int numHidden = Integer.parseInt(dimensions[2]);
        int numOutputs = Integer.parseInt(dimensions[3]);

        // Create weight arrays
        double[][] hiddenWeights = new double[numHidden][numInputs + 1]; // +1 for bias
        double[][] outputWeights = new double[numOutputs][numHidden + 1]; // +1 for bias

        // Read hidden layer weights
        for (int i = 0; i < numHidden; i++) {
            String[] values = reader.readLine().split(";");
            for (int j = 0; j < numInputs && j < values.length; j++) {
                if (!values[j].trim().isEmpty()) {
                    hiddenWeights[i][j] = Double.parseDouble(values[j].trim());
                }
            }
        }

        // Read hidden bias weights
        String[] biasValues = reader.readLine().split(";");
        for (int i = 0; i < numHidden && i < biasValues.length; i++) {
            if (!biasValues[i].trim().isEmpty()) {
                hiddenWeights[i][numInputs] = Double.parseDouble(biasValues[i].trim());
            }
        }

        reader.readLine(); // Skip separator

        // Read output layer weights
        for (int i = 0; i < numOutputs; i++) {
            String[] values = reader.readLine().split(";");
            for (int j = 0; j < numHidden && j < values.length; j++) {
                if (!values[j].trim().isEmpty()) {
                    outputWeights[i][j] = Double.parseDouble(values[j].trim());
                }
            }
        }

        // Read output bias weights
        String outputBiasLine = reader.readLine();
        if (outputBiasLine != null) {
            String[] outputBias = outputBiasLine.split(";");
            for (int i = 0; i < numOutputs && i < outputBias.length; i++) {
                if (!outputBias[i].trim().isEmpty()) {
                    outputWeights[i][numHidden] = Double.parseDouble(outputBias[i].trim());
                }
            }
        }

        reader.close();
        System.out.println("Weights Loaded");

        return new double[][][] { hiddenWeights, outputWeights };
    }

    // Save weights to CSV file
    private static void saveWeights(double[][][] weights, String filename) throws Exception {
        System.out.println("Saving weights to " + filename);

        // Get dimensions
        double[][] hiddenWeights = weights[0];
        double[][] outputWeights = weights[1];
        int numInputs = hiddenWeights[0].length - 1;
        int numHidden = hiddenWeights.length;
        int numOutputs = outputWeights.length;

        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));

        // Write header
        writer.write("layers;" + numInputs + ";" + numHidden + ";" + numOutputs);
        writer.newLine();

        // Write hidden weights
        for (int i = 0; i < numHidden; i++) {
            StringBuilder line = new StringBuilder();
            for (int j = 0; j < numInputs; j++) {
                line.append(hiddenWeights[i][j]).append(";");
            }
            writer.write(line.toString());
            writer.newLine();
        }

        // Write hidden bias
        StringBuilder biasLine = new StringBuilder();
        for (int i = 0; i < numHidden; i++) {
            biasLine.append(hiddenWeights[i][numInputs]);
            if (i < numHidden - 1) biasLine.append(";");
        }
        writer.write(biasLine.toString());
        writer.newLine();

        // Write separator
        writer.write(";;;");
        writer.newLine();

        // Write output weights
        for (int i = 0; i < numOutputs; i++) {
            StringBuilder line = new StringBuilder();
            for (int j = 0; j < numHidden; j++) {
                line.append(outputWeights[i][j]).append(";");
            }
            writer.write(line.toString());
            writer.newLine();
        }

        // Write output bias
        StringBuilder outBiasLine = new StringBuilder();
        for (int i = 0; i < numOutputs; i++) {
            outBiasLine.append(outputWeights[i][numHidden]);
            if (i < numOutputs - 1) outBiasLine.append(";");
        }
        writer.write(outBiasLine.toString());
        writer.newLine();

        writer.close();
        System.out.println("Weights saved");
    }

    // Test a simple network with manual weights
    private static void testSimpleNetwork() throws Exception {
        double[] data = { 1.0, 0.5};

        // Create a simple network with 2 inputs, 2 hidden neurons, and 1 output
        NeuralNetwork testNetwork = new NeuralNetwork(data.length, 2, 1);

        // Set weights and test
        testNetwork.setWeights(
            new double[][] {{ 0.5, 0.3, 0.2 }, { 0.3, 0.2, -0.5 }},  // Hidden weights
            new double[][] {{ 0.9, 0.2, -0.8 }}                      // Output weights
        );
        // Run forward pass with test input
        double[] output = testNetwork.forward(data);
        System.out.println("Test output: " + output[0]);
    }

    // Class to hold training data
    static class TrainingData {
        private double[] inputs;
        private double[] expectedOutputs;

        public TrainingData(double[] inputs, double[] expectedOutputs) {
            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
        }

        public double[] getInputs() {
            return inputs;
        }

        public double[] getExpectedOutputs() {
            return expectedOutputs;
        }
    }
}
