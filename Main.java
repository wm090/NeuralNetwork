import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {

        // training data
        String trainingDataFile = "KW17_traindata_trafficlights_classification.csv";
        String weightsFile = "KW17_weights_trafficlights_classification_simplified.csv";
        String outputWeightsFile = "trained_weights.csv";

        System.out.println("Loading training data " + trainingDataFile);
        List<TrainingData> trainingData = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(trainingDataFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\\s+");
                if (parts.length >= 2) {
                    String[] inputParts = parts[0].split(";");
                    double[] inputs = new double[inputParts.length];
                    for (int i = 0; i < inputParts.length; i++) {
                        inputs[i] = Double.parseDouble(inputParts[i]);
                    }

                    String[] outputParts = parts[1].split(";");
                    double[] outputs = new double[outputParts.length];
                    for (int i = 0; i < outputParts.length; i++) {
                        outputs[i] = Double.parseDouble(outputParts[i]);
                    }

                    trainingData.add(new TrainingData(inputs, outputs));
                }
            }
        } catch (IOException e) {
            System.out.println("Error reading training data: " + e.getMessage());
        }

        System.out.println("Loaded " + trainingData.size() + " training examples");

        // neural network
        TrainingData InputExample = trainingData.get(0);
        int numInputs = InputExample.getInputs().length;         //input neurons
        int numOutputs = InputExample.getExpectedOutputs().length; // output neurons
        int numHidden = 3;  // Hidden neurons, 3 as an example
        NeuralNetwork nn = new NeuralNetwork(numInputs, numHidden, numOutputs);

        System.out.printf("inputsize: " + numInputs);
        System.out.printf("\nhiddenneuronsize: " + numHidden);
        System.out.printf("\noutputsize: " + numOutputs);
        System.out.println();

        // Load weights
        System.out.println("Loading weights " + weightsFile);
        double[][][] weights = null;

        try (BufferedReader reader = new BufferedReader(new FileReader(weightsFile))) {
            // Read dimensions
            String[] dimensions = reader.readLine().split(";");
            int wNumInputs = Integer.parseInt(dimensions[1]);
            int wNumHidden = Integer.parseInt(dimensions[2]);
            int wNumOutputs = Integer.parseInt(dimensions[3]);

            // Read hidden weights
            double[][] hiddenWeights = new double[wNumHidden][wNumInputs + 1];
            for (int i = 0; i < wNumHidden; i++) {
                String[] values = reader.readLine().split(";");
                for (int j = 0; j < wNumInputs + 1 && j < values.length; j++) {
                    String value = values[j].trim();
                    if (!value.isEmpty()) {
                        hiddenWeights[i][j] = Double.parseDouble(value);
                    }
                }
            }

            // Skip separator
            reader.readLine();

            // Read output weights
            double[][] outputWeights = new double[wNumOutputs][wNumHidden + 1];
            for (int i = 0; i < wNumOutputs; i++) {
                String[] values = reader.readLine().split(";");
                for (int j = 0; j < wNumHidden + 1 && j < values.length; j++) {
                    String value = values[j].trim();
                    if (!value.isEmpty()) {
                        outputWeights[i][j] = Double.parseDouble(value);
                    }
                }
            }

            weights = new double[][][] { hiddenWeights, outputWeights };
        } catch (IOException e) {
            System.out.println("Error reading weights: " + e.getMessage());
        }

        if (weights != null) {
            nn.setWeights(weights[0], weights[1]);
            System.out.println("Weights loaded");
        }

        // Test the network before training
        System.out.println("\nBefore training - test:");
        nn.testNetwork(trainingData);

        // Train the network
        System.out.println("\nTraining...");
        nn.train(trainingData, 0.1, 1000);

        // Test the network after training
        System.out.println("\nAfter training:");
        nn.testNetwork(trainingData);

        // Save the trained weights
        System.out.println("\nSaving trained weights to " + outputWeightsFile);
        double[][][] trainedWeights = nn.getWeights();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputWeightsFile))) {
            double[][] hiddenWeights = trainedWeights[0];
            double[][] outputWeights = trainedWeights[1];

            // Calculate dimensions
            int weightsNumInputs = hiddenWeights[0].length - 1; // -1 to exclude bias
            int weightsNumHidden = hiddenWeights.length;
            int weightsNumOutputs = outputWeights.length;

            // Write header
            writer.write("layers;" + weightsNumInputs + ";" + weightsNumHidden + ";" + weightsNumOutputs);
            writer.newLine();

            // Write hidden weights
            for (double[] neuronWeights : hiddenWeights) {
                for (int j = 0; j < neuronWeights.length; j++) {
                    writer.write(String.valueOf(neuronWeights[j]));
                    if (j < neuronWeights.length - 1) writer.write(";");
                }
                writer.newLine();
            }

            // Write separator
            writer.write(";;;");
            writer.newLine();

            // Write output weights
            for (double[] neuronWeights : outputWeights) {
                for (int j = 0; j < neuronWeights.length; j++) {
                    writer.write(String.valueOf(neuronWeights[j]));
                    if (j < neuronWeights.length - 1) writer.write(";");
                }
                writer.newLine();
            }

            System.out.println("Weights saved");
        } catch (IOException e) {
            System.out.println("Error saving weights: " + e.getMessage());
        }

        System.out.println("\nDone!");

        // different architectures
        System.out.println("-----------------------------");
        double [][]hiddenWeights = {
            { 0.5, 0.3, 0.2 },
            { 0.3, 0.2, -0.5 }
        };
        double [][] outputWeights = {
            { 0.9, 0.2, -0.8 }
        };
        double[] data = { 1.0, 0.5 };
        NeuralNetwork nn2 = new NeuralNetwork(data.length, 2, 1);
        NeuralNetwork nn3 = new NeuralNetwork(data.length, 2, 2);
        NeuralNetwork nn4 = new NeuralNetwork(data.length, 5, 1);
        //NeuralNetwork nn5 = new NeuralNetwork(data.length, 0, 1);
        //NeuralNetwork nn6 = new NeuralNetwork(data.length, null, 1);

        nn2.setWeights(hiddenWeights, outputWeights);

        double [] output = nn2.forward(data);
        System.out.println(output[0]);
    }

    //inner class to hold training data (inputs and expected outputs)
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
