import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple file handler for loading and saving neural network data
 */
public class FileHandler {

    /**
     * Reads training data from a CSV file
     * @param filePath Path to the CSV file
     * @return A list of TrainingData objects
     */
    public static List<TrainingData> readTrainingData(String filePath) {
        // Create an empty list to store our training examples
        List<TrainingData> trainingDataList = new ArrayList<>();

        try {
            // Open the file for reading
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;

            // Read the file line by line
            while ((line = reader.readLine()) != null) {
                // Each line has inputs and outputs separated by tab or spaces
                String[] parts = line.split("\\s+");

                if (parts.length >= 2) {
                    // PART 1: Get the inputs (RGB values)
                    // Split by semicolon to get individual values
                    String[] inputParts = parts[0].split(";");
                    double[] inputs = new double[inputParts.length];

                    // Convert each string to a number
                    for (int i = 0; i < inputParts.length; i++) {
                        inputs[i] = Double.parseDouble(inputParts[i]);
                    }

                    // PART 2: Get the expected outputs (traffic light class)
                    String[] outputParts = parts[1].split(";");
                    double[] expectedOutputs = new double[outputParts.length];

                    // Convert each string to a number
                    for (int i = 0; i < outputParts.length; i++) {
                        expectedOutputs[i] = Double.parseDouble(outputParts[i]);
                    }

                    // Create a new training example and add it to our list
                    trainingDataList.add(new TrainingData(inputs, expectedOutputs));
                }
            }

            // Close the file when done
            reader.close();

        } catch (IOException e) {
            // If something goes wrong, print an error message
            System.out.println("Error reading file: " + e.getMessage());
        }

        return trainingDataList;
    }

    /**
     * Reads network weights from a CSV file
     * @param filePath Path to the CSV file
     * @return An array of weight matrices for each layer
     */
    public static double[][][] readWeights(String filePath) {
        try {
            // Open the file for reading
            BufferedReader reader = new BufferedReader(new FileReader(filePath));

            // Read the first line which contains network dimensions
            String firstLine = reader.readLine();
            String[] parts = firstLine.split(";");

            // Check if the file format is correct
            if (parts.length < 3 || !parts[0].equals("layers")) {
                System.out.println("The weights file has an invalid format");
                reader.close();
                return null;
            }

            // Get the network dimensions
            int numInputs = Integer.parseInt(parts[1]);
            int numHidden = Integer.parseInt(parts[2]);
            int numOutputs = Integer.parseInt(parts[3]);

            // Create arrays to store the weights
            // +1 for bias weight in each neuron
            double[][] hiddenWeights = new double[numHidden][numInputs + 1];
            double[][] outputWeights = new double[numOutputs][numHidden + 1];

            // Read hidden layer weights (one line per neuron)
            for (int i = 0; i < numHidden; i++) {
                String line = reader.readLine();
                String[] values = line.split(";");

                for (int j = 0; j < numInputs + 1 && j < values.length; j++) {
                    String value = values[j].trim();
                    if (!value.isEmpty()) {
                        hiddenWeights[i][j] = Double.parseDouble(value);
                    }
                }
            }

            // Skip the separator line
            reader.readLine();

            // Read output layer weights (one line per neuron)
            for (int i = 0; i < numOutputs; i++) {
                String line = reader.readLine();
                if (line != null) {
                    String[] values = line.split(";");

                    for (int j = 0; j < numHidden + 1 && j < values.length; j++) {
                        String value = values[j].trim();
                        if (!value.isEmpty()) {
                            outputWeights[i][j] = Double.parseDouble(value);
                        }
                    }
                }
            }

            // Close the file
            reader.close();

            // Return both weight arrays
            return new double[][][] { hiddenWeights, outputWeights };

        } catch (Exception e) {
            System.out.println("Error reading weights: " + e.getMessage());
            return null;
        }
    }

    /**
     * Saves network weights to a CSV file
     * @param filePath Path to save the CSV file
     * @param hiddenWeights Weights for the hidden layer
     * @param outputWeights Weights for the output layer
     */
    public static void saveWeights(String filePath, double[][] hiddenWeights, double[][] outputWeights) {
        try {
            // Open the file for writing
            BufferedWriter writer = new BufferedWriter(new FileWriter(filePath));

            // Calculate network dimensions
            int numInputs = hiddenWeights[0].length - 1; // -1 to exclude bias
            int numHidden = hiddenWeights.length;
            int numOutputs = outputWeights.length;

            // Write the header line with network dimensions
            writer.write("layers;" + numInputs + ";" + numHidden + ";" + numOutputs);
            writer.newLine();

            // Write hidden layer weights (one line per neuron)
            for (int i = 0; i < hiddenWeights.length; i++) {
                StringBuilder line = new StringBuilder();
                for (int j = 0; j < hiddenWeights[i].length; j++) {
                    line.append(hiddenWeights[i][j]).append("; ");
                }
                writer.write(line.toString());
                writer.newLine();
            }

            // Write separator line
            writer.write(";;;");
            writer.newLine();

            // Write output layer weights (one line per neuron)
            for (int i = 0; i < outputWeights.length; i++) {
                StringBuilder line = new StringBuilder();
                for (int j = 0; j < outputWeights[i].length; j++) {
                    line.append(outputWeights[i][j]).append("; ");
                }
                writer.write(line.toString());
                writer.newLine();
            }

            // Close the file
            writer.close();

            System.out.println("Weights saved successfully to " + filePath);

        } catch (IOException e) {
            System.out.println("Error saving weights: " + e.getMessage());
        }
    }
}
