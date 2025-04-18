import java.util.List;

/**
 * Main class to demonstrate the neural network for traffic light classification
 */
public class Main {
    public static void main(String[] args) {
        // ===== STEP 1: Set up file paths =====
        // File with training examples (RGB values and traffic light classes)
        String trainingDataFile = "KW17_traindata_trafficlights_classification.csv";
        // File with initial weights for the neural network
        String weightsFile = "KW17_weights_trafficlights_classification_simplified.csv";
        // File to save the trained weights
        String outputWeightsFile = "trained_weights.csv";

        // ===== STEP 2: Load the training data =====
        System.out.println("Loading training data from " + trainingDataFile);
        List<TrainingData> trainingData = FileHandler.readTrainingData(trainingDataFile);
        System.out.println("Loaded " + trainingData.size() + " training examples");

        // ===== STEP 3: Create the neural network =====
        // Get network dimensions from the first training example
        TrainingData firstExample = trainingData.get(0);
        int numInputs = firstExample.getInputs().length;         // Number of input neurons (RGB values)
        int numOutputs = firstExample.getExpectedOutputs().length; // Number of output neurons (classes)
        int numHidden = 3;  // Number of hidden neurons (we'll use 3)

        // Create the neural network
        System.out.println("Creating neural network with:" +
                           "\n - " + numInputs + " input neurons (for RGB values)" +
                           "\n - " + numHidden + " hidden neurons" +
                           "\n - " + numOutputs + " output neurons (for traffic light classes)");
        NeuralNetwork nn = new NeuralNetwork(numInputs, numHidden, numOutputs);

        // ===== STEP 4: Load initial weights =====
        System.out.println("\nLoading initial weights from " + weightsFile);
        double[][][] weights = FileHandler.readWeights(weightsFile);
        if (weights != null) {
            nn.setWeights(weights[0], weights[1]);
            System.out.println("Weights loaded successfully");
        }

        // ===== STEP 5: Test the network before training =====
        System.out.println("\nTesting network BEFORE training:");
        testNetwork(nn, trainingData);

        // ===== STEP 6: Train the network =====
        System.out.println("\nTraining the network...");
        // Parameters: training data, learning rate (0.1), number of epochs (1000)
        nn.train(trainingData, 0.1, 1000);

        // ===== STEP 7: Test the network after training =====
        System.out.println("\nTesting network AFTER training:");
        testNetwork(nn, trainingData);

        // ===== STEP 8: Save the trained weights =====
        System.out.println("\nSaving trained weights to " + outputWeightsFile);
        double[][][] trainedWeights = nn.getWeights();
        FileHandler.saveWeights(outputWeightsFile, trainedWeights[0], trainedWeights[1]);

        System.out.println("\nDone!");

        double [][]hiddenWeights = {
            { 0.5, 0.3, 0.2 },
            { 0.3, 0.2, -0.5 }
        };
        double [][] outputWeights = {
            { 0.9, 0.2, -0.8 }
        };
        double[] data = { 1.0, 0.5 };
        NeuralNetwork nn2 = new NeuralNetwork(data.length, 2, 1);
        nn2.setWeights(hiddenWeights, outputWeights);
      
        double [] output = nn2.forward(data);
        System.out.println(output[0]);
    }
    

    /**
     * Test the neural network on the training data and print results
     * @param nn The neural network to test
     * @param testData The data to test on
     */
    private static void testNetwork(NeuralNetwork nn, List<TrainingData> testData) {
        // Counter for correct predictions
        int correct = 0;

        // Test each example
        for (TrainingData data : testData) {
            // Get the inputs (RGB values)
            double[] inputs = data.getInputs();

            // Get the expected outputs (correct traffic light class)
            double[] expectedOutputs = data.getExpectedOutputs();

            // Get the network's prediction
            double[] actualOutputs = nn.forward(inputs);

            // Find which class has the highest value (the network's prediction)
            int expectedClass = findMaxIndex(expectedOutputs);
            int actualClass = findMaxIndex(actualOutputs);

            // Print the result
            System.out.print("Input (RGB): ");
            for (double input : inputs) {
                System.out.printf("%.2f ", input);
            }
            System.out.print("| Expected: " + expectedClass + " | Actual: " + actualClass);

            // Check if the prediction was correct
            if (expectedClass == actualClass) {
                System.out.println(" ✓");  // Correct
                correct++;
            } else {
                System.out.println(" ✗");  // Wrong
            }
        }

        // Calculate and print the accuracy
        double accuracy = (double) correct / testData.size() * 100;
        System.out.printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, testData.size());
    }

    /**
     * Find the index of the maximum value in an array
     * This helps us determine which class the network predicted
     * @param array The array to search
     * @return The index of the maximum value
     */
    private static int findMaxIndex(double[] array) {
        // Start with the first element
        int maxIndex = 0;
        double maxValue = array[0];

        // Check each element
        for (int i = 1; i < array.length; i++) {
            // If this element is bigger, remember it
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        // Return the index of the biggest value
        return maxIndex;
    }

}
