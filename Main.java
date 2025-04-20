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
        nn.testNetwork(trainingData);

        // ===== STEP 6: Train the network =====
        System.out.println("\nTraining the network...");
        // Parameters: training data, learning rate (0.1), number of epochs (1000)
        nn.train(trainingData, 0.1, 1000);

        // ===== STEP 7: Test the network after training =====
        System.out.println("\nTesting network AFTER training:");
        nn.testNetwork(trainingData);

        // ===== STEP 8: Save the trained weights =====
        System.out.println("\nSaving trained weights to " + outputWeightsFile);
        double[][][] trainedWeights = nn.getWeights();
        FileHandler.saveWeights(outputWeightsFile, trainedWeights[0], trainedWeights[1]);

        System.out.println("\nDone!");

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
}
