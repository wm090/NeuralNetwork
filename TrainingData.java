/**
 * Class to hold a single training example
 * Each example has inputs (RGB values) and expected outputs (traffic light class)
 */
public class TrainingData {
    // The input values (RGB values for traffic light)
    private double[] inputs;

    // The expected output values (which traffic light class it is)
    private double[] expectedOutputs;

    /**
     * Create a new training example
     * @param inputs The input values (RGB values)
     * @param expectedOutputs The expected output values (traffic light class)
     */
    public TrainingData(double[] inputs, double[] expectedOutputs) {
        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;
    }

    /**
     * Get the input values
     * @return Array of input values
     */
    public double[] getInputs() {
        return inputs;
    }

    /**
     * Get the expected output values
     * @return Array of expected output values
     */
    public double[] getExpectedOutputs() {
        return expectedOutputs;
    }

    /**
     * Convert the training example to a readable string
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        // Add the input values
        sb.append("Inputs (RGB): ");
        for (double input : inputs) {
            sb.append(input).append(" ");
        }

        // Add the expected output values
        sb.append("\nExpected Class: ");
        for (double output : expectedOutputs) {
            sb.append(output).append(" ");
        }

        return sb.toString();
    }
}
