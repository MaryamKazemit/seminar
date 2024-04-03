import java.util.HashMap;
import java.util.Map;

public class EvaluationMetrics {

    // Method to calculate G-mean
    public static double calculateGMean(int[][] confusionMatrix) {
        double product = 1.0;
        int numClasses = confusionMatrix.length;

        for (int i = 0; i < numClasses; i++) {
            double recall = (double) confusionMatrix[i][i] / sumRow(confusionMatrix, i);
            product *= recall;
        }

        return Math.pow(product, 1.0 / numClasses);
    }

    // Method to calculate Macro-F1
    public static double calculateMacroF1(int[][] confusionMatrix) {
        double totalF1 = 0.0;
        int numClasses = confusionMatrix.length;

        for (int i = 0; i < numClasses; i++) {
            double precision = (double) confusionMatrix[i][i] / sumColumn(confusionMatrix, i);
            double recall = (double) confusionMatrix[i][i] / sumRow(confusionMatrix, i);
            if (precision + recall != 0) { // Check to avoid division by zero
                double f1 = 2 * precision * recall / (precision + recall);
                totalF1 += f1;
            }
        }

        return totalF1 / numClasses;
    }

    // Helper method to sum a row in the confusion matrix
    private static int sumRow(int[][] matrix, int row) {
        int sum = 0;
        for (int i = 0; i < matrix[row].length; i++) {
            sum += matrix[row][i];
        }
        return sum;
    }

    // Helper method to sum a column in the confusion matrix
    private static int sumColumn(int[][] matrix, int col) {
        int sum = 0;
        for (int[] ints : matrix) {
            sum += ints[col];
        }
        return sum;
    }

    public static void main(String[] args) {
        // Example confusion matrix
        // Replace this with the actual confusion matrix from your classifier
        int[][] confusionMatrix = {
                {70, 30, 20},
                {10, 80, 10},
                {20, 10, 90}
        };

        double gMean = calculateGMean(confusionMatrix);
        double macroF1 = calculateMacroF1(confusionMatrix);

        System.out.println(STR."G-Mean: \{gMean}");
        System.out.println(STR."Macro-F1: \{macroF1}");
    }
}
