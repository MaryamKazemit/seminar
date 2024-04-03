import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class GMeanCalculator {

    private static final int POSITIVE = 1;
    private static final int NEGATIVE = 0;

    public static void main(String[] args) {
        String filePath = "src/dataset/csv_util_uswtdb/uswtdb_v6_1_20231128_mn.csv";
        List<int[]> data = loadBinaryData(filePath);
        int[][] confusionMatrix = constructConfusionMatrix(data);
        double gMean = calculateGMean(confusionMatrix);
        System.out.println("G-Mean: " + gMean);
    }

    private static List<int[]> loadBinaryData(String filePath) {
        List<int[]> data = new ArrayList<>();
        // Dummy loader - replace this with actual file reading and processing
        // int[] should contain {actual, predicted} labels.
        return data;
    }

    private static int[][] constructConfusionMatrix(List<int[]> data) {
        int[][] matrix = new int[2][2]; // For binary classification

        for (int[] record : data) {
            int actual = record[0];
            int predicted = record[1];
            matrix[actual][predicted]++;
        }

        return matrix;
    }

    private static double calculateGMean(int[][] matrix) {
        double sensitivity = (double) matrix[POSITIVE][POSITIVE] / (matrix[POSITIVE][POSITIVE] + matrix[POSITIVE][NEGATIVE]); // TP / (TP + FN)
        double specificity = (double) matrix[NEGATIVE][NEGATIVE] / (matrix[NEGATIVE][NEGATIVE] + matrix[NEGATIVE][POSITIVE]); // TN / (TN + FP)

        return Math.sqrt(sensitivity * specificity);
    }
}
