import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class RLCC {

    private static double[][] loadDataset(String filePath) {
        List<double[]> datasetList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] dataRow = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    dataRow[i] = Double.parseDouble(values[i]);
                }
                datasetList.add(dataRow);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Convert the list to a 2D array
        double[][] dataset = new double[datasetList.size()][];
        for (int i = 0; i < datasetList.size(); i++) {
            dataset[i] = datasetList.get(i);
        }
        return dataset;
    }

    public static void main(String[] args) {
        String filePath = "src/dataset/csv_util_uswtdb/uswtdb_v6_1_20231128_mn.csv";
        double[][] trainingDataset = loadDataset(filePath);
        // Continue with your RLCC algorithm...
    }
}
