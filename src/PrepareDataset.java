import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.*;
import java.nio.file.*;

public class PrepareDataset {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws IOException {
        String datasetPath = "dataset";
        File datasetDir = new File(datasetPath);
        File[] userDirs = datasetDir.listFiles(File::isDirectory);

        BufferedWriter writer = new BufferedWriter(new FileWriter("faces_dataset.csv"));
        // Format: label,pixel1,pixel2,...,pixel10000

        for (File userDir : userDirs) {
            String label = userDir.getName();
            File[] images = userDir.listFiles((d, name) -> name.endsWith(".jpg"));

            if (images == null) continue;

            for (File imgFile : images) {
                Mat img = Imgcodecs.imread(imgFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                Imgproc.resize(img, img, new Size(100, 100));

                StringBuilder line = new StringBuilder(label);
                for (int row = 0; row < img.rows(); row++) {
                    for (int col = 0; col < img.cols(); col++) {
                        double[] pixel = img.get(row, col);
                        line.append(",").append((int) pixel[0]);
                    }
                }

                writer.write(line.toString());
                writer.newLine();
            }
        }

        writer.close();
        System.out.println("✅ Dataset prepared and saved as faces_dataset.csv");
    }
}
