import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class FaceRecognizer {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // Load the dataset into memory
    public static List<LabeledImage> loadDataset(String filePath) throws IOException {
        List<LabeledImage> data = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            String label = parts[0];
            double[] pixels = new double[parts.length - 1];
            for (int i = 1; i < parts.length; i++) {
                pixels[i - 1] = Double.parseDouble(parts[i]);
            }
            data.add(new LabeledImage(label, pixels));
        }
        reader.close();
        return data;
    }

    // Euclidean distance
    public static double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    // KNN predict
    public static String predict(List<LabeledImage> dataset, double[] inputVector, int k) {
        PriorityQueue<LabeledImageDistance> pq = new PriorityQueue<>(Comparator.comparingDouble(o -> o.distance));
        for (LabeledImage img : dataset) {
            double dist = distance(inputVector, img.pixels);
            pq.offer(new LabeledImageDistance(img.label, dist));
        }

        Map<String, Integer> voteMap = new HashMap<>();
        for (int i = 0; i < k && !pq.isEmpty(); i++) {
            String label = pq.poll().label;
            voteMap.put(label, voteMap.getOrDefault(label, 0) + 1);
        }

        return Collections.max(voteMap.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // ✅ Method to mark attendance safely
    public static synchronized void markAttendance(String name) {
        String fileName = "attendance.csv";
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, true))) {
            writer.write(name + "," + timestamp);
            writer.newLine();
            System.out.println("✅ Attendance marked for " + name + " at " + timestamp);
        } catch (IOException e) {
            System.out.println("❌ Error writing attendance: " + e.getMessage());
        }
    }

    public static void main(String[] args) throws IOException {
        List<LabeledImage> dataset = loadDataset("faces_dataset.csv");

        CascadeClassifier faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_alt.xml");
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("❌ Cannot open camera.");
            return;
        }

        Mat frame = new Mat();
        System.out.println("🎥 Starting face recognition...");

        // ✅ To avoid duplicate attendance in the same session
        Set<String> markedNames = new HashSet<>();

        while (true) {
            camera.read(frame);
            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            for (Rect rect : faces.toArray()) {
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(255, 0, 0), 2);

                Mat face = new Mat(frame, rect);
                Imgproc.cvtColor(face, face, Imgproc.COLOR_BGR2GRAY);
                Imgproc.resize(face, face, new Size(100, 100));

                double[] inputVector = new double[10000];
                int index = 0;
                for (int row = 0; row < face.rows(); row++) {
                    for (int col = 0; col < face.cols(); col++) {
                        inputVector[index++] = face.get(row, col)[0];
                    }
                }

                String predictedLabel = predict(dataset, inputVector, 3);
                Imgproc.putText(frame, predictedLabel, new Point(rect.x, rect.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 255, 0), 2);

                // ✅ Mark attendance only once per person
                if (!markedNames.contains(predictedLabel)) {
                    markAttendance(predictedLabel);
                    markedNames.add(predictedLabel);
                }
            }

            HighGui.imshow("Face Recognition", frame);
            if (HighGui.waitKey(1) == 27) { // Press 'Esc' to quit
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}

// Helper class to store labeled images
class LabeledImage {
    String label;
    double[] pixels;

    public LabeledImage(String label, double[] pixels) {
        this.label = label;
        this.pixels = pixels;
    }
}

// Helper class for distances
class LabeledImageDistance {
    String label;
    double distance;

    public LabeledImageDistance(String label, double distance) {
        this.label = label;
        this.distance = distance;
    }
}
