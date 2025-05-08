import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.io.File;
import java.util.Scanner;

public class CaptureFacesForTraining {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        // Ask user for name
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter user name: ");
        String userName = scanner.nextLine();
        scanner.close();

        // Create folder to store images
        String folderPath = "dataset/" + userName;
        File folder = new File(folderPath);
        if (!folder.exists()) {
            folder.mkdirs();
        }

        // Load classifier and open camera
        CascadeClassifier faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_alt.xml");
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("❌ Error: Cannot open camera.");
            return;
        }

        Mat frame = new Mat();
        int imageCount = 0;
        int maxImages = 30;

        System.out.println("📸 Capturing faces... Look at the camera.");

        while (imageCount < maxImages) {
            if (!camera.read(frame)) {
                System.out.println("❌ Failed to capture frame.");
                break;
            }

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            for (Rect rect : faces.toArray()) {
                // Draw rectangle
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);

                // Crop and save the face
                Mat face = new Mat(frame, rect);
                String filename = folderPath + "/" + (imageCount + 1) + ".jpg";
                Imgcodecs.imwrite(filename, face);
                imageCount++;

                System.out.println("✅ Saved: " + filename);

                // Avoid saving too many at once
                try {
                    Thread.sleep(300);  // Wait for 300ms before next capture
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                // Break inner loop if enough images
                if (imageCount >= maxImages) {
                    break;
                }
            }

            HighGui.imshow("Capturing Faces for " + userName, frame);
            if (HighGui.waitKey(1) >= 0) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.out.println("✅ Finished capturing " + imageCount + " images for " + userName);
    }
}
