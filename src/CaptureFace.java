import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

public class CaptureFace {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("❌ Error: Camera could not be opened.");
            return;
        }

        CascadeClassifier faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_alt.xml");
        Mat frame = new Mat();
        boolean faceCaptured = false;

        System.out.println("📸 Press any key on the camera window after face is captured to exit.");

        while (true) {
            if (!camera.read(frame)) {
                System.out.println("❌ Error: Cannot read frame from camera.");
                break;
            }

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(frame, faces);

            for (Rect rect : faces.toArray()) {
                Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);

                if (!faceCaptured) {
                    Mat face = new Mat(frame, rect);
                    Imgcodecs.imwrite("captures/captured_face.jpg", face);
                    System.out.println("✅ Face captured and saved as captured_face.jpg");
                    faceCaptured = true;
                }
            }

            HighGui.imshow("Face Capture - Press any key to exit", frame);

            // Wait 1ms, return >0 if key is pressed
            int key = HighGui.waitKey(1);
            if (key > 0 || faceCaptured) {
                break;
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.out.println("✅ Camera released and window closed.");
    }
}
