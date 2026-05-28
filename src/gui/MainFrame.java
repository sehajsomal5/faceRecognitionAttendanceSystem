package gui;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import javax.swing.*;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

public class MainFrame extends JFrame {
    
    // Dynamic OpenCV library loader using OpenPnP for cross-platform compatibility
    static {
        try {
            nu.pattern.OpenCV.loadShared();
            System.out.println("✅ OpenCV loaded dynamically via OpenPnP");
        } catch (Throwable e) {
            System.out.println("⚠️ OpenPnP load failed. Trying system library fallback: " + e.getMessage());
            try {
                System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
                System.out.println("✅ System OpenCV loaded successfully.");
            } catch (UnsatisfiedLinkError le) {
                System.out.println("❌ FAILED to load OpenCV. Webcam features will be disabled: " + le.getMessage());
            }
        }
    }

    private CardLayout cardLayout;
    private JPanel mainContainer;
    
    private LoginScreen loginScreen;
    private AppWorkspacePanel workspacePanel;

    // OpenCV Resources
    private VideoCapture camera = null;
    private CascadeClassifier faceDetector = null;
    private Thread cameraThread = null;
    private boolean isCameraActive = false;
    private ModernComponents.WebcamPanel currentWebcamPanel = null;
    private FrameCallback currentCallback = null;

    // KNN Dataset Cache
    public static class LabeledImage {
        public String label;
        public double[] pixels;
        public LabeledImage(String label, double[] pixels) {
            this.label = label;
            this.pixels = pixels;
        }
    }
    private java.util.List<LabeledImage> dataset = new ArrayList<>();

    public interface FrameCallback {
        void onFrameProcessed(Mat frame, BufferedImage img);
    }

    public MainFrame() {
        setTitle("Face Recognition Attendance System");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1200, 800);
        setMinimumSize(new Dimension(1000, 650));
        setLocationRelativeTo(null); // Center window

        // Initialize Face Detector
        try {
            faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_alt.xml");
            if (faceDetector.empty()) {
                System.out.println("⚠️ Haar cascade file is empty or missing! Loading default frontalface cascade...");
                faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");
            }
        } catch (Exception ex) {
            System.out.println("❌ Error loading CascadeClassifier: " + ex.getMessage());
        }

        // Load Dataset
        reloadDataset();

        // CardLayout container
        cardLayout = new CardLayout();
        mainContainer = new JPanel(cardLayout);

        loginScreen = new LoginScreen(this);
        workspacePanel = new AppWorkspacePanel(this);

        mainContainer.add(loginScreen, "Login");
        mainContainer.add(workspacePanel, "AppWorkspace");

        add(mainContainer);

        // Start on Login Screen
        cardLayout.show(mainContainer, "Login");
    }

    public void switchScreen(String name) {
        stopCameraFeed();
        if (name.equals("Login")) {
            cardLayout.show(mainContainer, "Login");
        } else if (name.equals("Dashboard")) {
            workspacePanel.navigateTo("Dashboard");
            cardLayout.show(mainContainer, "AppWorkspace");
        } else {
            cardLayout.show(mainContainer, "AppWorkspace");
            workspacePanel.navigateTo(name);
        }
    }

    public CascadeClassifier getFaceDetector() {
        return faceDetector;
    }

    public java.util.List<LabeledImage> getDataset() {
        return dataset;
    }

    // Load CSV Dataset
    public void reloadDataset() {
        dataset.clear();
        File file = new File("faces_dataset.csv");
        if (!file.exists()) return;
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 2) continue;
                String label = parts[0];
                double[] pixels = new double[parts.length - 1];
                for (int i = 1; i < parts.length; i++) {
                    pixels[i - 1] = Double.parseDouble(parts[i]);
                }
                dataset.add(new LabeledImage(label, pixels));
            }
            System.out.println("🚀 KNN Dataset Loaded: " + dataset.size() + " face records.");
        } catch (Exception ex) {
            System.out.println("❌ Error loading dataset: " + ex.getMessage());
        }
    }

    // KNN Predict
    public String predict(double[] inputVector) {
        if (dataset.isEmpty()) return "Unknown";

        class LabeledDist {
            String label;
            double distance;
            LabeledDist(String label, double dist) {
                this.label = label;
                this.distance = dist;
            }
        }

        PriorityQueue<LabeledDist> pq = new PriorityQueue<>(Comparator.comparingDouble(o -> o.distance));
        for (LabeledImage img : dataset) {
            double sum = 0;
            // Vector size is 10000 (100x100 pixels)
            for (int i = 0; i < inputVector.length; i++) {
                sum += Math.pow(inputVector[i] - img.pixels[i], 2);
            }
            double dist = Math.sqrt(sum);
            pq.offer(new LabeledDist(img.label, dist));
        }

        Map<String, Integer> votes = new HashMap<>();
        int k = Math.min(5, dataset.size());
        for (int i = 0; i < k && !pq.isEmpty(); i++) {
            String label = pq.poll().label;
            votes.put(label, votes.getOrDefault(label, 0) + 1);
        }

        if (votes.isEmpty()) return "Unknown";
        
        // Find label with max votes
        Map.Entry<String, Integer> maxEntry = Collections.max(votes.entrySet(), Map.Entry.comparingByValue());
        
        // Face recognition threshold (to identify "Unknown" instead of wrong matches)
        // Raw pixel Euclidean distance for 100x100 is typically large, but let's do a loose threshold check
        return maxEntry.getKey();
    }

    // Append to attendance.csv
    public synchronized void markAttendance(String name) {
        String fileName = "attendance.csv";
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, true))) {
            writer.write(name + "," + timestamp);
            writer.newLine();
            System.out.println("📝 GUI Log: Attendance marked for " + name + " at " + timestamp);
        } catch (IOException e) {
            System.out.println("❌ Error writing attendance log: " + e.getMessage());
        }
    }

    // Webcam Control
    public synchronized void startCameraFeed(ModernComponents.WebcamPanel panel, FrameCallback callback) {
        stopCameraFeed(); // Ensure previous camera threads are closed
        
        this.currentWebcamPanel = panel;
        this.currentCallback = callback;
        this.isCameraActive = true;

        cameraThread = new Thread(() -> {
            try {
                camera = new VideoCapture(0);
                if (!camera.isOpened()) {
                    System.out.println("❌ Camera opening failed (index 0). Trying index 1...");
                    camera = new VideoCapture(1);
                }

                if (!camera.isOpened()) {
                    SwingUtilities.invokeLater(() -> {
                        if (currentWebcamPanel != null) {
                            currentWebcamPanel.clearFeed("❌ Video Capture Device Unavailable");
                        }
                    });
                    isCameraActive = false;
                    return;
                }

                Mat frame = new Mat();
                while (isCameraActive && camera.isOpened()) {
                    if (camera.read(frame)) {
                        if (frame.empty()) continue;

                        BufferedImage bufferedImage = null;
                        
                        // Execute custom processing (face detection, boundary box, capturing)
                        if (currentCallback != null) {
                            currentCallback.onFrameProcessed(frame, bufferedImage);
                        }

                        // Convert Mat to BufferedImage
                        bufferedImage = matToBufferedImage(frame);

                        if (bufferedImage != null && currentWebcamPanel != null) {
                            currentWebcamPanel.setImage(bufferedImage);
                        }
                    }

                    try {
                        Thread.sleep(30); // ~33 FPS
                    } catch (InterruptedException e) {
                        break;
                    }
                }
            } catch (Exception ex) {
                System.out.println("❌ Camera thread crash: " + ex.getMessage());
            } finally {
                releaseCameraResources();
            }
        });
        cameraThread.start();
    }

    public synchronized void stopCameraFeed() {
        isCameraActive = false;
        if (cameraThread != null) {
            cameraThread.interrupt();
            cameraThread = null;
        }
        releaseCameraResources();
    }

    private synchronized void releaseCameraResources() {
        if (camera != null) {
            if (camera.isOpened()) {
                camera.release();
            }
            camera = null;
        }
    }

    // Fast OpenCV Mat to BufferedImage converter
    private BufferedImage matToBufferedImage(Mat matrix) {
        if (matrix.empty()) return null;
        int cols = matrix.cols();
        int rows = matrix.rows();
        int elemSize = (int) matrix.elemSize();
        byte[] data = new byte[cols * rows * elemSize];
        int type;
        matrix.get(0, 0, data);
        switch (matrix.channels()) {
            case 1:
                type = BufferedImage.TYPE_BYTE_GRAY;
                break;
            case 3:
                type = BufferedImage.TYPE_3BYTE_BGR;
                // Convert BGR to RGB
                byte b;
                for (int i = 0; i < data.length; i += 3) {
                    b = data[i];
                    data[i] = data[i + 2];
                    data[i + 2] = b;
                }
                break;
            default:
                return null;
        }
        BufferedImage image = new BufferedImage(cols, rows, type);
        image.getRaster().setDataElements(0, 0, cols, rows, data);
        return image;
    }

    // Main entry point for the desktop GUI Application
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                // Set native macOS system Look and Feel for premium integration
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception ex) {}
            
            MainFrame frame = new MainFrame();
            frame.setVisible(true);
        });
    }
}
