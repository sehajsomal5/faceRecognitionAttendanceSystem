package gui;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import javax.swing.*;
import gui.ModernComponents.*;

public class AttendanceMonitoringScreen extends JPanel {
    private MainFrame mainFrame;
    private AppWorkspacePanel workspacePanel;

    // Camera view
    private WebcamPanel webcamPanel;
    private ModernButton startStopBtn;
    private boolean isCameraRunning = false;

    // Student profile panel (Right side)
    private JLabel studentAvatar;
    private JLabel studentNameLbl;
    private JLabel studentRollLbl;
    private JLabel studentDeptLbl;
    private JPanel statusBanner;
    private JLabel statusBannerLbl;

    // Stats Panel
    private JLabel presentCountLbl;
    private JLabel absentCountLbl;

    // Metadata caches
    private Map<String, String[]> studentDatabase = new HashMap<>(); // Name -> [Roll, Dept]
    private Set<String> markedAttendanceToday = new HashSet<>();

    public AttendanceMonitoringScreen(MainFrame mainFrame, AppWorkspacePanel workspacePanel) {
        this.mainFrame = mainFrame;
        this.workspacePanel = workspacePanel;

        setLayout(new BorderLayout());
        setBackground(Theme.MAIN_BG);
        setBorder(BorderFactory.createEmptyBorder(25, 30, 25, 30));

        initHeader();

        JPanel contentSplit = new JPanel(new GridBagLayout());
        contentSplit.setOpaque(false);

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weighty = 1.0;

        // Left Webcam panel
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weightx = 0.6;
        gbc.insets = new Insets(0, 0, 0, 15);
        contentSplit.add(initWebcamPanel(), gbc);

        // Right details panel
        gbc.gridx = 1;
        gbc.gridy = 0;
        gbc.weightx = 0.4;
        gbc.insets = new Insets(0, 15, 0, 0);
        contentSplit.add(initDetailsPanel(), gbc);

        add(contentSplit, BorderLayout.CENTER);
    }

    private void initHeader() {
        JPanel header = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        header.setOpaque(false);
        header.setBorder(BorderFactory.createEmptyBorder(0, 0, 20, 0));

        JLabel title = new JLabel("Mark Attendance");
        title.setFont(Theme.FONT_TITLE_LARGE);
        title.setForeground(Theme.TEXT_DARK);
        header.add(title);

        add(header, BorderLayout.NORTH);
    }

    private JPanel initWebcamPanel() {
        ModernPanel camPanel = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        camPanel.setLayout(new BorderLayout());
        camPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        JLabel camTitle = new JLabel("Live Camera Feed");
        camTitle.setFont(Theme.FONT_SUBTITLE);
        camTitle.setForeground(Theme.TEXT_DARK);
        camTitle.setBorder(BorderFactory.createEmptyBorder(0, 0, 12, 0));
        camPanel.add(camTitle, BorderLayout.NORTH);

        webcamPanel = new WebcamPanel();
        camPanel.add(webcamPanel, BorderLayout.CENTER);

        // Control button bar
        JPanel btnPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 15));
        btnPanel.setOpaque(false);

        startStopBtn = new ModernButton("Start Camera");
        startStopBtn.setPreferredSize(new Dimension(185, 40));
        startStopBtn.addActionListener(e -> toggleCamera());
        btnPanel.add(startStopBtn);

        camPanel.add(btnPanel, BorderLayout.SOUTH);

        return camPanel;
    }

    private JPanel initDetailsPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setOpaque(false);

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = GridBagConstraints.RELATIVE;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.insets = new Insets(0, 0, 20, 0);

        // Details Panel Card
        ModernPanel detailsCard = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        detailsCard.setLayout(new GridBagLayout());
        detailsCard.setBorder(BorderFactory.createEmptyBorder(25, 25, 25, 25));

        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = GridBagConstraints.RELATIVE;
        c.insets = new Insets(8, 0, 8, 0);
        c.anchor = GridBagConstraints.CENTER;

        JLabel title = new JLabel("Recognition Result");
        title.setFont(Theme.FONT_SUBTITLE);
        title.setForeground(Theme.TEXT_DARK);
        c.anchor = GridBagConstraints.WEST;
        detailsCard.add(title, c);
        
        // Avatar circle / frame photo
        c.anchor = GridBagConstraints.CENTER;
        studentAvatar = new JLabel() {
            private BufferedImage defaultAvatar = null;
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                
                int w = getWidth();
                int h = getHeight();
                
                // Draw a beautiful circular frame
                g2.setColor(new Color(241, 245, 249));
                g2.fillOval(0, 0, w - 1, h - 1);
                g2.setColor(new Color(226, 232, 240));
                g2.setStroke(new BasicStroke(2.0f));
                g2.drawOval(0, 0, w - 1, h - 1);
                
                // Draw face placeholder icon if no photo loaded
                g2.setColor(Theme.TEXT_LIGHT);
                g2.fillOval(w/2 - 15, h/2 - 20, 30, 30); // head
                g2.fillArc(w/2 - 25, h/2 + 5, 50, 40, 0, 180); // shoulders
                
                g2.dispose();
            }
        };
        studentAvatar.setPreferredSize(new Dimension(100, 100));
        c.insets = new Insets(15, 0, 15, 0);
        detailsCard.add(studentAvatar, c);

        studentNameLbl = new JLabel("Scanning...", JLabel.CENTER);
        studentNameLbl.setFont(Theme.FONT_TITLE);
        studentNameLbl.setForeground(Theme.TEXT_DARK);
        c.insets = new Insets(4, 0, 4, 0);
        detailsCard.add(studentNameLbl, c);

        studentRollLbl = new JLabel("Roll No: --", JLabel.CENTER);
        studentRollLbl.setFont(Theme.FONT_BODY);
        studentRollLbl.setForeground(Theme.TEXT_LIGHT);
        detailsCard.add(studentRollLbl, c);

        studentDeptLbl = new JLabel("Dept: --", JLabel.CENTER);
        studentDeptLbl.setFont(Theme.FONT_BODY);
        studentDeptLbl.setForeground(Theme.TEXT_LIGHT);
        c.insets = new Insets(4, 0, 20, 0);
        detailsCard.add(studentDeptLbl, c);

        // Status banner (e.g. green "Attendance Marked" card)
        statusBanner = new JPanel(new GridBagLayout());
        statusBanner.setBackground(new Color(34, 197, 94, 30)); // 30alpha green
        statusBanner.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(Theme.COLOR_GREEN, 1),
            BorderFactory.createEmptyBorder(10, 20, 10, 20)
        ));
        statusBanner.setVisible(false);

        statusBannerLbl = new JLabel("Attendance Marked", JLabel.CENTER);
        statusBannerLbl.setFont(Theme.FONT_BODY_BOLD);
        statusBannerLbl.setForeground(Theme.COLOR_GREEN);
        statusBanner.add(statusBannerLbl);
        
        c.fill = GridBagConstraints.HORIZONTAL;
        c.insets = new Insets(5, 0, 5, 0);
        detailsCard.add(statusBanner, c);

        panel.add(detailsCard, gbc);

        // Stats Card Card (Bottom Right)
        ModernPanel statsCard = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        statsCard.setLayout(new GridLayout(1, 2, 20, 0));
        statsCard.setBorder(BorderFactory.createEmptyBorder(15, 20, 15, 20));

        JPanel pres = new JPanel(new GridLayout(2, 1, 2, 0));
        pres.setOpaque(false);
        JLabel pTitle = new JLabel("Today's Present", JLabel.CENTER);
        pTitle.setFont(Theme.FONT_CAPTION);
        pTitle.setForeground(Theme.TEXT_LIGHT);
        presentCountLbl = new JLabel("0", JLabel.CENTER);
        presentCountLbl.setFont(Theme.FONT_TITLE);
        presentCountLbl.setForeground(Theme.COLOR_GREEN);
        pres.add(pTitle);
        pres.add(presentCountLbl);

        JPanel abs = new JPanel(new GridLayout(2, 1, 2, 0));
        abs.setOpaque(false);
        JLabel aTitle = new JLabel("Today's Absent", JLabel.CENTER);
        aTitle.setFont(Theme.FONT_CAPTION);
        aTitle.setForeground(Theme.TEXT_LIGHT);
        absentCountLbl = new JLabel("0", JLabel.CENTER);
        absentCountLbl.setFont(Theme.FONT_TITLE);
        absentCountLbl.setForeground(Theme.COLOR_RED);
        abs.add(aTitle);
        abs.add(absentCountLbl);

        statsCard.add(pres);
        statsCard.add(abs);

        panel.add(statsCard, gbc);

        return panel;
    }

    private void toggleCamera() {
        if (isCameraRunning) {
            stopCamera();
        } else {
            startCamera();
        }
    }

    private void startCamera() {
        if (mainFrame.getDataset() == null || mainFrame.getDataset().isEmpty()) {
            JOptionPane.showMessageDialog(this, "⚠️ AI model has no trained dataset. Please train faces first!", "Dataset Empty", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // Cache metadata files
        loadMetadata();
        
        isCameraRunning = true;
        startStopBtn.setText("Stop Camera");
        startStopBtn.setColors(Theme.COLOR_RED, Theme.COLOR_RED.darker());
        webcamPanel.setShowFrameOutline(true);

        mainFrame.startCameraFeed(webcamPanel, new MainFrame.FrameCallback() {
            private long lastRecognizedTime = 0;

            @Override
            public void onFrameProcessed(org.opencv.core.Mat frame, BufferedImage img) {
                // Live Face Detection
                org.opencv.objdetect.CascadeClassifier detector = mainFrame.getFaceDetector();
                org.opencv.core.MatOfRect faceDetections = new org.opencv.core.MatOfRect();
                detector.detectMultiScale(frame, faceDetections);
                
                org.opencv.core.Rect[] faces = faceDetections.toArray();
                if (faces.length > 0) {
                    org.opencv.core.Rect face = faces[0];
                    
                    // Predict face matching in background thread or safely inside callback
                    org.opencv.core.Mat faceCrop = new org.opencv.core.Mat(frame, face);
                    org.opencv.imgproc.Imgproc.resize(faceCrop, faceCrop, new org.opencv.core.Size(100, 100));
                    org.opencv.imgproc.Imgproc.cvtColor(faceCrop, faceCrop, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY);

                    // Flatten face vector
                    double[] vector = new double[10000];
                    int idx = 0;
                    for (int r = 0; r < faceCrop.rows(); r++) {
                        for (int c = 0; c < faceCrop.cols(); c++) {
                            vector[idx++] = faceCrop.get(r, c)[0];
                        }
                    }

                    // Predict
                    String predictedName = mainFrame.predict(vector);
                    
                    // Draw bounding box on frame
                    org.opencv.imgproc.Imgproc.rectangle(frame, face, new org.opencv.core.Scalar(34, 197, 94), 2);

                    if (predictedName != null && !predictedName.equals("Unknown")) {
                        long now = System.currentTimeMillis();
                        if (now - lastRecognizedTime > 2500) { // Throttling: trigger check-in every 2.5s
                            lastRecognizedTime = now;
                            triggerCheckin(predictedName);
                        }
                    } else {
                        // Drawing unknown label
                        org.opencv.imgproc.Imgproc.rectangle(frame, face, new org.opencv.core.Scalar(239, 68, 68), 2);
                        SwingUtilities.invokeLater(() -> {
                            studentNameLbl.setText("Unknown Face");
                            studentRollLbl.setText("Roll No: --");
                            studentDeptLbl.setText("Dept: --");
                            statusBanner.setVisible(false);
                        });
                    }
                }
            }
        });
    }

    private void stopCamera() {
        isCameraRunning = false;
        startStopBtn.setText("Start Camera");
        startStopBtn.setColors(Theme.ACCENT_GRADIENT_START, Theme.ACCENT_GRADIENT_END);
        webcamPanel.clearFeed("Camera Offline");
        studentNameLbl.setText("Scanning...");
        studentRollLbl.setText("Roll No: --");
        studentDeptLbl.setText("Dept: --");
        statusBanner.setVisible(false);
        mainFrame.stopCameraFeed();
    }

    private void loadMetadata() {
        studentDatabase.clear();
        markedAttendanceToday.clear();

        // 1. Read registered students details
        try (BufferedReader br = new BufferedReader(new FileReader("students.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 3) {
                    studentDatabase.put(parts[1], new String[]{parts[0], parts[2]}); // Name -> [Roll, Dept]
                }
            }
        } catch (Exception ex) {}

        // 2. Read marked check-ins today
        String today = new SimpleDateFormat("yyyy-MM-dd").format(new Date());
        try (BufferedReader br = new BufferedReader(new FileReader("attendance.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length >= 2) {
                    String name = parts[0];
                    String timestamp = parts[1];
                    if (timestamp.startsWith(today)) {
                        markedAttendanceToday.add(name);
                    }
                }
            }
        } catch (Exception ex) {}

        refreshStatsLabels();
    }

    private void triggerCheckin(String name) {
        String roll = "N/A";
        String dept = "N/A";
        
        if (studentDatabase.containsKey(name)) {
            String[] det = studentDatabase.get(name);
            roll = det[0];
            dept = det[1];
        }

        final String finalRoll = roll;
        final String finalDept = dept;

        // Perform attendance mark
        boolean newlyMarked = false;
        if (!markedAttendanceToday.contains(name)) {
            newlyMarked = true;
            markedAttendanceToday.add(name);
            mainFrame.markAttendance(name);
        }

        final boolean showMarkedLabel = newlyMarked;
        
        SwingUtilities.invokeLater(() -> {
            studentNameLbl.setText(name);
            studentRollLbl.setText("Roll No: " + finalRoll);
            studentDeptLbl.setText("Dept: " + finalDept);
            
            SimpleDateFormat timeFormat = new SimpleDateFormat("hh:mm:ss a");
            String timeStr = timeFormat.format(new Date());
            
            if (showMarkedLabel) {
                statusBannerLbl.setText("Attendance Marked: " + timeStr);
                statusBanner.setBackground(new Color(34, 197, 94, 30));
                statusBannerLbl.setForeground(Theme.COLOR_GREEN);
                statusBanner.setBorder(BorderFactory.createLineBorder(Theme.COLOR_GREEN, 1));
            } else {
                statusBannerLbl.setText("Checked In (Already Marked)");
                statusBanner.setBackground(new Color(59, 130, 246, 30));
                statusBannerLbl.setForeground(Theme.ACCENT_BLUE);
                statusBanner.setBorder(BorderFactory.createLineBorder(Theme.ACCENT_BLUE, 1));
            }
            statusBanner.setVisible(true);

            // Attempt to load student image from dataset
            try {
                String cleanName = name.replaceAll("\\s+", "_");
                java.io.File imgFile = new java.io.File("dataset/" + cleanName + "/1.jpg");
                if (imgFile.exists()) {
                    ImageIcon icon = new ImageIcon(imgFile.getAbsolutePath());
                    // Resize to fit avatar
                    Image imgScaled = icon.getImage().getScaledInstance(100, 100, Image.SCALE_SMOOTH);
                    studentAvatar.setIcon(new ImageIcon(imgScaled));
                } else {
                    studentAvatar.setIcon(null);
                }
            } catch (Exception ex) {
                studentAvatar.setIcon(null);
            }

            refreshStatsLabels();
        });
    }

    private void refreshStatsLabels() {
        int totalStudents = 0;
        try {
            java.io.File dir = new java.io.File("dataset");
            if (dir.exists() && dir.isDirectory()) {
                java.io.File[] files = dir.listFiles(java.io.File::isDirectory);
                if (files != null) totalStudents = files.length;
            }
        } catch (Exception ex) {}

        int presentCount = markedAttendanceToday.size();
        int absentCount = Math.max(0, totalStudents - presentCount);

        presentCountLbl.setText(String.valueOf(presentCount));
        absentCountLbl.setText(String.valueOf(absentCount));
    }
}
