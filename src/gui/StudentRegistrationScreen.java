package gui;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import javax.swing.*;
import gui.ModernComponents.*;

public class StudentRegistrationScreen extends JPanel {
    private MainFrame mainFrame;
    private AppWorkspacePanel workspacePanel;

    // Form inputs
    private ModernTextField nameField;
    private ModernTextField rollField;
    private JComboBox<String> deptCombo;
    private ModernTextField emailField;
    private ModernTextField phoneField;

    // Webcam panel and buttons
    private WebcamPanel webcamPanel;
    private ModernButton captureBtn;
    private ModernButton saveBtn;
    private JLabel statusLabel;

    private boolean isCapturing = false;
    private int captureCount = 0;
    private final int TOTAL_REQUIRED_CAPTURES = 30;

    public StudentRegistrationScreen(MainFrame mainFrame, AppWorkspacePanel workspacePanel) {
        this.mainFrame = mainFrame;
        this.workspacePanel = workspacePanel;

        setLayout(new BorderLayout());
        setBackground(Theme.MAIN_BG);
        setBorder(BorderFactory.createEmptyBorder(25, 30, 25, 30));

        initHeader();

        // Split panel: Left = Form, Right = Camera
        JPanel mainSplit = new JPanel(new GridLayout(1, 2, 25, 0));
        mainSplit.setOpaque(false);

        mainSplit.add(initFormPanel());
        mainSplit.add(initCameraPanel());

        add(mainSplit, BorderLayout.CENTER);
    }

    private void initHeader() {
        JPanel header = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        header.setOpaque(false);
        header.setBorder(BorderFactory.createEmptyBorder(0, 0, 20, 0));

        JLabel title = new JLabel("Register Student");
        title.setFont(Theme.FONT_TITLE_LARGE);
        title.setForeground(Theme.TEXT_DARK);
        header.add(title);

        add(header, BorderLayout.NORTH);
    }

    private JPanel initFormPanel() {
        ModernPanel formPanel = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        formPanel.setLayout(new GridBagLayout());
        formPanel.setBorder(BorderFactory.createEmptyBorder(20, 25, 20, 25));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = GridBagConstraints.RELATIVE;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 1.0;
        gbc.insets = new Insets(6, 0, 2, 0);

        // Name
        JLabel nameLbl = new JLabel("Student Name *");
        nameLbl.setFont(Theme.FONT_BODY_BOLD);
        nameLbl.setForeground(Theme.TEXT_DARK);
        formPanel.add(nameLbl, gbc);

        nameField = new ModernTextField("Enter full name", 20);
        gbc.insets = new Insets(0, 0, 10, 0);
        formPanel.add(nameField, gbc);

        // Roll Number
        JLabel rollLbl = new JLabel("Roll Number *");
        rollLbl.setFont(Theme.FONT_BODY_BOLD);
        rollLbl.setForeground(Theme.TEXT_DARK);
        gbc.insets = new Insets(6, 0, 2, 0);
        formPanel.add(rollLbl, gbc);

        rollField = new ModernTextField("Enter roll number", 20);
        gbc.insets = new Insets(0, 0, 10, 0);
        formPanel.add(rollField, gbc);

        // Department
        JLabel deptLbl = new JLabel("Department *");
        deptLbl.setFont(Theme.FONT_BODY_BOLD);
        deptLbl.setForeground(Theme.TEXT_DARK);
        gbc.insets = new Insets(6, 0, 2, 0);
        formPanel.add(deptLbl, gbc);

        String[] depts = {"Computer Science", "Information Tech", "Electronics", "Mechanical", "Civil"};
        deptCombo = new JComboBox<>(depts);
        deptCombo.setFont(Theme.FONT_BODY);
        deptCombo.setBackground(Color.WHITE);
        gbc.insets = new Insets(0, 0, 10, 0);
        formPanel.add(deptCombo, gbc);

        // Email
        JLabel emailLbl = new JLabel("Email (Optional)");
        emailLbl.setFont(Theme.FONT_BODY_BOLD);
        emailLbl.setForeground(Theme.TEXT_DARK);
        gbc.insets = new Insets(6, 0, 2, 0);
        formPanel.add(emailLbl, gbc);

        emailField = new ModernTextField("Enter email address", 20);
        gbc.insets = new Insets(0, 0, 10, 0);
        formPanel.add(emailField, gbc);

        // Phone
        JLabel phoneLbl = new JLabel("Phone (Optional)");
        phoneLbl.setFont(Theme.FONT_BODY_BOLD);
        phoneLbl.setForeground(Theme.TEXT_DARK);
        gbc.insets = new Insets(6, 0, 2, 0);
        formPanel.add(phoneLbl, gbc);

        phoneField = new ModernTextField("Enter phone number", 20);
        gbc.insets = new Insets(0, 0, 20, 0);
        formPanel.add(phoneField, gbc);

        // Form buttons
        JPanel btns = new JPanel(new GridLayout(1, 2, 15, 0));
        btns.setOpaque(false);

        ModernButton clearBtn = new ModernButton("Clear", new Color(226, 232, 240), new Color(203, 213, 225));
        clearBtn.setForeground(Theme.TEXT_DARK);
        clearBtn.addActionListener(e -> clearForm());

        saveBtn = new ModernButton("Save Student");
        saveBtn.setEnabled(false); // Enabled after capturing faces
        saveBtn.addActionListener(e -> handleSaveStudent());

        btns.add(clearBtn);
        btns.add(saveBtn);
        
        gbc.insets = new Insets(10, 0, 5, 0);
        formPanel.add(btns, gbc);

        return formPanel;
    }

    private JPanel initCameraPanel() {
        ModernPanel camPanel = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        camPanel.setLayout(new BorderLayout());
        camPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Heading info
        JPanel heading = new JPanel(new BorderLayout());
        heading.setOpaque(false);
        heading.setBorder(BorderFactory.createEmptyBorder(0, 0, 10, 0));

        JLabel title = new JLabel("Capture Face");
        title.setFont(Theme.FONT_SUBTITLE);
        title.setForeground(Theme.TEXT_DARK);
        heading.add(title, BorderLayout.WEST);

        statusLabel = new JLabel("Camera Offline", JLabel.RIGHT);
        statusLabel.setFont(Theme.FONT_CAPTION);
        statusLabel.setForeground(Theme.TEXT_LIGHT);
        heading.add(statusLabel, BorderLayout.EAST);

        camPanel.add(heading, BorderLayout.NORTH);

        // Webcam canvas
        webcamPanel = new WebcamPanel();
        webcamPanel.setShowFrameOutline(true);
        camPanel.add(webcamPanel, BorderLayout.CENTER);

        // Camera control button
        JPanel btnPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 0, 15));
        btnPanel.setOpaque(false);

        captureBtn = new ModernButton("Capture Photo");
        captureBtn.setPreferredSize(new Dimension(180, 40));
        captureBtn.addActionListener(e -> handleCaptureFaces());
        btnPanel.add(captureBtn);

        camPanel.add(btnPanel, BorderLayout.SOUTH);

        return camPanel;
    }

    private void clearForm() {
        nameField.setText("");
        rollField.setText("");
        deptCombo.setSelectedIndex(0);
        emailField.setText("");
        phoneField.setText("");
        
        mainFrame.stopCameraFeed();
        webcamPanel.clearFeed("Camera Offline");
        statusLabel.setText("Camera Offline");
        captureBtn.setText("Capture Photo");
        captureBtn.setEnabled(true);
        saveBtn.setEnabled(false);
        isCapturing = false;
        captureCount = 0;
    }

    private void handleCaptureFaces() {
        String studentName = nameField.getText().trim();
        String rollNum = rollField.getText().trim();

        if (studentName.isEmpty() || rollNum.isEmpty()) {
            JOptionPane.showMessageDialog(this, "⚠️ Please fill in Student Name and Roll Number first!", "Missing Fields", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // Replace spaces for folder name safety
        String cleanName = studentName.replaceAll("\\s+", "_");
        java.io.File folder = new java.io.File("dataset/" + cleanName);
        if (folder.exists()) {
            int confirm = JOptionPane.showConfirmDialog(this, 
                "⚠️ Dataset folder for '" + studentName + "' already exists. Overwrite?", 
                "Folder Exists", JOptionPane.YES_NO_OPTION);
            if (confirm != JOptionPane.YES_OPTION) {
                return;
            }
        }

        isCapturing = true;
        captureCount = 0;
        captureBtn.setEnabled(false);
        statusLabel.setText("Initializing Camera...");

        // Start webcam feed and running face capture loop in background
        mainFrame.startCameraFeed(webcamPanel, new MainFrame.FrameCallback() {
            @Override
            public void onFrameProcessed(org.opencv.core.Mat frame, BufferedImage img) {
                if (!isCapturing) return;

                // Simple face detection and capture
                org.opencv.objdetect.CascadeClassifier detector = mainFrame.getFaceDetector();
                org.opencv.core.MatOfRect faceDetections = new org.opencv.core.MatOfRect();
                
                detector.detectMultiScale(frame, faceDetections);
                org.opencv.core.Rect[] facesArray = faceDetections.toArray();

                if (facesArray.length > 0) {
                    captureCount++;
                    statusLabel.setText("Capturing: " + captureCount + " / " + TOTAL_REQUIRED_CAPTURES);

                    // Crop face frame
                    org.opencv.core.Rect rect = facesArray[0];
                    org.opencv.core.Mat faceCrop = new org.opencv.core.Mat(frame, rect);

                    // Save face cropped image
                    if (!folder.exists()) {
                        folder.mkdirs();
                    }
                    String path = folder.getAbsolutePath() + "/" + captureCount + ".jpg";
                    org.opencv.imgcodecs.Imgcodecs.imwrite(path, faceCrop);

                    // Check if capture finished
                    if (captureCount >= TOTAL_REQUIRED_CAPTURES) {
                        isCapturing = false;
                        mainFrame.stopCameraFeed();
                        SwingUtilities.invokeLater(() -> {
                            webcamPanel.clearFeed("✅ Faces Captured Successfully!");
                            statusLabel.setText("Capture Complete");
                            captureBtn.setText("Re-Capture Photo");
                            captureBtn.setEnabled(true);
                            saveBtn.setEnabled(true);
                            JOptionPane.showMessageDialog(StudentRegistrationScreen.this, 
                                "✅ 30 face frames captured successfully! Click 'Save Student' to register.", 
                                "Captures Finished", JOptionPane.INFORMATION_MESSAGE);
                        });
                    }
                }
            }
        });
    }

    private void handleSaveStudent() {
        String studentName = nameField.getText().trim();
        String rollNum = rollField.getText().trim();
        String department = (String) deptCombo.getSelectedItem();
        String email = emailField.getText().trim();
        String phone = phoneField.getText().trim();

        if (studentName.isEmpty() || rollNum.isEmpty()) {
            JOptionPane.showMessageDialog(this, "⚠️ Please fill in Student Name and Roll Number!", "Missing Fields", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // Save registration details to students.csv
        try (BufferedWriter writer = new java.io.BufferedWriter(new FileWriter("students.csv", true))) {
            writer.write(rollNum + "," + studentName + "," + department + "," + (email.isEmpty() ? "N/A" : email) + "," + (phone.isEmpty() ? "N/A" : phone));
            writer.newLine();
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "❌ Error saving student metadata: " + ex.getMessage(), "IO Error", JOptionPane.ERROR_MESSAGE);
            return;
        }

        int trainChoice = JOptionPane.showConfirmDialog(this, 
            "✅ Student registered successfully!\nWould you like to train the AI model now to activate face recognition?", 
            "Register Success", JOptionPane.YES_NO_OPTION);

        clearForm();

        if (trainChoice == JOptionPane.YES_OPTION) {
            workspacePanel.triggerDatasetTraining();
        } else {
            workspacePanel.navigateTo("Dashboard");
        }
    }
}
