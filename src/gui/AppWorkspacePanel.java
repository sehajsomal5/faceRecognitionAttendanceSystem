package gui;

import java.awt.*;
import javax.swing.*;

public class AppWorkspacePanel extends JPanel {
    private MainFrame mainFrame;
    private SidebarPanel sidebarPanel;
    private JPanel cardContainer;
    private CardLayout cardLayout;

    // Content Screen panels
    private DashboardContent dashboardContent;
    private StudentRegistrationScreen registrationContent;
    private AttendanceMonitoringScreen monitoringContent;
    private AttendanceRecordsScreen recordsContent;
    private ReportsAnalyticsScreen reportsContent;
    private JPanel settingsContent;

    public AppWorkspacePanel(MainFrame mainFrame) {
        this.mainFrame = mainFrame;
        setLayout(new BorderLayout());
        setBackground(Theme.MAIN_BG);

        // Center card container
        cardLayout = new CardLayout();
        cardContainer = new JPanel(cardLayout);
        cardContainer.setOpaque(false);

        // Sidebar Panel (passing this workspace reference to link actions)
        sidebarPanel = new SidebarPanel(mainFrame, this);
        add(sidebarPanel, BorderLayout.WEST);

        // Initialize and add all workspace screens
        dashboardContent = new DashboardContent(mainFrame, this);
        registrationContent = new StudentRegistrationScreen(mainFrame, this);
        monitoringContent = new AttendanceMonitoringScreen(mainFrame, this);
        recordsContent = new AttendanceRecordsScreen(mainFrame, this);
        reportsContent = new ReportsAnalyticsScreen(mainFrame, this);
        initSettingsContent();

        cardContainer.add(dashboardContent, "Dashboard");
        cardContainer.add(registrationContent, "RegisterStudent");
        cardContainer.add(monitoringContent, "MarkAttendance");
        cardContainer.add(recordsContent, "ViewAttendance");
        cardContainer.add(reportsContent, "Reports");
        cardContainer.add(settingsContent, "Settings");

        add(cardContainer, BorderLayout.CENTER);
    }

    private void initSettingsContent() {
        settingsContent = new JPanel(new GridBagLayout());
        settingsContent.setBackground(Theme.MAIN_BG);

        ModernComponents.ModernPanel card = new ModernComponents.ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        card.setLayout(new GridBagLayout());
        card.setBorder(BorderFactory.createEmptyBorder(30, 40, 30, 40));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = GridBagConstraints.RELATIVE;
        gbc.insets = new Insets(10, 0, 10, 0);

        JLabel title = new JLabel("System Settings", JLabel.CENTER);
        title.setFont(Theme.FONT_TITLE);
        title.setForeground(Theme.TEXT_DARK);
        card.add(title, gbc);

        JLabel desc = new JLabel("<html><center>Configure camera device indexes and system parameters.<br>Default webcam index: 0</center></html>", JLabel.CENTER);
        desc.setFont(Theme.FONT_BODY);
        desc.setForeground(Theme.TEXT_LIGHT);
        card.add(desc, gbc);

        settingsContent.add(card);
    }

    public void showWorkspaceCard(String cardName) {
        // Stop camera feed when moving away from capture or monitoring screens
        mainFrame.stopCameraFeed();
        
        // Refresh records or stats when opening Dashboard or Records screen
        if (cardName.equals("Dashboard")) {
            dashboardContent.refreshStats();
        } else if (cardName.equals("ViewAttendance")) {
            recordsContent.refreshTable();
        } else if (cardName.equals("Reports")) {
            reportsContent.refreshReports();
        }

        cardLayout.show(cardContainer, cardName);
    }

    public void navigateTo(String cardName) {
        sidebarPanel.selectMenuItemByCardName(cardName);
    }

    // Trigger dataset training via background thread to prevent UI freezing
    public void triggerDatasetTraining() {
        mainFrame.stopCameraFeed();

        // Show progress overlay
        JDialog trainingDialog = new JDialog(mainFrame, "Training AI Model", true);
        trainingDialog.setLayout(new GridBagLayout());
        trainingDialog.setSize(350, 150);
        trainingDialog.setLocationRelativeTo(mainFrame);
        trainingDialog.setResizable(false);
        trainingDialog.setUndecorated(true);

        ModernComponents.ModernPanel card = new ModernComponents.ModernPanel(Theme.ROUNDNESS_CARD, Theme.SIDEBAR_BG);
        card.setLayout(new GridBagLayout());
        card.setBorder(BorderFactory.createEmptyBorder(15, 20, 15, 20));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = GridBagConstraints.RELATIVE;
        gbc.insets = new Insets(10, 0, 10, 0);
        gbc.anchor = GridBagConstraints.CENTER;

        JLabel title = new JLabel("Training AI Face Dataset...", JLabel.CENTER);
        title.setFont(Theme.FONT_SUBTITLE);
        title.setForeground(Theme.TEXT_WHITE);
        card.add(title, gbc);

        JProgressBar progress = new JProgressBar();
        progress.setIndeterminate(true);
        progress.setForeground(Theme.ACCENT_BLUE);
        progress.setBackground(Theme.SIDEBAR_HOVER);
        progress.setPreferredSize(new Dimension(280, 16));
        card.add(progress, gbc);

        trainingDialog.add(card);

        // Run training in a background SwingWorker thread
        SwingWorker<Boolean, Void> worker = new SwingWorker<Boolean, Void>() {
            @Override
            protected Boolean doInBackground() throws Exception {
                // Call PrepareDataset logic directly
                try {
                    String datasetPath = "dataset";
                    java.io.File datasetDir = new java.io.File(datasetPath);
                    if (!datasetDir.exists()) {
                        datasetDir.mkdirs();
                    }
                    java.io.File[] userDirs = datasetDir.listFiles(java.io.File::isDirectory);

                    if (userDirs == null || userDirs.length == 0) {
                        return false;
                    }

                    try (java.io.BufferedWriter writer = new java.io.BufferedWriter(new java.io.FileWriter("faces_dataset.csv"))) {
                        for (java.io.File userDir : userDirs) {
                            String label = userDir.getName();
                            java.io.File[] images = userDir.listFiles((d, name) -> name.toLowerCase().endsWith(".jpg"));

                            if (images == null) continue;

                            for (java.io.File imgFile : images) {
                                org.opencv.imgcodecs.Imgcodecs imgcodecs = new org.opencv.imgcodecs.Imgcodecs();
                                org.opencv.core.Mat img = org.opencv.imgcodecs.Imgcodecs.imread(imgFile.getAbsolutePath(), org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE);
                                if (img.empty()) continue;

                                org.opencv.imgproc.Imgproc.resize(img, img, new org.opencv.core.Size(100, 100));

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
                    }
                    // reload global dataset in main frame
                    mainFrame.reloadDataset();
                    return true;
                } catch (Exception ex) {
                    ex.printStackTrace();
                    return false;
                }
            }

            @Override
            protected void done() {
                trainingDialog.dispose();
                try {
                    boolean success = get();
                    if (success) {
                        JOptionPane.showMessageDialog(mainFrame, 
                            "✅ AI Face Recognizer trained successfully!", 
                            "Training Completed", JOptionPane.INFORMATION_MESSAGE);
                        dashboardContent.refreshStats();
                    } else {
                        JOptionPane.showMessageDialog(mainFrame, 
                            "❌ Failed to compile dataset. Make sure you have registered students with captured photos.", 
                            "Training Failed", JOptionPane.ERROR_MESSAGE);
                    }
                } catch (Exception ex) {
                    JOptionPane.showMessageDialog(mainFrame, 
                        "❌ Error during training: " + ex.getMessage(), 
                        "Training Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        };

        worker.execute();
        trainingDialog.setVisible(true);
    }
}
