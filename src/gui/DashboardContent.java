package gui;

import java.awt.*;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import gui.ModernComponents.*;

public class DashboardContent extends JPanel {
    private MainFrame mainFrame;
    private AppWorkspacePanel workspacePanel;

    // Stat Label references to update dynamically
    private JLabel totalStudentsLabel;
    private JLabel totalAttendanceLabel;
    private JLabel presentLabel;
    private JLabel absentLabel;
    
    private DefaultTableModel tableModel;
    private JLabel clockLabel;

    public DashboardContent(MainFrame mainFrame, AppWorkspacePanel workspacePanel) {
        this.mainFrame = mainFrame;
        this.workspacePanel = workspacePanel;
        
        setLayout(new BorderLayout());
        setBackground(Theme.MAIN_BG);
        setBorder(BorderFactory.createEmptyBorder(25, 30, 25, 30));

        initHeaderPanel();
        
        JPanel mainContent = new JPanel();
        mainContent.setLayout(new BoxLayout(mainContent, BoxLayout.Y_AXIS));
        mainContent.setOpaque(false);

        mainContent.add(initStatCardsPanel());
        mainContent.add(Box.createVerticalStrut(25));
        mainContent.add(initRecentAttendancePanel());

        add(mainContent, BorderLayout.CENTER);

        // Start Clock Timer
        startClock();
        
        // Load Initial Stats
        refreshStats();
    }

    private void initHeaderPanel() {
        JPanel header = new JPanel(new BorderLayout());
        header.setOpaque(false);
        header.setBorder(BorderFactory.createEmptyBorder(0, 0, 20, 0));

        // Dashboard Title
        JLabel title = new JLabel("Dashboard");
        title.setFont(Theme.FONT_TITLE_LARGE);
        title.setForeground(Theme.TEXT_DARK);
        header.add(title, BorderLayout.WEST);

        // Right side info (Clock + Profile)
        JPanel rightPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 20, 0));
        rightPanel.setOpaque(false);

        // Live Clock
        clockLabel = new JLabel();
        clockLabel.setFont(Theme.FONT_CLOCK);
        clockLabel.setForeground(Theme.TEXT_LIGHT);
        rightPanel.add(clockLabel);

        // User Avatar Graphic
        JLabel avatarLabel = new JLabel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                
                int w = getWidth();
                int h = getHeight();
                
                // Draw circle backing
                g2.setColor(Theme.SIDEBAR_BG);
                g2.fillOval(0, 0, w - 1, h - 1);
                
                // Draw text label
                g2.setColor(Theme.TEXT_WHITE);
                g2.setFont(Theme.FONT_BODY_BOLD);
                FontMetrics fm = g2.getFontMetrics();
                g2.drawString("AD", (w - fm.stringWidth("AD")) / 2, (h - fm.getHeight()) / 2 + fm.getAscent());
                g2.dispose();
            }
        };
        avatarLabel.setPreferredSize(new Dimension(32, 32));

        JLabel profileName = new JLabel("Admin");
        profileName.setFont(Theme.FONT_BODY_BOLD);
        profileName.setForeground(Theme.TEXT_DARK);

        JPanel profilePanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 0));
        profilePanel.setOpaque(false);
        profilePanel.add(avatarLabel);
        profilePanel.add(profileName);

        rightPanel.add(profilePanel);
        header.add(rightPanel, BorderLayout.EAST);

        add(header, BorderLayout.NORTH);
    }

    private JPanel initStatCardsPanel() {
        JPanel panel = new JPanel(new GridLayout(1, 4, 15, 0));
        panel.setOpaque(false);
        panel.setPreferredSize(new Dimension(100, 110));

        // Card 1: Total Students
        totalStudentsLabel = new JLabel("0", JLabel.LEFT);
        panel.add(createStatCard("Total Students", totalStudentsLabel, "Registered", Theme.ACCENT_BLUE, "👥"));

        // Card 2: Total Attendance
        totalAttendanceLabel = new JLabel("0", JLabel.LEFT);
        panel.add(createStatCard("Total Attendance", totalAttendanceLabel, "Today", Theme.ACCENT_GRADIENT_START, "✓"));

        // Card 3: Present
        presentLabel = new JLabel("0", JLabel.LEFT);
        panel.add(createStatCard("Present Today", presentLabel, "Students", Theme.COLOR_GREEN, "👤"));

        // Card 4: Absent
        absentLabel = new JLabel("0", JLabel.LEFT);
        panel.add(createStatCard("Absent Today", absentLabel, "Students", Theme.COLOR_RED, "👤"));

        return panel;
    }

    private ModernPanel createStatCard(String title, JLabel valLabel, String subText, Color accentColor, String iconText) {
        ModernPanel card = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        card.setLayout(new BorderLayout());
        card.setBorder(BorderFactory.createEmptyBorder(12, 16, 12, 16));

        // Top info section
        JPanel infoPanel = new JPanel(new GridLayout(3, 1, 2, 0));
        infoPanel.setOpaque(false);

        JLabel titleLabel = new JLabel(title);
        titleLabel.setFont(Theme.FONT_CAPTION);
        titleLabel.setForeground(Theme.TEXT_LIGHT);
        infoPanel.add(titleLabel);

        valLabel.setFont(Theme.FONT_TITLE);
        valLabel.setForeground(Theme.TEXT_DARK);
        infoPanel.add(valLabel);

        JLabel subLabel = new JLabel(subText);
        subLabel.setFont(Theme.FONT_CAPTION);
        subLabel.setForeground(Theme.TEXT_LIGHT);
        infoPanel.add(subLabel);

        card.add(infoPanel, BorderLayout.CENTER);

        // Icon Section
        JLabel iconLabel = new JLabel(iconText, JLabel.CENTER) {
            @Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(new Color(accentColor.getRed(), accentColor.getGreen(), accentColor.getBlue(), 30));
                g2.fillRoundRect(0, 0, getWidth(), getHeight(), 10, 10);
                g2.dispose();
                super.paintComponent(g);
            }
        };
        iconLabel.setFont(new Font("Segoe UI", Font.PLAIN, 18));
        iconLabel.setForeground(accentColor);
        iconLabel.setPreferredSize(new Dimension(38, 38));

        JPanel iconContainer = new JPanel(new GridBagLayout());
        iconContainer.setOpaque(false);
        iconContainer.add(iconLabel);

        card.add(iconContainer, BorderLayout.EAST);

        return card;
    }

    private JPanel initRecentAttendancePanel() {
        ModernPanel panel = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        panel.setLayout(new BorderLayout());
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Title Header of Panel
        JPanel titlePanel = new JPanel(new BorderLayout());
        titlePanel.setOpaque(false);
        titlePanel.setBorder(BorderFactory.createEmptyBorder(0, 0, 15, 0));

        JLabel title = new JLabel("Recent Attendance");
        title.setFont(Theme.FONT_SUBTITLE);
        title.setForeground(Theme.TEXT_DARK);
        titlePanel.add(title, BorderLayout.WEST);

        JButton viewAll = new JButton("View All");
        viewAll.setFont(Theme.FONT_CAPTION);
        viewAll.setForeground(Theme.ACCENT_BLUE);
        viewAll.setBorderPainted(false);
        viewAll.setContentAreaFilled(false);
        viewAll.setFocusPainted(false);
        viewAll.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        viewAll.addActionListener(e -> workspacePanel.navigateTo("ViewAttendance"));
        titlePanel.add(viewAll, BorderLayout.EAST);

        panel.add(titlePanel, BorderLayout.NORTH);

        // Modern Data Table setup
        String[] cols = {"Name", "Roll No.", "Department", "Time", "Status"};
        tableModel = new DefaultTableModel(cols, 0) {
            @Override
            public boolean isCellEditable(int r, int c) { return false; }
        };
        JTable table = new JTable(tableModel);
        
        panel.add(ModernComponents.createModernTable(table), BorderLayout.CENTER);

        return panel;
    }

    private void startClock() {
        javax.swing.Timer timer = new javax.swing.Timer(1000, e -> {
            SimpleDateFormat sdf = new SimpleDateFormat("dd MMM yyyy | hh:mm:ss a");
            clockLabel.setText(sdf.format(new Date()));
        });
        timer.start();
    }

    public void refreshStats() {
        // Read dataset student folders
        int totalStudents = 0;
        try {
            java.io.File dir = new java.io.File("dataset");
            if (dir.exists() && dir.isDirectory()) {
                java.io.File[] folders = dir.listFiles(java.io.File::isDirectory);
                if (folders != null) totalStudents = folders.length;
            }
        } catch (Exception ex) {}

        // Read attendance.csv metrics
        int presentToday = 0;
        int absentToday = 0;
        int totalAttendance = 0;
        
        String todayDateStr = new SimpleDateFormat("yyyy-MM-dd").format(new Date());
        
        // Build map of students to check attendance status
        Map<String, String> studentDetails = new HashMap<>(); // Name -> Roll,Dept
        Map<String, String> attendanceTodayMap = new HashMap<>(); // Name -> CheckinTime
        
        // Read registered metadata
        try (BufferedReader br = new BufferedReader(new FileReader("students.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length >= 3) {
                    studentDetails.put(p[1], p[0] + "," + p[2]); // Name -> Roll,Dept
                }
            }
        } catch (Exception ex) {}

        // Read check-ins
        try (BufferedReader br = new BufferedReader(new FileReader("attendance.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length >= 2) {
                    String name = p[0];
                    String timestamp = p[1]; // yyyy-MM-dd HH:mm:ss
                    if (timestamp.startsWith(todayDateStr)) {
                        String timeStr = timestamp.substring(11);
                        attendanceTodayMap.put(name, timeStr);
                    }
                }
            }
        } catch (Exception ex) {}

        totalAttendance = attendanceTodayMap.size();
        presentToday = totalAttendance;
        absentToday = Math.max(0, totalStudents - presentToday);

        // Update card numbers
        totalStudentsLabel.setText(String.valueOf(totalStudents));
        totalAttendanceLabel.setText(String.valueOf(totalAttendance));
        presentLabel.setText(String.valueOf(presentToday));
        absentLabel.setText(String.valueOf(absentToday));

        // Refresh Recent Table rows (last 6 check-ins)
        tableModel.setRowCount(0);
        
        // Compile list of recent records
        java.util.List<Object[]> recentRecords = new ArrayList<>();
        
        // Read attendance.csv again to extract logs
        try (BufferedReader br = new BufferedReader(new FileReader("attendance.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length >= 2) {
                    String name = p[0];
                    String timestamp = p[1];
                    
                    String roll = "N/A";
                    String dept = "N/A";
                    if (studentDetails.containsKey(name)) {
                        String[] details = studentDetails.get(name).split(",");
                        roll = details[0];
                        dept = details[1];
                    }
                    
                    String timeOnly = timestamp.substring(11);
                    recentRecords.add(new Object[]{name, roll, dept, timeOnly, "Present"});
                }
            }
        } catch (Exception ex) {}

        // Populate table (latest at the top)
        int displayed = 0;
        for (int i = recentRecords.size() - 1; i >= 0 && displayed < 6; i--) {
            tableModel.addRow(recentRecords.get(i));
            displayed++;
        }
        
        // If there are registered students who are absent, append them for today's visual log as "Absent"
        if (displayed < 6) {
            for (String regName : studentDetails.keySet()) {
                if (!attendanceTodayMap.containsKey(regName) && displayed < 6) {
                    String[] details = studentDetails.get(regName).split(",");
                    tableModel.addRow(new Object[]{regName, details[0], details[1], "--:--", "Absent"});
                    displayed++;
                }
            }
        }
    }
}
