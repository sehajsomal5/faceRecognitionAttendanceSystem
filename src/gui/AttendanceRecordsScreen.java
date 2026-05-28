package gui;

import java.awt.*;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import gui.ModernComponents.*;

public class AttendanceRecordsScreen extends JPanel {
    private MainFrame mainFrame;
    private AppWorkspacePanel workspacePanel;

    private ModernTextField dateField;
    private JComboBox<String> deptCombo;
    private ModernTextField searchField;
    private DefaultTableModel tableModel;
    private JTable table;

    // Cache lists
    private java.util.List<Object[]> allRecordsCache = new ArrayList<>();

    public AttendanceRecordsScreen(MainFrame mainFrame, AppWorkspacePanel workspacePanel) {
        this.mainFrame = mainFrame;
        this.workspacePanel = workspacePanel;

        setLayout(new BorderLayout());
        setBackground(Theme.MAIN_BG);
        setBorder(BorderFactory.createEmptyBorder(25, 30, 25, 30));

        initHeader();
        initFilterBar();
        initTablePanel();
    }

    private void initHeader() {
        JPanel header = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        header.setOpaque(false);
        header.setBorder(BorderFactory.createEmptyBorder(0, 0, 15, 0));

        JLabel title = new JLabel("Attendance Records");
        title.setFont(Theme.FONT_TITLE_LARGE);
        title.setForeground(Theme.TEXT_DARK);
        header.add(title);

        add(header, BorderLayout.NORTH);
    }

    private void initFilterBar() {
        JPanel filterPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 10)) {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Color.WHITE);
                g2.fillRoundRect(0, 0, getWidth(), getHeight(), 12, 12);
                g2.dispose();
            }
        };
        filterPanel.setOpaque(false);
        filterPanel.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));

        // Date input
        JLabel dateLbl = new JLabel("Date (yyyy-MM-dd)");
        dateLbl.setFont(Theme.FONT_CAPTION);
        dateLbl.setForeground(Theme.TEXT_LIGHT);
        filterPanel.add(dateLbl);

        String today = new SimpleDateFormat("yyyy-MM-dd").format(new Date());
        dateField = new ModernTextField(today, 10);
        filterPanel.add(dateField);

        // Dept Input
        JLabel deptLbl = new JLabel("Department");
        deptLbl.setFont(Theme.FONT_CAPTION);
        deptLbl.setForeground(Theme.TEXT_LIGHT);
        filterPanel.add(deptLbl);

        String[] depts = {"All Departments", "Computer Science", "Information Tech", "Electronics", "Mechanical", "Civil"};
        deptCombo = new JComboBox<>(depts);
        deptCombo.setFont(Theme.FONT_BODY);
        deptCombo.setBackground(Color.WHITE);
        filterPanel.add(deptCombo);

        // Search Input
        searchField = new ModernTextField("Search Name/Roll...", 12);
        filterPanel.add(searchField);

        // Buttons
        ModernButton searchBtn = new ModernButton("Search");
        searchBtn.setPreferredSize(new Dimension(100, 36));
        searchBtn.addActionListener(e -> performFilter());
        filterPanel.add(searchBtn);

        ModernButton exportBtn = new ModernButton("Export", Theme.COLOR_GREEN, Theme.COLOR_GREEN.darker());
        exportBtn.setPreferredSize(new Dimension(100, 36));
        exportBtn.addActionListener(e -> handleExportCSV());
        filterPanel.add(exportBtn);

        add(filterPanel, BorderLayout.NORTH);
    }

    private void initTablePanel() {
        ModernPanel tablePanel = new ModernPanel(Theme.ROUNDNESS_CARD, Color.WHITE);
        tablePanel.setLayout(new BorderLayout());
        tablePanel.setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));

        String[] cols = {"Name", "Roll No.", "Department", "Time", "Status"};
        tableModel = new DefaultTableModel(cols, 0) {
            @Override
            public boolean isCellEditable(int r, int c) { return false; }
        };
        table = new JTable(tableModel);

        tablePanel.add(ModernComponents.createModernTable(table), BorderLayout.CENTER);

        // Add padding spacing
        JPanel container = new JPanel(new BorderLayout());
        container.setOpaque(false);
        container.setBorder(BorderFactory.createEmptyBorder(15, 0, 0, 0));
        container.add(tablePanel, BorderLayout.CENTER);

        add(container, BorderLayout.CENTER);
    }

    public void refreshTable() {
        allRecordsCache.clear();

        // 1. Read all registered student profiles
        Map<String, String[]> students = new HashMap<>(); // Name -> [Roll, Dept]
        try (BufferedReader br = new BufferedReader(new FileReader("students.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length >= 3) {
                    students.put(p[1], new String[]{p[0], p[2]}); // Name -> [Roll, Dept]
                }
            }
        } catch (Exception ex) {}

        // 2. Read checkins for target date
        String filterDate = dateField.getText().trim();
        Map<String, String> checkins = new HashMap<>(); // Name -> Time
        try (BufferedReader br = new BufferedReader(new FileReader("attendance.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length >= 2) {
                    String name = p[0];
                    String timestamp = p[1]; // yyyy-MM-dd HH:mm:ss
                    if (timestamp.startsWith(filterDate)) {
                        String timeStr = timestamp.substring(11);
                        checkins.put(name, timeStr);
                    }
                }
            }
        } catch (Exception ex) {}

        // Combine
        for (String name : students.keySet()) {
            String[] det = students.get(name);
            String roll = det[0];
            String dept = det[1];

            String time = "--:--";
            String status = "Absent";
            if (checkins.containsKey(name)) {
                time = checkins.get(name);
                status = "Present";
            }

            allRecordsCache.add(new Object[]{name, roll, dept, time, status});
        }

        // Apply visual display sorting (Present first, then Absent)
        allRecordsCache.sort((a, b) -> {
            String s1 = (String) a[4];
            String s2 = (String) b[4];
            return s2.compareTo(s1); // "Present" comes before "Absent"
        });

        performFilter();
    }

    private void performFilter() {
        tableModel.setRowCount(0);

        String deptFilter = (String) deptCombo.getSelectedItem();
        String query = searchField.getText().trim().toLowerCase();

        for (Object[] row : allRecordsCache) {
            String name = ((String) row[0]).toLowerCase();
            String roll = ((String) row[1]).toLowerCase();
            String dept = (String) row[2];

            // Department filter
            if (!deptFilter.equals("All Departments") && !dept.equals(deptFilter)) {
                continue;
            }

            // Search query filter
            if (!query.isEmpty() && !name.contains(query) && !roll.contains(query) && !query.equals("search name/roll...")) {
                continue;
            }

            tableModel.addRow(row);
        }
    }

    private void handleExportCSV() {
        if (tableModel.getRowCount() == 0) {
            JOptionPane.showMessageDialog(this, "⚠️ No records to export!", "Export Empty", JOptionPane.WARNING_MESSAGE);
            return;
        }

        String filterDate = dateField.getText().trim();
        String exportPath = "attendance_export_" + filterDate + ".csv";

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(exportPath))) {
            // Write CSV Header
            writer.write("Student Name,Roll Number,Department,Time,Status");
            writer.newLine();

            // Write Rows
            for (int i = 0; i < tableModel.getRowCount(); i++) {
                StringBuilder row = new StringBuilder();
                for (int j = 0; j < tableModel.getColumnCount(); j++) {
                    if (j > 0) row.append(",");
                    row.append(tableModel.getValueAt(i, j).toString());
                }
                writer.write(row.toString());
                writer.newLine();
            }

            JOptionPane.showMessageDialog(this, "✅ Attendance records exported successfully to:\n" + exportPath, "Export Complete", JOptionPane.INFORMATION_MESSAGE);
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "❌ Error writing CSV export file: " + ex.getMessage(), "Export Error", JOptionPane.ERROR_MESSAGE);
        }
    }
}
