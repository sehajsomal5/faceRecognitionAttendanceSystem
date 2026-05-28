package gui;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class SidebarPanel extends JPanel {
    private MainFrame mainFrame;
    private AppWorkspacePanel workspacePanel;
    private JButton[] menuButtons;
    private String[] screenNames = {
        "Dashboard", "RegisterStudent", "TrainFaces", 
        "MarkAttendance", "ViewAttendance", "Reports", "Settings"
    };
    private int activeIndex = 0;

    public SidebarPanel(MainFrame mainFrame, AppWorkspacePanel workspacePanel) {
        this.mainFrame = mainFrame;
        this.workspacePanel = workspacePanel;
        
        setPreferredSize(new Dimension(250, 800));
        setBackground(Theme.SIDEBAR_BG);
        setLayout(new BorderLayout());

        initHeader();
        initMenu();
        initFooter();
    }

    private void initHeader() {
        JPanel headerPanel = new JPanel(new GridBagLayout());
        headerPanel.setOpaque(false);
        headerPanel.setBorder(BorderFactory.createEmptyBorder(30, 20, 20, 20));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = GridBagConstraints.RELATIVE;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.insets = new Insets(4, 0, 4, 0);

        // Subtitle Icon
        JLabel logoLabel = new JLabel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Theme.ACCENT_BLUE);
                g2.setStroke(new BasicStroke(2.0f));
                
                int w = getWidth();
                int h = getHeight();
                int x = 2;
                int y = 2;
                int size = 26;
                g2.drawOval(x, y, size, size);
                g2.drawOval(x + 7, y + 5, 12, 14);
                g2.fillOval(x + 10, y + 10, 2, 2);
                g2.fillOval(x + 14, y + 10, 2, 2);
                g2.drawArc(x + 11, y + 13, 4, 3, 180, 180);
                g2.dispose();
            }
        };
        logoLabel.setPreferredSize(new Dimension(32, 32));

        JLabel titleLabel = new JLabel("Face Recognition");
        titleLabel.setFont(Theme.FONT_SUBTITLE);
        titleLabel.setForeground(Theme.TEXT_WHITE);

        JLabel subtitleLabel = new JLabel("Attendance System");
        subtitleLabel.setFont(Theme.FONT_CAPTION);
        subtitleLabel.setForeground(Theme.ACCENT_BLUE);

        JPanel branding = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 0));
        branding.setOpaque(false);
        branding.add(logoLabel);
        branding.add(titleLabel);

        headerPanel.add(branding, gbc);
        
        gbc.insets = new Insets(2, 42, 10, 0);
        headerPanel.add(subtitleLabel, gbc);

        add(headerPanel, BorderLayout.NORTH);
    }

    private void initMenu() {
        JPanel menuPanel = new JPanel();
        menuPanel.setOpaque(false);
        menuPanel.setLayout(new BoxLayout(menuPanel, BoxLayout.Y_AXIS));
        menuPanel.setBorder(BorderFactory.createEmptyBorder(20, 15, 20, 15));

        String[] menuLabels = {
            "Dashboard", "Register Student", "Train Faces", 
            "Mark Attendance", "View Attendance", "Reports", "Settings"
        };
        
        // Native Unicode icons that render beautifully on macOS
        String[] menuIcons = {
            "▤ ", "👤 ", "⚙ ", "🎥 ", "📅 ", "📊 ", "🛠 "
        };

        menuButtons = new JButton[menuLabels.length];

        for (int i = 0; i < menuLabels.length; i++) {
            final int index = i;
            JButton btn = new JButton(menuIcons[i] + "  " + menuLabels[index]) {
                @Override
                protected void paintComponent(Graphics g) {
                    Graphics2D g2 = (Graphics2D) g.create();
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    
                    int w = getWidth();
                    int h = getHeight();
                    
                    if (index == activeIndex) {
                        // Paint highlighted active button background
                        g2.setColor(Theme.ACCENT_BLUE);
                        g2.fillRoundRect(0, 0, w, h, 10, 10);
                    } else if (getModel().isRollover()) {
                        g2.setColor(Theme.SIDEBAR_HOVER);
                        g2.fillRoundRect(0, 0, w, h, 10, 10);
                    }
                    g2.dispose();
                    super.paintComponent(g);
                }
            };

            btn.setFont(Theme.FONT_BODY);
            btn.setForeground(index == activeIndex ? Theme.TEXT_WHITE : Theme.TEXT_LIGHT);
            btn.setOpaque(false);
            btn.setContentAreaFilled(false);
            btn.setFocusPainted(false);
            btn.setBorderPainted(false);
            btn.setHorizontalAlignment(SwingConstants.LEFT);
            btn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
            btn.setBorder(BorderFactory.createEmptyBorder(10, 15, 10, 15));
            btn.setMaximumSize(new Dimension(220, 40));
            btn.setPreferredSize(new Dimension(220, 40));

            btn.addActionListener(e -> selectMenuItem(index));

            menuButtons[i] = btn;
            menuPanel.add(btn);
            menuPanel.add(Box.createVerticalStrut(10));
        }

        add(menuPanel, BorderLayout.CENTER);
    }

    private void initFooter() {
        JPanel footerPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 15));
        footerPanel.setOpaque(false);
        footerPanel.setBorder(BorderFactory.createEmptyBorder(0, 15, 20, 15));

        JButton logoutBtn = new JButton("🚪   Logout") {
            @Override
            protected void paintComponent(Graphics g) {
                if (getModel().isRollover()) {
                    Graphics2D g2 = (Graphics2D) g.create();
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    g2.setColor(Theme.COLOR_RED);
                    g2.fillRoundRect(0, 0, getWidth(), getHeight(), 10, 10);
                    g2.dispose();
                }
                super.paintComponent(g);
            }
        };
        logoutBtn.setFont(Theme.FONT_BODY_BOLD);
        logoutBtn.setForeground(Theme.TEXT_LIGHT);
        logoutBtn.setOpaque(false);
        logoutBtn.setContentAreaFilled(false);
        logoutBtn.setFocusPainted(false);
        logoutBtn.setBorderPainted(false);
        logoutBtn.setHorizontalAlignment(SwingConstants.LEFT);
        logoutBtn.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        logoutBtn.setBorder(BorderFactory.createEmptyBorder(10, 15, 10, 15));
        logoutBtn.setMaximumSize(new Dimension(220, 40));
        logoutBtn.setPreferredSize(new Dimension(220, 40));

        logoutBtn.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseEntered(MouseEvent e) {
                logoutBtn.setForeground(Theme.TEXT_WHITE);
            }
            @Override
            public void mouseExited(MouseEvent e) {
                logoutBtn.setForeground(Theme.TEXT_LIGHT);
            }
        });

        logoutBtn.addActionListener(e -> {
            mainFrame.stopCameraFeed();
            mainFrame.switchScreen("Login");
        });

        footerPanel.add(logoutBtn);
        add(footerPanel, BorderLayout.SOUTH);
    }

    public void selectMenuItem(int index) {
        if (index == activeIndex) return;
        
        // Specially handle "Train Faces" directly
        if (screenNames[index].equals("TrainFaces")) {
            workspacePanel.triggerDatasetTraining();
            return;
        }

        // Deactivate previous active item
        menuButtons[activeIndex].setForeground(Theme.TEXT_LIGHT);
        
        activeIndex = index;
        
        // Activate new item
        menuButtons[activeIndex].setForeground(Theme.TEXT_WHITE);
        repaint();

        // Switch corresponding screen workspace card
        workspacePanel.showWorkspaceCard(screenNames[activeIndex]);
    }

    public void selectMenuItemByCardName(String cardName) {
        for (int i = 0; i < screenNames.length; i++) {
            if (screenNames[i].equalsIgnoreCase(cardName)) {
                selectMenuItem(i);
                break;
            }
        }
    }
}
