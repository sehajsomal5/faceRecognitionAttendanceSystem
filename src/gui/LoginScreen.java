package gui;

import java.awt.*;
import javax.swing.*;
import gui.ModernComponents.*;

public class LoginScreen extends JPanel {
    private MainFrame mainFrame;
    private ModernTextField usernameField;
    private ModernPasswordField passwordField;
    private JCheckBox rememberMeCheckbox;
    private JLabel errorLabel;

    public LoginScreen(MainFrame frame) {
        this.mainFrame = frame;
        setLayout(new GridLayout(1, 2));
        setBackground(Theme.MAIN_BG);

        initLeftBrandPanel();
        initRightLoginPanel();
    }

    private void initLeftBrandPanel() {
        ModernPanel leftPanel = new ModernPanel(0, Theme.SIDEBAR_BG);
        leftPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = GridBagConstraints.RELATIVE;
        gbc.insets = new Insets(10, 30, 10, 30);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.CENTER;

        // Face recognition modern icon placeholder
        JLabel logoIconLabel = new JLabel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Theme.ACCENT_BLUE);
                g2.setStroke(new BasicStroke(3.0f));
                
                // Draw a modern face scanning bracket icon
                int w = getWidth();
                int h = getHeight();
                int size = 60;
                int x = (w - size) / 2;
                int y = (h - size) / 2;
                
                // Scanning corners
                int len = 15;
                g2.drawArc(x, y, len * 2, len * 2, 90, 90);           // Top Left
                g2.drawArc(x + size - len * 2, y, len * 2, len * 2, 0, 90); // Top Right
                g2.drawArc(x, y + size - len * 2, len * 2, len * 2, 180, 90); // Bottom Left
                g2.drawArc(x + size - len * 2, y + size - len * 2, len * 2, len * 2, 270, 90); // Bottom Right
                
                // Face oval
                g2.drawOval(x + 18, y + 12, 24, 28);
                // Eyes
                g2.fillOval(x + 23, y + 20, 4, 4);
                g2.fillOval(x + 33, y + 20, 4, 4);
                // Smile
                g2.drawArc(x + 25, y + 26, 10, 8, 180, 180);
                
                g2.dispose();
            }
        };
        logoIconLabel.setPreferredSize(new Dimension(150, 150));
        leftPanel.add(logoIconLabel, gbc);

        // App Title
        JLabel titleLabel = new JLabel("Face Recognition", JLabel.CENTER);
        titleLabel.setFont(Theme.FONT_TITLE_LARGE);
        titleLabel.setForeground(Theme.TEXT_WHITE);
        leftPanel.add(titleLabel, gbc);

        JLabel subTitleLabel = new JLabel("Attendance System", JLabel.CENTER);
        subTitleLabel.setFont(Theme.FONT_TITLE);
        subTitleLabel.setForeground(Theme.TEXT_WHITE);
        leftPanel.add(subTitleLabel, gbc);

        // Slogan
        JLabel sloganLabel = new JLabel("Smart. Secure. Automated.", JLabel.CENTER);
        sloganLabel.setFont(Theme.FONT_SUBTITLE);
        sloganLabel.setForeground(Theme.ACCENT_BLUE);
        gbc.insets = new Insets(30, 30, 10, 30);
        leftPanel.add(sloganLabel, gbc);

        add(leftPanel);
    }

    private void initRightLoginPanel() {
        JPanel rightContainer = new JPanel(new GridBagLayout());
        rightContainer.setBackground(Color.WHITE);

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = GridBagConstraints.RELATIVE;
        gbc.insets = new Insets(8, 50, 8, 50);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;

        // Welcome titles
        JLabel welcomeLabel = new JLabel("Welcome Back!");
        welcomeLabel.setFont(Theme.FONT_TITLE_LARGE);
        welcomeLabel.setForeground(Theme.TEXT_DARK);
        rightContainer.add(welcomeLabel, gbc);

        JLabel subWelcomeLabel = new JLabel("Login to continue");
        subWelcomeLabel.setFont(Theme.FONT_BODY);
        subWelcomeLabel.setForeground(Theme.TEXT_LIGHT);
        gbc.insets = new Insets(2, 50, 20, 50);
        rightContainer.add(subWelcomeLabel, gbc);

        // Username Field
        JLabel userLabel = new JLabel("Username");
        userLabel.setFont(Theme.FONT_BODY_BOLD);
        userLabel.setForeground(Theme.TEXT_DARK);
        gbc.insets = new Insets(10, 50, 4, 50);
        rightContainer.add(userLabel, gbc);

        usernameField = new ModernTextField("Enter username", 24);
        gbc.insets = new Insets(2, 50, 10, 50);
        rightContainer.add(usernameField, gbc);

        // Password Field
        JLabel passLabel = new JLabel("Password");
        passLabel.setFont(Theme.FONT_BODY_BOLD);
        passLabel.setForeground(Theme.TEXT_DARK);
        gbc.insets = new Insets(10, 50, 4, 50);
        rightContainer.add(passLabel, gbc);

        passwordField = new ModernPasswordField("Enter password", 24);
        gbc.insets = new Insets(2, 50, 15, 50);
        rightContainer.add(passwordField, gbc);

        // Error message label
        errorLabel = new JLabel("");
        errorLabel.setFont(Theme.FONT_CAPTION);
        errorLabel.setForeground(Theme.COLOR_RED);
        gbc.insets = new Insets(0, 50, 4, 50);
        rightContainer.add(errorLabel, gbc);

        // Remember me & Forgot Password
        JPanel optionsPanel = new JPanel(new BorderLayout());
        optionsPanel.setOpaque(false);
        rememberMeCheckbox = new JCheckBox("Remember me");
        rememberMeCheckbox.setFont(Theme.FONT_CAPTION);
        rememberMeCheckbox.setForeground(Theme.TEXT_LIGHT);
        rememberMeCheckbox.setOpaque(false);
        rememberMeCheckbox.setFocusPainted(false);
        
        JLabel forgotPasswordLabel = new JLabel("Forgot Password?");
        forgotPasswordLabel.setFont(Theme.FONT_CAPTION);
        forgotPasswordLabel.setForeground(Theme.ACCENT_BLUE);
        forgotPasswordLabel.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));

        optionsPanel.add(rememberMeCheckbox, BorderLayout.WEST);
        optionsPanel.add(forgotPasswordLabel, BorderLayout.EAST);
        gbc.insets = new Insets(4, 50, 20, 50);
        rightContainer.add(optionsPanel, gbc);

        // Login Button
        ModernButton loginButton = new ModernButton("Login");
        loginButton.setPreferredSize(new Dimension(100, 42));
        loginButton.addActionListener(e -> handleLogin());
        gbc.insets = new Insets(10, 50, 30, 50);
        rightContainer.add(loginButton, gbc);

        // Bottom footer
        JLabel footerLabel = new JLabel("© 2026 Face Recognition Attendance System", JLabel.CENTER);
        footerLabel.setFont(Theme.FONT_CAPTION);
        footerLabel.setForeground(Theme.TEXT_LIGHT);
        gbc.insets = new Insets(10, 50, 10, 50);
        gbc.anchor = GridBagConstraints.CENTER;
        rightContainer.add(footerLabel, gbc);

        add(rightContainer);
    }

    private void handleLogin() {
        String username = usernameField.getText().trim();
        String password = new String(passwordField.getPassword());

        // Simple hardcoded login for demo/mockup purposes
        if (username.equals("admin") && password.equals("admin")) {
            errorLabel.setText("");
            mainFrame.switchScreen("Dashboard");
        } else {
            errorLabel.setText("❌ Invalid username or password (use admin/admin)");
        }
    }
}
