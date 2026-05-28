package gui;

import java.awt.*;
import java.awt.geom.*;
import java.awt.image.BufferedImage;
import javax.swing.*;
import javax.swing.border.AbstractBorder;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.JTableHeader;

public class ModernComponents {

    // 1. Modern Panel with Rounded Corners and Drop Shadow
    public static class ModernPanel extends JPanel {
        private int roundness = Theme.ROUNDNESS_CARD;
        private Color bgColor = Theme.CARD_BG;
        private Color gradientEnd = null;

        public ModernPanel() {
            setOpaque(false);
        }

        public ModernPanel(int roundness, Color bgColor) {
            this();
            this.roundness = roundness;
            this.bgColor = bgColor;
        }

        public ModernPanel(int roundness, Color startColor, Color endColor) {
            this(roundness, startColor);
            this.gradientEnd = endColor;
        }

        public void setBgColor(Color color) {
            this.bgColor = color;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth();
            int h = getHeight();

            // Fill background with optional gradient
            if (gradientEnd != null) {
                GradientPaint gp = new GradientPaint(0, 0, bgColor, 0, h, gradientEnd);
                g2.setPaint(gp);
            } else {
                g2.setColor(bgColor);
            }

            g2.fillRoundRect(0, 0, w - 1, h - 1, roundness, roundness);
            g2.dispose();
        }
    }

    // 2. Glassmorphic Panel (Translucent card effect)
    public static class GlassPanel extends JPanel {
        private int roundness = Theme.ROUNDNESS_CARD;
        private float opacity = 0.08f; // Alpha transparency

        public GlassPanel() {
            setOpaque(false);
        }

        public GlassPanel(int roundness, float opacity) {
            this();
            this.roundness = roundness;
            this.opacity = opacity;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth();
            int h = getHeight();

            // Paint translucent glass backing
            g2.setColor(new Color(255, 255, 255, (int) (opacity * 255)));
            g2.fillRoundRect(0, 0, w - 1, h - 1, roundness, roundness);

            // Paint elegant thin white border
            g2.setColor(new Color(255, 255, 255, 30));
            g2.setStroke(new BasicStroke(1.0f));
            g2.drawRoundRect(0, 0, w - 1, h - 1, roundness, roundness);

            g2.dispose();
        }
    }

    // 3. Modern Button with Rounded Corners, Hover effects and Gradients
    public static class ModernButton extends JButton {
        private Color startColor = Theme.ACCENT_GRADIENT_START;
        private Color endColor = Theme.ACCENT_GRADIENT_END;
        private boolean isHovered = false;

        public ModernButton(String text) {
            super(text);
            setContentAreaFilled(false);
            setFocusPainted(false);
            setBorderPainted(false);
            setForeground(Theme.TEXT_WHITE);
            setFont(Theme.FONT_BODY_BOLD);
            setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));

            addMouseListener(new java.awt.event.MouseAdapter() {
                public void mouseEntered(java.awt.event.MouseEvent e) {
                    isHovered = true;
                    repaint();
                }
                public void mouseExited(java.awt.event.MouseEvent e) {
                    isHovered = false;
                    repaint();
                }
            });
        }

        public ModernButton(String text, Color startColor, Color endColor) {
            this(text);
            this.startColor = startColor;
            this.endColor = endColor;
        }

        public void setColors(Color start, Color end) {
            this.startColor = start;
            this.endColor = end;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth();
            int h = getHeight();

            Color sCol = isHovered ? startColor.brighter() : startColor;
            Color eCol = isHovered ? endColor.brighter() : endColor;

            GradientPaint gp = new GradientPaint(0, 0, sCol, w, 0, eCol);
            g2.setPaint(gp);

            g2.fillRoundRect(0, 0, w, h, Theme.ROUNDNESS_BUTTON, Theme.ROUNDNESS_BUTTON);

            // Print Text
            FontMetrics fm = g2.getFontMetrics();
            Rectangle2D r = fm.getStringBounds(getText(), g2);
            int x = (w - (int) r.getWidth()) / 2;
            int y = (h - (int) r.getHeight()) / 2 + fm.getAscent();

            g2.setColor(getForeground());
            g2.setFont(getFont());
            g2.drawString(getText(), x, y);

            g2.dispose();
        }
    }

    // 4. Modern Input Text Field
    public static class ModernTextField extends JTextField {
        private String placeholder = "";

        public ModernTextField(int columns) {
            super(columns);
            setOpaque(false);
            setFont(Theme.FONT_BODY);
            setForeground(Theme.TEXT_DARK);
            setCaretColor(Theme.TEXT_DARK);
            setMargin(new Insets(6, 12, 6, 12));
            setBorder(new RoundedCornerBorder(Theme.ROUNDNESS_FIELD));
        }

        public ModernTextField(String placeholder, int columns) {
            this(columns);
            this.placeholder = placeholder;
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Paint background card shape
            g2.setColor(Color.WHITE);
            g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, Theme.ROUNDNESS_FIELD, Theme.ROUNDNESS_FIELD);
            g2.dispose();

            super.paintComponent(g);

            // Draw placeholder
            if (getText().isEmpty() && !placeholder.isEmpty()) {
                Graphics2D gPlaceholder = (Graphics2D) g.create();
                gPlaceholder.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                gPlaceholder.setColor(Theme.TEXT_LIGHT);
                gPlaceholder.setFont(getFont());
                Insets insets = getInsets();
                FontMetrics fm = gPlaceholder.getFontMetrics();
                int y = (getHeight() - fm.getHeight()) / 2 + fm.getAscent();
                gPlaceholder.drawString(placeholder, insets.left, y);
                gPlaceholder.dispose();
            }
        }
    }

    // 5. Modern Input Password Field
    public static class ModernPasswordField extends JPasswordField {
        private String placeholder = "";

        public ModernPasswordField(int columns) {
            super(columns);
            setOpaque(false);
            setFont(Theme.FONT_BODY);
            setForeground(Theme.TEXT_DARK);
            setCaretColor(Theme.TEXT_DARK);
            setMargin(new Insets(6, 12, 6, 12));
            setBorder(new RoundedCornerBorder(Theme.ROUNDNESS_FIELD));
        }

        public ModernPasswordField(String placeholder, int columns) {
            this(columns);
            this.placeholder = placeholder;
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            g2.setColor(Color.WHITE);
            g2.fillRoundRect(0, 0, getWidth() - 1, getHeight() - 1, Theme.ROUNDNESS_FIELD, Theme.ROUNDNESS_FIELD);
            g2.dispose();

            super.paintComponent(g);

            if (getPassword().length == 0 && !placeholder.isEmpty()) {
                Graphics2D gPlaceholder = (Graphics2D) g.create();
                gPlaceholder.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                gPlaceholder.setColor(Theme.TEXT_LIGHT);
                gPlaceholder.setFont(getFont());
                Insets insets = getInsets();
                FontMetrics fm = gPlaceholder.getFontMetrics();
                int y = (getHeight() - fm.getHeight()) / 2 + fm.getAscent();
                gPlaceholder.drawString(placeholder, insets.left, y);
                gPlaceholder.dispose();
            }
        }
    }

    // Border Helper class for text inputs
    private static class RoundedCornerBorder extends AbstractBorder {
        private final int radius;

        public RoundedCornerBorder(int radius) {
            this.radius = radius;
        }

        @Override
        public void paintBorder(Component c, Graphics g, int x, int y, int width, int height) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(new Color(226, 232, 240)); // Border #E2E8F0
            g2.setStroke(new BasicStroke(1.2f));
            g2.drawRoundRect(x, y, width - 1, height - 1, radius, radius);
            g2.dispose();
        }

        @Override
        public Insets getBorderInsets(Component c) {
            return new Insets(8, 14, 8, 14);
        }
    }

    // 6. Modern Table Styling Wrapper
    public static JScrollPane createModernTable(JTable table) {
        table.setFont(Theme.FONT_BODY);
        table.setForeground(Theme.TEXT_DARK);
        table.setRowHeight(38);
        table.setShowGrid(false);
        table.setIntercellSpacing(new Dimension(0, 0));
        table.setSelectionBackground(new Color(59, 130, 246, 30)); // 30alpha ACCENT_BLUE
        table.setSelectionForeground(Theme.TEXT_DARK);

        // Header Styling
        JTableHeader header = table.getTableHeader();
        header.setFont(Theme.FONT_BODY_BOLD);
        header.setBackground(new Color(241, 245, 249)); // Grey header background (#F1F5F9)
        header.setForeground(Theme.TEXT_LIGHT);
        header.setPreferredSize(new Dimension(100, 42));
        ((DefaultTableCellRenderer) header.getDefaultRenderer()).setHorizontalAlignment(JLabel.LEFT);

        // Grid/Cell Alignment and Padding Renderer
        DefaultTableCellRenderer cellRenderer = new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable t, Object val, boolean isSel, boolean hasF, int r, int c) {
                super.getTableCellRendererComponent(t, val, isSel, hasF, r, c);
                setBorder(BorderFactory.createEmptyBorder(0, 12, 0, 12));
                
                // Colorize Status column specially
                if (c == t.getColumnCount() - 1) { // Status column
                    String valStr = val != null ? val.toString() : "";
                    if (valStr.equalsIgnoreCase("Present")) {
                        setForeground(Theme.COLOR_GREEN);
                        setFont(Theme.FONT_BODY_BOLD);
                    } else if (valStr.equalsIgnoreCase("Absent")) {
                        setForeground(Theme.COLOR_RED);
                        setFont(Theme.FONT_BODY_BOLD);
                    } else {
                        setForeground(Theme.TEXT_DARK);
                    }
                } else {
                    setForeground(Theme.TEXT_DARK);
                }
                
                // Zebra striping
                if (!isSel) {
                    setBackground(r % 2 == 0 ? Color.WHITE : new Color(250, 250, 250));
                }
                return this;
            }
        };

        for (int i = 0; i < table.getColumnCount(); i++) {
            table.getColumnModel().getColumn(i).setCellRenderer(cellRenderer);
        }

        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.setBorder(BorderFactory.createEmptyBorder());
        scrollPane.getViewport().setBackground(Color.WHITE);
        return scrollPane;
    }

    // 7. Webcam Viewer Canvas Panel
    public static class WebcamPanel extends JPanel {
        private BufferedImage image = null;
        private String overlayText = "Camera Offline";
        private boolean showFrameOutline = false;

        public WebcamPanel() {
            setBackground(new Color(15, 23, 42)); // Dark Slate
            setOpaque(true);
        }

        public synchronized void setImage(BufferedImage img) {
            this.image = img;
            this.overlayText = "";
            repaint();
        }

        public synchronized void clearFeed(String statusText) {
            this.image = null;
            this.overlayText = statusText;
            repaint();
        }

        public void setShowFrameOutline(boolean show) {
            this.showFrameOutline = show;
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth();
            int h = getHeight();

            synchronized (this) {
                if (image != null) {
                    // Draw webcam image maintaining aspect ratio
                    double imgW = image.getWidth();
                    double imgH = image.getHeight();
                    double scale = Math.max(w / imgW, h / imgH);
                    int nw = (int) (imgW * scale);
                    int nh = (int) (imgH * scale);
                    int x = (w - nw) / 2;
                    int y = (h - nh) / 2;

                    g2.drawImage(image, x, y, nw, nh, null);

                    // Draw green scanning face alignment boundaries if requested
                    if (showFrameOutline) {
                        g2.setColor(Theme.ACCENT_BLUE);
                        g2.setStroke(new BasicStroke(2.0f));
                        // Bounding scanning corners
                        int offset = 60;
                        g2.drawRoundRect(offset, offset, w - (offset * 2), h - (offset * 2), 24, 24);

                        g2.setColor(new Color(255, 255, 255, 120));
                        g2.setFont(Theme.FONT_CAPTION);
                        String info = "ALIGN FACE WITHIN SCANNER";
                        FontMetrics fm = g2.getFontMetrics();
                        g2.drawString(info, (w - fm.stringWidth(info)) / 2, offset - 15);
                    }
                } else {
                    // Draw placeholder background with neon scanning visualizer
                    g2.setColor(new Color(30, 41, 59));
                    g2.fillRect(0, 0, w, h);

                    g2.setColor(new Color(255, 255, 255, 100));
                    g2.setFont(Theme.FONT_SUBTITLE);
                    FontMetrics fm = g2.getFontMetrics();
                    int x = (w - fm.stringWidth(overlayText)) / 2;
                    int y = (h - fm.getHeight()) / 2 + fm.getAscent();
                    g2.drawString(overlayText, x, y);
                }
            }
            g2.dispose();
        }
    }
}
