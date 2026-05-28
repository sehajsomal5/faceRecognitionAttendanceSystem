package gui;

import java.awt.Color;
import java.awt.Font;
import javax.swing.BorderFactory;
import javax.swing.border.Border;

public class Theme {
    // Futuristic Theme Colors
    public static final Color SIDEBAR_BG = new Color(15, 23, 42);       // Dark Slate Blue (#0F172A)
    public static final Color SIDEBAR_HOVER = new Color(30, 41, 59);    // Slightly Lighter Slate (#1E293B)
    public static final Color ACCENT_BLUE = new Color(59, 130, 246);     // Vivid Blue (#3B82F6)
    public static final Color ACCENT_GRADIENT_START = new Color(79, 70, 229); // Royal Purple-Blue (#4F46E5)
    public static final Color ACCENT_GRADIENT_END = new Color(37, 99, 235);   // Indigo Blue (#2563EB)
    
    public static final Color MAIN_BG = new Color(248, 250, 252);        // Off-White/Light Grey (#F8FAFC)
    public static final Color CARD_BG = Color.WHITE;
    public static final Color TEXT_DARK = new Color(15, 23, 42);        // Near Black
    public static final Color TEXT_LIGHT = new Color(100, 116, 139);    // Slate Grey (#64748B)
    public static final Color TEXT_WHITE = Color.WHITE;
    
    public static final Color COLOR_GREEN = new Color(34, 197, 94);     // Emerald (#22C55E)
    public static final Color COLOR_RED = new Color(239, 68, 68);       // Coral Red (#EF4444)
    public static final Color COLOR_AMBER = new Color(245, 158, 11);    // Amber/Orange (#F59E0B)

    // Modern Fonts
    public static final Font FONT_TITLE_LARGE = new Font("Segoe UI", Font.BOLD, 26);
    public static final Font FONT_TITLE = new Font("Segoe UI", Font.BOLD, 20);
    public static final Font FONT_SUBTITLE = new Font("Segoe UI", Font.BOLD, 15);
    public static final Font FONT_BODY = new Font("Segoe UI", Font.PLAIN, 13);
    public static final Font FONT_BODY_BOLD = new Font("Segoe UI", Font.BOLD, 13);
    public static final Font FONT_CAPTION = new Font("Segoe UI", Font.PLAIN, 11);
    public static final Font FONT_CLOCK = new Font("Segoe UI", Font.BOLD, 14);

    // Rounded Border styling constants
    public static final Border BORDER_CARD = BorderFactory.createLineBorder(new Color(226, 232, 240), 1); // Border #E2E8F0
    public static final int ROUNDNESS_CARD = 16;
    public static final int ROUNDNESS_BUTTON = 12;
    public static final int ROUNDNESS_FIELD = 10;
}
