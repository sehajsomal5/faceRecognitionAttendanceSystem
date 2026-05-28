# Futuristic Face Recognition Attendance System

A modern, desktop-based Face Recognition Attendance System built in Java Swing and powered by OpenCV. It features a sleek, dark-blue glassmorphic UI design, real-time webcam video stream rendering, and an embedded K-Nearest Neighbors (KNN) machine learning classifier.

---

## 🚀 Key Features

*   **Glassmorphic Design System**: Sleek cards, glowing buttons with smooth hover states, custom typography, and modern form elements.
*   **Split-Pane Admin Login**: Secure authentication gateway panel.
*   **Interactive Dashboard**: Real-time summary cards (student counts, system statuses), a live-updating clock widget, and a rolling log of today's check-ins.
*   **Webcam Face Capturer**: Automatically snaps, crops, and processes 30 face frames in a background worker thread for training new students.
*   **Real-Time Scanner**: Renders boundary scanning boxes directly onto the live 30 FPS video feed and performs instant matching against the local dataset.
*   **Zebra-Striped Records Log**: A filterable database grid displaying check-in times and statuses, equipped with search and a one-click CSV spreadsheet exporter.
*   **Reports & Visual Analytics**: Custom-drawn, anti-aliased charts (Donut Chart for attendance rate, Bar Chart for department metrics) without relying on heavy third-party plotting libraries.
*   **Cross-Platform Native Linking**: Uses a dynamic classloader to auto-detect and load OpenCV native binaries (`.dll` on Windows, `.dylib` on macOS Intel/Apple Silicon, and `.so` on Linux).

---

## 🔑 Default Credentials

To log into the system:
*   **Username**: `admin`
*   **Password**: `admin`

---

## 🛠️ Installation & Setup

### Prerequisites
*   **Java JDK 17 or higher** installed.
*   A connected **webcam** (built-in or USB).

---

## 💻 Running the Application

Open your terminal or command prompt in the root of the project directory and follow the instructions for your operating system:

### 🍏 macOS & Linux

1.  **Compile the source files**:
    ```bash
    mkdir -p bin
    javac -d bin -cp lib/opencv-4.9.0-0.jar src/*.java src/gui/*.java
    ```

2.  **Run the application**:
    ```bash
    java -cp bin:lib/opencv-4.9.0-0.jar gui.MainFrame
    ```

*Note: On macOS, make sure to grant **Camera permissions** to your Terminal or IDE (VS Code) under System Settings > Privacy & Security > Camera.*

---

### 🔌 Windows

1.  **Compile the source files**:
    ```cmd
    mkdir bin
    javac -d bin -cp "lib/opencv-4.9.0-0.jar" src/*.java src/gui/*.java
    ```

2.  **Run the application**:
    ```cmd
    java -cp "bin;lib/opencv-4.9.0-0.jar" gui.MainFrame
    ```

---

## 📂 Project Structure

```
├── dataset/                  # Saved face training folders (Name/1.jpg...30.jpg)
├── lib/
│   └── opencv-4.9.0-0.jar    # Platform-independent OpenCV bindings
├── resources/
│   ├── haarcascade_frontalface_alt.xml   # Haar cascade face detector
│   └── haarcascade_frontalface_default.xml
├── src/
│   ├── gui/                  # GUI screens & customized Swing components
│   │   ├── Theme.java
│   │   ├── ModernComponents.java
│   │   ├── LoginScreen.java
│   │   ├── SidebarPanel.java
│   │   ├── AppWorkspacePanel.java
│   │   ├── DashboardContent.java
│   │   ├── StudentRegistrationScreen.java
│   │   ├── AttendanceMonitoringScreen.java
│   │   ├── AttendanceRecordsScreen.java
│   │   ├── ReportsAnalyticsScreen.java
│   │   └── MainFrame.java
│   ├── Main.java             # Entry points & Core recognizer backend
│   └── FaceRecognizer.java
├── attendance.csv            # Log storage for check-ins
├── students.csv              # Student metadata database
└── README.md
```
