# Advanced Virtual Painter - AI-Powered Hand Gesture Drawing Application

A sophisticated Python application that allows you to draw on screen using hand gestures captured through your webcam. Features AI-powered shape detection, voice commands, and advanced drawing tools. Uses MediaPipe for hand tracking, OpenCV for computer vision, and SpeechRecognition for voice control.

## Features

### **Core Features**
- **Hand Gesture Recognition**: Track your hand movements in real-time
- **Virtual Drawing**: Draw on screen using your index finger
- **Color Selection**: Choose from 7 different colors using hand gestures
- **Tool Selection**: Switch between multiple drawing tools
- **Save Functionality**: Save your artwork as PNG files with timestamps
- **Clear Canvas**: Clear the drawing area with a gesture

### **Advanced Features**
- **AI Shape Detection**: Automatically detect and complete circles and rectangles
- **Voice Commands**: Control the application using voice commands
- **Multiple Drawing Tools**: Brush, Eraser, Shape, Text, and Fill tools
- **Brush Size Control**: 5 different brush sizes (5-35 pixels)
- **Eraser Size Control**: 5 different eraser sizes (20-100 pixels)
- **Undo/Redo System**: 20-level undo/redo history
- **Flood Fill Tool**: Fill enclosed areas with color
- **Shape Drawing**: Draw shapes and let AI complete them
- **Real-time Voice Feedback**: Voice commands with immediate response

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam
- macOS, Windows, or Linux

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install opencv-python numpy mediapipe SpeechRecognition pyaudio
```

**Note**: If you encounter compatibility issues, use these specific versions:
```bash
pip install "opencv-python<4.9.0" "numpy<2.0.0" mediapipe SpeechRecognition pyaudio
```

**System Dependencies (macOS):**
```bash
brew install portaudio
```

## Usage

### Running the Application

1. **Basic Application (with camera)**:
   ```bash
   python onscreen.py
   ```

2. **Enhanced Application (with additional features)**:
   ```bash
   python onscreen_enhanced.py
   ```

3. **Advanced Application (with AI and voice commands)**:
   ```bash
   python onscreen_advanced.py
   ```

4. **Demo Mode (without camera)**:
   ```bash
   python onscreen_demo.py
   ```

### Camera Access Setup

#### macOS
1. Go to **System Preferences** > **Security & Privacy** > **Privacy** > **Camera**
2. Find your terminal application (Terminal, iTerm, etc.) and enable camera access
3. Restart your terminal application
4. Run the application again

#### Windows
- Camera access is usually granted automatically
- If prompted, allow camera access when the application requests it

#### Linux
- Ensure your user has access to video devices
- You might need to run: `sudo usermod -a -G video $USER`

### How to Use

#### **Basic Controls**
1. **Drawing**: Point your index finger to draw on the screen
2. **Selection Mode**: Point both index and middle fingers to enter selection mode
3. **Color Selection**: In selection mode, point at the color swatches in the header
4. **Tool Selection**: Use the tool buttons in the header
5. **Clear Canvas**: Click the Clear button to erase everything
6. **Save Artwork**: Click the Save button to save your drawing
7. **Quit**: Press 'q' to exit the application

#### **Advanced Controls (Advanced Version)**
1. **AI Shape Detection**: 
   - Use Shape tool to draw rough shapes
   - Point 3 fingers up to let AI complete the shape
   - Supports circles and rectangles

2. **Voice Commands**:
   - "Brush" - Switch to brush tool
   - "Eraser" - Switch to eraser tool
   - "Shape" - Switch to shape tool
   - "Fill" - Switch to fill tool
   - "Red", "Green", "Blue", "Yellow", "Purple", "White" - Change colors
   - "Clear" - Clear canvas
   - "Save" - Save artwork
   - "Undo" - Undo last action
   - "Redo" - Redo last action

3. **Advanced Tools**:
   - **Brush Tool**: Freehand drawing with adjustable size
   - **Eraser Tool**: Erase with adjustable size
   - **Shape Tool**: Draw shapes for AI completion
   - **Fill Tool**: Fill enclosed areas with color
   - **Size Control**: Click size buttons to change brush/eraser size

### Gesture Controls

#### **Basic Gestures**
- **Index finger up**: Draw mode
- **Index + Middle finger up**: Selection mode
- **All fingers down**: No action

#### **Advanced Gestures (Advanced Version)**
- **Index finger up**: Draw mode
- **Index + Middle finger up**: Selection mode
- **Index + Middle + Ring finger up**: Complete shape (AI mode)
- **All fingers down**: No action

## Troubleshooting

### Camera Not Working
- Ensure camera permissions are enabled
- Check if another application is using the camera
- Try restarting your terminal/IDE
- On macOS, make sure to grant camera access to your terminal application

### Import Errors
If you get import errors related to NumPy or TensorFlow:
```bash
pip uninstall numpy opencv-python mediapipe
pip install "numpy<2.0.0" "opencv-python<4.9.0" mediapipe
```

### Performance Issues
- Ensure good lighting for hand detection
- Keep your hand clearly visible to the camera
- Close other applications using the camera

## File Structure

```
onscreen_board/
├── onscreen.py              # Basic application with camera
├── onscreen_enhanced.py     # Enhanced version with additional features
├── onscreen_advanced.py     # Advanced version with AI and voice commands
├── onscreen_demo.py         # Demo version without camera
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Technical Details

- **Hand Detection**: MediaPipe Hands model
- **Computer Vision**: OpenCV
- **Hand Tracking**: Real-time landmark detection
- **Drawing**: Canvas-based drawing with OpenCV
- **Voice Recognition**: SpeechRecognition with Google Speech API
- **AI Shape Detection**: Custom algorithm for circle and rectangle detection
- **Flood Fill**: Recursive algorithm for area filling
- **Undo/Redo**: Stack-based history management

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- MediaPipe
- SpeechRecognition
- PyAudio
- Webcam
- Microphone (for voice commands)

## License

This project is open source and available under the MIT License.