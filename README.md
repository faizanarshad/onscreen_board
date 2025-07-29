# Virtual Painter - Hand Gesture Drawing Application

A Python application that allows you to draw on screen using hand gestures captured through your webcam. The application uses MediaPipe for hand tracking and OpenCV for computer vision.

## Features

- **Hand Gesture Recognition**: Track your hand movements in real-time
- **Virtual Drawing**: Draw on screen using your index finger
- **Color Selection**: Choose from 7 different colors using hand gestures
- **Tool Selection**: Switch between brush and eraser modes
- **Save Functionality**: Save your artwork as PNG files
- **Clear Canvas**: Clear the drawing area with a gesture

## Installation

### Prerequisites
- Python 3.7 or higher
- Webcam
- macOS, Windows, or Linux

### Install Dependencies

```bash
pip install opencv-python numpy mediapipe
```

**Note**: If you encounter compatibility issues, use these specific versions:
```bash
pip install "opencv-python<4.9.0" "numpy<2.0.0" mediapipe
```

## Usage

### Running the Application

1. **Full Application (with camera)**:
   ```bash
   python onscreen.py
   ```

2. **Demo Mode (without camera)**:
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

1. **Drawing**: Point your index finger to draw on the screen
2. **Selection Mode**: Point both index and middle fingers to enter selection mode
3. **Color Selection**: In selection mode, point at the color swatches in the header
4. **Tool Selection**: Use the Brush/Eraser buttons in the header
5. **Clear Canvas**: Click the Clear button to erase everything
6. **Save Artwork**: Click the Save button to save your drawing
7. **Quit**: Press 'q' to exit the application

### Gesture Controls

- **Index finger up**: Draw mode
- **Index + Middle finger up**: Selection mode
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
├── onscreen.py          # Main application with camera
├── onscreen_demo.py     # Demo version without camera
└── README.md           # This file
```

## Technical Details

- **Hand Detection**: MediaPipe Hands model
- **Computer Vision**: OpenCV
- **Hand Tracking**: Real-time landmark detection
- **Drawing**: Canvas-based drawing with OpenCV

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- MediaPipe
- Webcam

## License

This project is open source and available under the MIT License.