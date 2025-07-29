import cv2
import numpy as np
import mediapipe as mp
import time
import math
from collections import deque
import speech_recognition as sr
import threading
import queue

class HandDetector:
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) != 0:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

class VoiceController:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()
        self.is_listening = False
        
    def start_listening(self):
        """Start voice recognition in a separate thread"""
        self.is_listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
    
    def stop_listening(self):
        """Stop voice recognition"""
        self.is_listening = False
    
    def _listen_loop(self):
        """Background thread for voice recognition"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                command = self.recognizer.recognize_google(audio).lower()
                self.command_queue.put(command)
                print(f"Voice command: {command}")
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"Voice recognition error: {e}")
    
    def get_command(self):
        """Get the latest voice command"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

class ShapeDetector:
    """AI-powered shape detection and auto-completion"""
    
    @staticmethod
    def detect_shape(points, tolerance=0.1):
        """Detect if points form a recognizable shape"""
        if len(points) < 3:
            return None
        
        # Calculate distances between points
        distances = []
        for i in range(len(points) - 1):
            dist = math.sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)
            distances.append(dist)
        
        # Check if it's a circle (similar distances, closed shape)
        if len(points) >= 8:
            avg_dist = sum(distances) / len(distances)
            variance = sum((d - avg_dist)**2 for d in distances) / len(distances)
            if variance < avg_dist * tolerance:
                return "circle"
        
        # Check if it's a rectangle (4 points, right angles)
        if len(points) == 4:
            # Calculate angles
            angles = []
            for i in range(4):
                p1 = points[i]
                p2 = points[(i+1) % 4]
                p3 = points[(i+2) % 4]
                
                v1 = (p1[0] - p2[0], p1[1] - p2[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot / (mag1 * mag2)
                    angle = math.acos(max(-1, min(1, cos_angle)))
                    angles.append(angle)
            
            # Check if angles are close to 90 degrees
            if all(abs(angle - math.pi/2) < 0.3 for angle in angles):
                return "rectangle"
        
        return None
    
    @staticmethod
    def complete_shape(shape_type, points, imgCanvas, color, thickness):
        """Complete the detected shape"""
        if shape_type == "circle":
            # Find center and radius
            center_x = sum(p[0] for p in points) // len(points)
            center_y = sum(p[1] for p in points) // len(points)
            radius = int(sum(math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in points) / len(points))
            cv2.circle(imgCanvas, (center_x, center_y), radius, color, thickness)
            return True
        
        elif shape_type == "rectangle":
            # Find bounding rectangle
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            cv2.rectangle(imgCanvas, (x1, y1), (x2, y2), color, thickness)
            return True
        
        return False

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    # Initialize hand detector
    detector = HandDetector(detectionCon=0.85)
    
    # Initialize voice controller
    voice_controller = VoiceController()
    voice_controller.start_listening()
    
    # Initialize shape detector
    shape_detector = ShapeDetector()
    
    # Drawing variables
    xp, yp = 0, 0
    brushThickness = 15
    eraserThickness = 50
    drawColor = (255, 0, 255)
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
    # NEW FEATURES: Advanced tools and modes
    brushSizes = [5, 10, 15, 25, 35]
    currentBrushSize = 2
    eraserSizes = [20, 35, 50, 75, 100]
    currentEraserSize = 2
    
    # Tool modes
    TOOL_BRUSH = "brush"
    TOOL_ERASER = "eraser"
    TOOL_SHAPE = "shape"
    TOOL_TEXT = "text"
    TOOL_FILL = "fill"
    current_tool = TOOL_BRUSH
    
    # Shape drawing variables
    shape_points = []
    is_drawing_shape = False
    
    # Text variables
    text_mode = False
    text_input = ""
    text_position = (0, 0)
    
    # Undo/Redo functionality
    undoStack = deque(maxlen=20)
    redoStack = deque(maxlen=20)
    
    # UI Setup
    header_height = 150
    header_width = 1280
    header = np.zeros((header_height, header_width, 3), np.uint8)
    header[:] = (100, 50, 0)
    cv2.putText(header, "Advanced Virtual Painter", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
    num_colors = len(colors)
    swatch_width = 120  # Reduced width
    swatch_height = 75
    swatch_start_y = 37
    swatch_spacing = 15  # Reduced spacing
    
    button_width = 100
    button_height = 50
    
    # Tool buttons (middle section)
    brush_button_x = 1000
    brush_button_y = 25
    eraser_button_x = 1000
    eraser_button_y = 85
    shape_button_x = 1000
    shape_button_y = 145
    text_button_x = 1000
    text_button_y = 205
    fill_button_x = 1000
    fill_button_y = 265
    
    # Action buttons (right side)
    clear_button_x = 1100
    clear_button_y = 25
    save_button_x = 1100
    save_button_y = 85
    undo_button_x = 1100
    undo_button_y = 145
    redo_button_x = 1100
    redo_button_y = 205
    voice_button_x = 1100
    voice_button_y = 265
    
    # Size control buttons (left of tool buttons)
    brush_size_button_x = 900
    brush_size_button_y = 25
    eraser_size_button_x = 900
    eraser_size_button_y = 85
    
    def saveCanvasState():
        """Save current canvas state for undo"""
        undoStack.append(imgCanvas.copy())
        redoStack.clear()
    
    def undo():
        """Undo last action"""
        if undoStack:
            redoStack.append(imgCanvas.copy())
            imgCanvas[:] = undoStack.pop()
    
    def redo():
        """Redo last undone action"""
        if redoStack:
            undoStack.append(imgCanvas.copy())
            imgCanvas[:] = redoStack.pop()
    
    def flood_fill(start_point, target_color, replacement_color):
        """Flood fill algorithm"""
        x, y = start_point
        if x < 0 or x >= imgCanvas.shape[1] or y < 0 or y >= imgCanvas.shape[0]:
            return
        
        if np.array_equal(imgCanvas[y, x], target_color):
            imgCanvas[y, x] = replacement_color
            flood_fill((x+1, y), target_color, replacement_color)
            flood_fill((x-1, y), target_color, replacement_color)
            flood_fill((x, y+1), target_color, replacement_color)
            flood_fill((x, y-1), target_color, replacement_color)
    
    def process_voice_command(command):
        """Process voice commands"""
        if "brush" in command:
            return "tool", TOOL_BRUSH
        elif "eraser" in command:
            return "tool", TOOL_ERASER
        elif "shape" in command:
            return "tool", TOOL_SHAPE
        elif "text" in command:
            return "tool", TOOL_TEXT
        elif "fill" in command:
            return "tool", TOOL_FILL
        elif "clear" in command:
            return "action", "clear"
        elif "save" in command:
            return "action", "save"
        elif "undo" in command:
            return "action", "undo"
        elif "redo" in command:
            return "action", "redo"
        elif "red" in command:
            return "color", (0, 0, 255)
        elif "green" in command:
            return "color", (0, 255, 0)
        elif "blue" in command:
            return "color", (255, 0, 0)
        elif "yellow" in command:
            return "color", (0, 255, 255)
        elif "purple" in command:
            return "color", (255, 0, 255)
        elif "white" in command:
            return "color", (255, 255, 255)
        return None, None
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        img = cv2.flip(img, 1)  # Mirror the image
        
        # Process voice commands
        voice_command = voice_controller.get_command()
        if voice_command:
            command_type, command_value = process_voice_command(voice_command)
            if command_type == "tool":
                current_tool = command_value
                print(f"Voice: Switched to {command_value} tool")
            elif command_type == "action":
                if command_value == "clear":
                    saveCanvasState()
                    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                    print("Voice: Canvas cleared")
                elif command_value == "save":
                    timestamp = int(time.time())
                    filename = f"voice_painting_{timestamp}.png"
                    cv2.imwrite(filename, imgCanvas)
                    print(f"Voice: Painting saved as {filename}")
                elif command_value == "undo":
                    undo()
                    print("Voice: Undo performed")
                elif command_value == "redo":
                    redo()
                    print("Voice: Redo performed")
            elif command_type == "color":
                drawColor = command_value
                print(f"Voice: Color changed")
        
        # Find hands
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
            x2, y2 = lmList[12][1], lmList[12][2]  # Middle finger tip
            
            # Check which fingers are up
            fingers = detector.fingersUp()
            
            # Selection mode - Two fingers up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                
                # Check if clicking on color swatches
                if y1 < header_height:
                    for i, color in enumerate(colors):
                        x1_swatch = swatch_spacing * (i + 1) + swatch_width * i
                        y1_swatch = swatch_start_y
                        if (x1_swatch < x1 < x1_swatch + swatch_width and 
                            y1_swatch < y1 < y1_swatch + swatch_height):
                            drawColor = color
                            current_tool = TOOL_BRUSH
                
                # Check if clicking on tool buttons
                if y1 < header_height:
                    # Tool buttons
                    if (brush_button_x < x1 < brush_button_x + button_width and 
                        brush_button_y < y1 < brush_button_y + button_height):
                        current_tool = TOOL_BRUSH
                        print("Brush tool activated!")
                    
                    elif (eraser_button_x < x1 < eraser_button_x + button_width and 
                          eraser_button_y < y1 < eraser_button_y + button_height):
                        current_tool = TOOL_ERASER
                        print("Eraser tool activated!")
                    
                    elif (shape_button_x < x1 < shape_button_x + button_width and 
                          shape_button_y < y1 < shape_button_y + button_height):
                        current_tool = TOOL_SHAPE
                        print("Shape tool activated!")
                    
                    elif (text_button_x < x1 < text_button_x + button_width and 
                          text_button_y < y1 < text_button_y + button_height):
                        current_tool = TOOL_TEXT
                        print("Text tool activated!")
                    
                    elif (fill_button_x < x1 < fill_button_x + button_width and 
                          fill_button_y < y1 < fill_button_y + button_height):
                        current_tool = TOOL_FILL
                        print("Fill tool activated!")
                    
                    # Action buttons
                    elif (clear_button_x < x1 < clear_button_x + button_width and 
                          clear_button_y < y1 < clear_button_y + button_height):
                        saveCanvasState()
                        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                        print("Canvas cleared!")
                    
                    elif (save_button_x < x1 < save_button_x + button_width and 
                          save_button_y < y1 < save_button_y + button_height):
                        timestamp = int(time.time())
                        filename = f"advanced_painting_{timestamp}.png"
                        cv2.imwrite(filename, imgCanvas)
                        print(f"Painting saved as {filename}")
                    
                    elif (undo_button_x < x1 < undo_button_x + button_width and 
                          undo_button_y < y1 < undo_button_y + button_height):
                        undo()
                        print("Undo performed!")
                    
                    elif (redo_button_x < x1 < redo_button_x + button_width and 
                          redo_button_y < y1 < redo_button_y + button_height):
                        redo()
                        print("Redo performed!")
                    
                    # Size control buttons
                    elif (brush_size_button_x < x1 < brush_size_button_x + button_width and 
                          brush_size_button_y < y1 < brush_size_button_y + button_height):
                        currentBrushSize = (currentBrushSize + 1) % len(brushSizes)
                        brushThickness = brushSizes[currentBrushSize]
                        print(f"Brush size: {brushThickness}")
                    
                    elif (eraser_size_button_x < x1 < eraser_size_button_x + button_width and 
                          eraser_size_button_y < y1 < eraser_size_button_y + button_height):
                        currentEraserSize = (currentEraserSize + 1) % len(eraserSizes)
                        eraserThickness = eraserSizes[currentEraserSize]
                        print(f"Eraser size: {eraserThickness}")
                
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            
            # Drawing mode - Index finger up
            elif fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                # Handle different tools
                if current_tool == TOOL_BRUSH:
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                
                elif current_tool == TOOL_ERASER:
                    mask = np.zeros(imgCanvas.shape[:2], dtype=np.uint8)
                    cv2.line(mask, (xp, yp), (x1, y1), 255, eraserThickness)
                    imgCanvas[mask == 255] = [0, 0, 0]
                
                elif current_tool == TOOL_SHAPE:
                    if not is_drawing_shape:
                        is_drawing_shape = True
                        shape_points = [(x1, y1)]
                    else:
                        shape_points.append((x1, y1))
                        # Draw temporary line
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                
                elif current_tool == TOOL_FILL:
                    # Flood fill at current position
                    target_color = imgCanvas[y1, x1].copy()
                    flood_fill((x1, y1), target_color, drawColor)
                
                xp, yp = x1, y1
            
            # Shape completion - Three fingers up
            elif fingers[1] and fingers[2] and fingers[3]:
                if current_tool == TOOL_SHAPE and is_drawing_shape and len(shape_points) > 2:
                    # Try to detect and complete shape
                    detected_shape = shape_detector.detect_shape(shape_points)
                    if detected_shape:
                        if shape_detector.complete_shape(detected_shape, shape_points, imgCanvas, drawColor, brushThickness):
                            print(f"AI completed {detected_shape}!")
                    is_drawing_shape = False
                    shape_points = []
        
        # Create header with UI
        header_with_ui = header.copy()
        
        # Draw color swatches
        for i, color in enumerate(colors):
            x1 = swatch_spacing * (i + 1) + swatch_width * i
            y1 = swatch_start_y
            cv2.rectangle(header_with_ui, (x1, y1), (x1 + swatch_width, y1 + swatch_height), color, cv2.FILLED)
            if drawColor == color:
                cv2.rectangle(header_with_ui, (x1, y1), (x1 + swatch_width, y1 + swatch_height), (0, 0, 0), 5)
        
        # Draw tool buttons with active state indication
        tools = [
            (brush_button_x, brush_button_y, "Brush", TOOL_BRUSH, (0, 255, 0)),
            (eraser_button_x, eraser_button_y, "Eraser", TOOL_ERASER, (0, 0, 255)),
            (shape_button_x, shape_button_y, "Shape", TOOL_SHAPE, (255, 255, 0)),
            (text_button_x, text_button_y, "Text", TOOL_TEXT, (255, 0, 255)),
            (fill_button_x, fill_button_y, "Fill", TOOL_FILL, (0, 255, 255))
        ]
        
        for x, y, name, tool, active_color in tools:
            if current_tool == tool:
                cv2.rectangle(header_with_ui, (x, y), (x + button_width, y + button_height), active_color, 3)
                cv2.putText(header_with_ui, name, (x + 5, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, active_color, 2)
            else:
                cv2.rectangle(header_with_ui, (x, y), (x + button_width, y + button_height), (255, 255, 255), 2)
                cv2.putText(header_with_ui, name, (x + 5, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw action buttons
        actions = [
            (clear_button_x, clear_button_y, "Clear"),
            (save_button_x, save_button_y, "Save"),
            (undo_button_x, undo_button_y, "Undo"),
            (redo_button_x, redo_button_y, "Redo"),
            (voice_button_x, voice_button_y, "Voice")
        ]
        
        for x, y, name in actions:
            cv2.rectangle(header_with_ui, (x, y), (x + button_width, y + button_height), (255, 255, 255), 2)
            cv2.putText(header_with_ui, name, (x + 5, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw size control buttons
        cv2.rectangle(header_with_ui, (brush_size_button_x, brush_size_button_y), 
                     (brush_size_button_x + button_width, brush_size_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, f"B:{brushThickness}", (brush_size_button_x + 5, brush_size_button_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.rectangle(header_with_ui, (eraser_size_button_x, eraser_size_button_y), 
                     (eraser_size_button_x + button_width, eraser_size_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, f"E:{eraserThickness}", (eraser_size_button_x + 5, eraser_size_button_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine header with main image
        img[0:header_height, 0:header_width] = header_with_ui
        
        # Blend canvas with image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        # Display mode indicator
        tool_names = {
            TOOL_BRUSH: f"Brush (Size: {brushThickness})",
            TOOL_ERASER: f"Eraser (Size: {eraserThickness})",
            TOOL_SHAPE: "Shape Tool - Draw shape, 3 fingers to complete",
            TOOL_TEXT: "Text Tool - Click to add text",
            TOOL_FILL: "Fill Tool - Click to fill area"
        }
        
        mode_text = tool_names.get(current_tool, "Unknown Tool")
        mode_color = (0, 255, 0) if current_tool == TOOL_BRUSH else (0, 0, 255) if current_tool == TOOL_ERASER else (255, 255, 0)
        
        cv2.putText(img, mode_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Show voice status
        cv2.putText(img, "Voice: Active", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show the image
        cv2.imshow("Advanced Virtual Painter", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    voice_controller.stop_listening()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 