import cv2
import numpy as np
import mediapipe as mp
import time
import math
from collections import deque

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

def create_gradient_background(width, height, color1, color2, direction='horizontal'):
    """Create a gradient background"""
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    if direction == 'horizontal':
        for x in range(width):
            ratio = x / width
            color = tuple(int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3))
            background[:, x] = color
    else:  # vertical
        for y in range(height):
            ratio = y / height
            color = tuple(int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3))
            background[y, :] = color
    
    return background

def create_button(img, x, y, width, height, text, color, is_active=False):
    """Create a simple button"""
    # Button background
    cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
    
    # Active state glow
    if is_active:
        cv2.rectangle(img, (x-2, y-2), (x + width+2, y + height+2), (255, 255, 255), 3)
    
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 2)
    
    # Text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_shape(canvas, shape_type, start_point, end_point, color, thickness=2):
    """Draw different shapes based on type"""
    if shape_type == "Circle":
        # Calculate center and radius
        center_x = (start_point[0] + end_point[0]) // 2
        center_y = (start_point[1] + end_point[1]) // 2
        radius = int(math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) // 2)
        cv2.circle(canvas, (center_x, center_y), radius, color, thickness)
        
    elif shape_type == "Rectangle":
        cv2.rectangle(canvas, start_point, end_point, color, thickness)
        
    elif shape_type == "Triangle":
        # Calculate triangle points
        x1, y1 = start_point
        x2, y2 = end_point
        x3 = x1 + (x2 - x1) // 2
        y3 = y1 - abs(y2 - y1) // 2
        
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
        cv2.polylines(canvas, [pts], True, color, thickness)
        
    elif shape_type == "Line":
        cv2.line(canvas, start_point, end_point, color, thickness)

def main():
    # Initialize webcam with error handling
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available. Running in demo mode...")
        cap = None
    else:
        cap.set(3, 1280)
        cap.set(4, 720)
    
    # Initialize hand detector
    detector = HandDetector(detectionCon=0.85)
    
    # Drawing variables
    xp, yp = 0, 0
    brushThickness = 15
    eraserThickness = 50
    drawColor = (255, 0, 255)
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
    # Enhanced features
    brushSizes = [5, 10, 15, 20, 25, 30]
    currentBrushSize = 2
    eraserSizes = [20, 30, 40, 50, 60, 70]
    currentEraserSize = 3
    
    # Undo/Redo system
    undoStack = deque(maxlen=20)
    redoStack = deque(maxlen=20)
    
    # Current tool
    drawing_mode = True  # True for brush, False for eraser
    shape_mode = False   # True for shape drawing
    selected_shape = None
    shape_start_point = None
    
    # Beautiful colors palette
    colors = [
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (255, 255, 255),  # White
        (255, 165, 0),    # Orange
        (128, 0, 128),    # Purple
        (0, 255, 128)     # Lime
    ]
    
    # NEW LAYOUT: Colors on top, buttons on right, shapes on left
    header_height = 100
    header_width = 1280
    
    # Color palette on top
    color_start_x = 50
    color_y = 20
    color_size = 50
    color_spacing = 15
    
    # Right side buttons (vertical)
    button_width = 120
    button_height = 50
    button_spacing = 15
    right_button_x = 1100
    
    # Left side shapes (vertical)
    shape_width = 120
    shape_height = 50
    shape_spacing = 15
    left_shape_x = 20
    
    # Tool buttons
    tools = [
        {"name": "Brush", "color": (0, 255, 100)},
        {"name": "Eraser", "color": (255, 100, 100)},
        {"name": "Clear", "color": (255, 150, 0)},
        {"name": "Save", "color": (0, 200, 100)}
    ]
    
    # Shape options
    shapes = [
        {"name": "Circle", "color": (255, 100, 100)},
        {"name": "Rectangle", "color": (100, 255, 100)},
        {"name": "Triangle", "color": (100, 100, 255)},
        {"name": "Line", "color": (255, 255, 100)}
    ]
    
    def saveCanvasState():
        undoStack.append(imgCanvas.copy())
        redoStack.clear()
    
    def undo():
        if undoStack:
            redoStack.append(imgCanvas.copy())
            imgCanvas[:] = undoStack.pop()
    
    def redo():
        if redoStack:
            undoStack.append(imgCanvas.copy())
            imgCanvas[:] = redoStack.pop()
    
    # Save initial state
    saveCanvasState()
    
    while True:
        # Get camera frame or create demo frame
        if cap is not None:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame. Creating demo frame...")
                img = create_gradient_background(1280, 720, (50, 50, 100), (100, 50, 150), 'vertical')
        else:
            # Demo mode - create a static frame
            img = create_gradient_background(1280, 720, (50, 50, 100), (100, 50, 150), 'vertical')
            # Add demo text
            cv2.putText(img, "DEMO MODE - Camera not available", (400, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Use mouse to interact with buttons", (400, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
        img = cv2.flip(img, 1)  # Mirror the image
        
        # Find hands only if camera is available
        if cap is not None:
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
                    shape_start_point = None  # Reset shape drawing
                    
                    # Check if clicking on color swatches (top)
                    if y1 < header_height:
                        for i, color in enumerate(colors):
                            color_x = color_start_x + i * (color_size + color_spacing)
                            if (color_x < x1 < color_x + color_size and 
                                color_y < y1 < color_y + color_size):
                                drawColor = color
                                drawing_mode = True
                                shape_mode = False
                    
                    # Check if clicking on right side buttons
                    if x1 > right_button_x - button_width:
                        button_index = (y1 - header_height) // (button_height + button_spacing)
                        
                        if 0 <= button_index < len(tools):
                            if button_index == 0:  # Brush
                                drawing_mode = True
                                shape_mode = False
                                brushThickness = brushSizes[currentBrushSize]
                            elif button_index == 1:  # Eraser
                                drawing_mode = False
                                shape_mode = False
                                eraserThickness = eraserSizes[currentEraserSize]
                            elif button_index == 2:  # Clear
                                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                                saveCanvasState()
                                print("Canvas cleared!")
                            elif button_index == 3:  # Save
                                timestamp = int(time.time())
                                filename = f"beautiful_painting_{timestamp}.png"
                                cv2.imwrite(filename, imgCanvas)
                                print(f"Painting saved as {filename}")
                    
                    # Check if clicking on left side shapes
                    if x1 < left_shape_x + shape_width:
                        shape_index = (y1 - header_height) // (shape_height + shape_spacing)
                        if 0 <= shape_index < len(shapes):
                            selected_shape = shapes[shape_index]["name"]
                            shape_mode = True
                            drawing_mode = False
                            print(f"Selected shape: {selected_shape}")
                    
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                
                # Drawing mode - Index finger up
                elif fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                        if shape_mode:
                            shape_start_point = (x1, y1)
                    
                    if drawing_mode:
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    elif shape_mode and selected_shape:
                        # For shapes, draw immediately when moving
                        if shape_start_point:
                            # Clear previous shape by redrawing canvas
                            temp_canvas = imgCanvas.copy()
                            # Draw the shape
                            draw_shape(temp_canvas, selected_shape, shape_start_point, (x1, y1), drawColor, 3)
                            # Update the main canvas
                            imgCanvas[:] = temp_canvas[:]
                    else:
                        # Eraser: create a mask and remove drawn content
                        mask = np.zeros(imgCanvas.shape[:2], dtype=np.uint8)
                        cv2.line(mask, (xp, yp), (x1, y1), 255, eraserThickness)
                        imgCanvas[mask == 255] = [0, 0, 0]
                    
                    xp, yp = x1, y1
                
                # Shape completion - When finger is lifted (no fingers up)
                elif not fingers[1] and shape_mode and selected_shape and shape_start_point:
                    # Save the completed shape
                    saveCanvasState()
                    print(f"Completed {selected_shape} shape!")
                    shape_start_point = None
                    xp, yp = 0, 0
        
        # Create beautiful header with colors on top
        header = create_gradient_background(header_width, header_height, (80, 40, 120), (120, 60, 180), 'horizontal')
        
        # Draw color palette on top
        for i, color in enumerate(colors):
            color_x = color_start_x + i * (color_size + color_spacing)
            
            # Create gradient swatch
            swatch = create_gradient_background(color_size, color_size, 
                                              tuple(max(0, c - 30) for c in color), 
                                              color, 'vertical')
            
            # Add glow if selected
            if color == drawColor:
                glow_color = tuple(min(255, c + 80) for c in color)
                cv2.rectangle(header, (color_x-3, color_y-3), 
                             (color_x+color_size+3, color_y+color_size+3), 
                             glow_color, 3)
            
            # Place swatch
            header[color_y:color_y+color_size, color_x:color_x+color_size] = swatch
            
            # Add border
            cv2.rectangle(header, (color_x, color_y), 
                         (color_x+color_size, color_y+color_size), 
                         (255, 255, 255), 2)
        
        # Combine header with main image
        img[0:header_height, 0:header_width] = header
        
        # Draw right side buttons (vertical) - on the main image
        button_y_start = header_height + 20
        for i, tool in enumerate(tools):
            button_y = button_y_start + i * (button_height + button_spacing)
            is_active = (i == 0 and drawing_mode) or (i == 1 and not drawing_mode and not shape_mode)
            create_button(img, right_button_x, button_y, button_width, button_height, 
                         tool["name"], tool["color"], is_active)
        
        # Draw left side shapes (vertical) - on the main image
        shape_y_start = header_height + 20
        for i, shape in enumerate(shapes):
            shape_y = shape_y_start + i * (shape_height + shape_spacing)
            is_active = shape_mode and selected_shape == shape["name"]
            create_button(img, left_shape_x, shape_y, shape_width, shape_height, 
                         shape["name"], shape["color"], is_active)
        
        # Beautiful title with glow effect
        title = "Beautiful Virtual Painter"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (header_width - title_size[0]) // 2
        title_y = header_height // 2 + 10
        
        # Title shadow
        cv2.putText(header, title, (title_x + 2, title_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        # Title glow
        cv2.putText(header, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        
        # Add animated particles to header
        for i in range(5):
            particle_x = int(50 + (time.time() * 50 + i * 100) % (header_width - 100))
            particle_y = int(20 + 10 * np.sin(time.time() * 2 + i))
            cv2.circle(header, (particle_x, particle_y), 3, (255, 255, 255), -1)
        
        # Blend canvas with image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        # Display current tool and size
        if drawing_mode:
            mode_text = f"Brush Mode (Size: {brushThickness})"
            mode_color = (0, 255, 0)
        elif shape_mode and selected_shape:
            mode_text = f"Shape Mode: {selected_shape}"
            mode_color = (255, 0, 255)
        else:
            mode_text = f"Eraser Mode (Size: {eraserThickness})"
            mode_color = (0, 0, 255)
        
        cv2.putText(img, mode_text, (10, header_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Add shape drawing instructions
        if shape_mode and selected_shape:
            instruction_text = f"Draw {selected_shape}: Point and drag to draw, lift finger to complete"
            cv2.putText(img, instruction_text, (10, header_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow("Beautiful Virtual Painter", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 