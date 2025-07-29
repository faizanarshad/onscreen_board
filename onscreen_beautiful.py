import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import math

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
    """Create a beautiful gradient background"""
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    if direction == 'horizontal':
        for x in range(width):
            ratio = x / width
            color = [int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3)]
            background[:, x] = color
    else:  # vertical
        for y in range(height):
            ratio = y / height
            color = [int(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(3)]
            background[y, :] = color
    
    return background

def create_rounded_rectangle(img, x1, y1, x2, y2, color, radius=10, thickness=-1):
    """Draw a rounded rectangle"""
    # Draw the main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw the rounded corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def create_glowing_effect(img, x, y, radius, color, intensity=0.8):
    """Create a glowing effect around a point"""
    for r in range(radius, 0, -2):
        alpha = intensity * (1 - r / radius)
        glow_color = [int(c * alpha) for c in color]
        cv2.circle(img, (x, y), r, glow_color, -1)

def create_animated_button(img, x, y, width, height, text, color, is_active=False, animation_frame=0):
    """Create an animated button with hover effects"""
    # Base button with gradient
    if is_active:
        # Active state - glowing effect
        glow_color = [min(255, c + 50) for c in color]
        create_rounded_rectangle(img, x-2, y-2, x+width+2, y+height+2, glow_color, 12, -1)
    
    # Main button
    create_rounded_rectangle(img, x, y, x+width, y+height, color, 10, -1)
    
    # Add subtle shadow
    shadow_color = [max(0, c - 30) for c in color]
    create_rounded_rectangle(img, x+2, y+2, x+width+2, y+height+2, shadow_color, 10, -1)
    
    # Animated border
    if animation_frame > 0:
        border_color = [min(255, c + 30) for c in color]
        create_rounded_rectangle(img, x-1, y-1, x+width+1, y+height+1, border_color, 10, 2)
    
    # Text with shadow
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    
    # Text shadow
    cv2.putText(img, text, (text_x+1, text_y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    # Main text
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def create_color_palette(img, x, y, colors, selected_color, swatch_size=60):
    """Create a beautiful color palette with effects"""
    for i, color in enumerate(colors):
        swatch_x = x + i * (swatch_size + 10)
        swatch_y = y
        
        # Glowing effect for selected color
        if color == selected_color:
            glow_radius = swatch_size // 2 + 5
            create_glowing_effect(img, swatch_x + swatch_size//2, swatch_y + swatch_size//2, glow_radius, color, 0.6)
        
        # Main color swatch with rounded corners
        create_rounded_rectangle(img, swatch_x, swatch_y, swatch_x + swatch_size, swatch_y + swatch_size, color, 8, -1)
        
        # Subtle border
        border_color = [max(0, c - 40) for c in color]
        create_rounded_rectangle(img, swatch_x, swatch_y, swatch_x + swatch_size, swatch_y + swatch_size, border_color, 8, 2)
        
        # Selection indicator
        if color == selected_color:
            cv2.circle(img, (swatch_x + swatch_size//2, swatch_y + swatch_size//2), swatch_size//3, (255, 255, 255), 3)
            cv2.circle(img, (swatch_x + swatch_size//2, swatch_y + swatch_size//2), swatch_size//3 - 2, (0, 0, 0), 2)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
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
    
    # Beautiful color palette
    colors = [
        (255, 0, 0),    # Red
        (255, 165, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
        (0, 255, 255),  # Cyan
        (0, 0, 255),    # Blue
        (128, 0, 128),  # Purple
        (255, 192, 203), # Pink
        (255, 255, 255), # White
        (0, 0, 0)       # Black
    ]
    
    # Brush sizes with beautiful names
    brushSizes = [5, 10, 15, 25, 35]
    brushNames = ["Fine", "Thin", "Medium", "Thick", "Bold"]
    currentBrushSize = 2
    
    # Tool modes
    TOOL_BRUSH = "brush"
    TOOL_ERASER = "eraser"
    TOOL_SHAPE = "shape"
    current_tool = TOOL_BRUSH
    
    # Animation variables
    animation_frame = 0
    hover_effects = {}
    
    # Undo/Redo functionality
    undoStack = deque(maxlen=20)
    redoStack = deque(maxlen=20)
    
    # UI Setup
    header_height = 180
    header_width = 1280
    
    # Beautiful gradient colors
    header_gradient1 = (45, 45, 85)   # Dark blue
    header_gradient2 = (85, 45, 85)   # Purple
    
    # Tool button definitions
    tools = [
        {"name": "Brush", "tool": TOOL_BRUSH, "color": (0, 255, 100), "icon": "B"},
        {"name": "Eraser", "tool": TOOL_ERASER, "color": (255, 100, 100), "icon": "E"},
        {"name": "Shape", "tool": TOOL_SHAPE, "color": (100, 100, 255), "icon": "S"}
    ]
    
    # Action buttons
    actions = [
        {"name": "Clear", "color": (255, 150, 0), "icon": "C"},
        {"name": "Save", "color": (0, 200, 100), "icon": "S"},
        {"name": "Undo", "color": (200, 100, 200), "icon": "U"},
        {"name": "Redo", "color": (100, 200, 200), "icon": "R"}
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
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        img = cv2.flip(img, 1)  # Mirror the image
        
        # Create beautiful header with gradient
        header = create_gradient_background(header_width, header_height, header_gradient1, header_gradient2, 'horizontal')
        
        # Add decorative elements to header
        # Floating particles effect
        for i in range(5):
            x = int((time.time() * 50 + i * 200) % header_width)
            y = int(50 + 20 * math.sin(time.time() * 2 + i))
            cv2.circle(header, (x, y), 3, (255, 255, 255, 100), -1)
        
        # Title with shadow and glow
        title = "Beautiful Virtual Painter"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (header_width - title_size[0]) // 2
        title_y = 80
        
        # Title glow effect
        for i in range(3):
            glow_color = (100 + i * 50, 100 + i * 50, 255)
            cv2.putText(header, title, (title_x + i, title_y + i), cv2.FONT_HERSHEY_SIMPLEX, 1.5, glow_color, 3)
        
        # Main title
        cv2.putText(header, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Create color palette
        create_color_palette(header, 50, 100, colors, drawColor, 50)
        
        # Create tool buttons
        tool_x = 650
        tool_y = 100
        for i, tool in enumerate(tools):
            button_x = tool_x + i * 120
            button_y = tool_y
            is_active = current_tool == tool["tool"]
            
            # Check for hover effect
            hover_key = f"tool_{i}"
            if hover_key in hover_effects:
                animation_frame = hover_effects[hover_key]
            else:
                animation_frame = 0
            
            create_animated_button(header, button_x, button_y, 100, 50, 
                                 f"{tool['icon']} {tool['name']}", tool["color"], is_active, animation_frame)
        
        # Create action buttons
        action_x = 1000
        action_y = 100
        for i, action in enumerate(actions):
            button_x = action_x + i * 120
            button_y = action_y
            
            create_animated_button(header, button_x, button_y, 100, 50, 
                                 f"{action['icon']} {action['name']}", action["color"])
        
        # Size indicator with beautiful design
        size_x = 50
        size_y = 160
        size_color = (255, 255, 255)
        size_text = f"Size: {brushNames[currentBrushSize]} ({brushSizes[currentBrushSize]}px)"
        
        # Size indicator background
        create_rounded_rectangle(header, size_x, size_y, size_x + 200, size_y + 30, (0, 0, 0, 100), 15, -1)
        cv2.putText(header, size_text, (size_x + 10, size_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, size_color, 2)
        
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
                
                # Check if clicking on color palette
                if y1 < header_height and 50 <= x1 <= 650:
                    color_index = (x1 - 50) // 60
                    if 0 <= color_index < len(colors):
                        drawColor = colors[color_index]
                        current_tool = TOOL_BRUSH
                
                # Check if clicking on tool buttons
                if y1 < header_height and tool_y <= y1 <= tool_y + 50:
                    for i, tool in enumerate(tools):
                        button_x = tool_x + i * 120
                        if button_x <= x1 <= button_x + 100:
                            current_tool = tool["tool"]
                            hover_effects[f"tool_{i}"] = 10
                
                # Check if clicking on action buttons
                if y1 < header_height and action_y <= y1 <= action_y + 50:
                    for i, action in enumerate(actions):
                        button_x = action_x + i * 120
                        if button_x <= x1 <= button_x + 100:
                            if action["name"] == "Clear":
                                saveCanvasState()
                                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                                print("Canvas cleared!")
                            elif action["name"] == "Save":
                                timestamp = int(time.time())
                                filename = f"beautiful_painting_{timestamp}.png"
                                cv2.imwrite(filename, imgCanvas)
                                print(f"Painting saved as {filename}")
                            elif action["name"] == "Undo":
                                undo()
                                print("Undo performed!")
                            elif action["name"] == "Redo":
                                redo()
                                print("Redo performed!")
                
                # Beautiful cursor indicator
                cv2.circle(img, (x1, y1), 20, drawColor, -1)
                cv2.circle(img, (x1, y1), 25, (255, 255, 255), 2)
            
            # Drawing mode - Index finger up
            elif fingers[1] and not fingers[2]:
                # Beautiful drawing cursor
                cv2.circle(img, (x1, y1), 15, drawColor, -1)
                cv2.circle(img, (x1, y1), 20, (255, 255, 255), 2)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                if current_tool == TOOL_BRUSH:
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                elif current_tool == TOOL_ERASER:
                    mask = np.zeros(imgCanvas.shape[:2], dtype=np.uint8)
                    cv2.line(mask, (xp, yp), (x1, y1), 255, eraserThickness)
                    imgCanvas[mask == 255] = [0, 0, 0]
                
                xp, yp = x1, y1
        
        # Combine header with main image
        img[0:header_height, 0:header_width] = header
        
        # Blend canvas with image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        # Beautiful status indicator
        status_x = 10
        status_y = header_height + 30
        
        # Status background
        status_bg_color = (0, 0, 0, 150)
        cv2.rectangle(img, (status_x, status_y), (status_x + 400, status_y + 40), status_bg_color, -1)
        cv2.rectangle(img, (status_x, status_y), (status_x + 400, status_y + 40), (255, 255, 255), 2)
        
        # Status text
        tool_names = {
            TOOL_BRUSH: "🖌️ Brush Mode",
            TOOL_ERASER: "🧽 Eraser Mode",
            TOOL_SHAPE: "🔷 Shape Mode"
        }
        
        status_text = f"{tool_names.get(current_tool, 'Unknown')} | Size: {brushNames[currentBrushSize]}"
        cv2.putText(img, status_text, (status_x + 10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Animate hover effects
        for key in list(hover_effects.keys()):
            hover_effects[key] = max(0, hover_effects[key] - 1)
            if hover_effects[key] == 0:
                del hover_effects[key]
        
        # Show the image
        cv2.imshow("Beautiful Virtual Painter", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 