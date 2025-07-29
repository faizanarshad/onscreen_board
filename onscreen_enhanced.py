import cv2
import numpy as np
import mediapipe as mp
import time
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
    
    # NEW FEATURES: Brush sizes and undo/redo
    brushSizes = [5, 10, 15, 25, 35]
    currentBrushSize = 2  # Index for brushSizes
    eraserSizes = [20, 35, 50, 75, 100]
    currentEraserSize = 2  # Index for eraserSizes
    
    # Undo/Redo functionality
    undoStack = deque(maxlen=10)  # Store last 10 canvas states
    redoStack = deque(maxlen=10)
    
    # UI Setup
    header_height = 150
    header_width = 1280
    header = np.zeros((header_height, header_width, 3), np.uint8)
    header[:] = (100, 50, 0)
    cv2.putText(header, "Enhanced Virtual Painter", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
    num_colors = len(colors)
    swatch_width = 120  # Reduced width
    swatch_height = 75
    swatch_start_y = 37
    swatch_spacing = 15  # Reduced spacing
    
    button_width = 100
    button_height = 50
    # Position buttons on the right side, stacked vertically
    brush_button_x = 1080
    brush_button_y = 25
    eraser_button_x = 1080
    eraser_button_y = 85
    clear_button_x = 1080
    clear_button_y = 145
    save_button_x = 1080
    save_button_y = 205
    undo_button_x = 1080
    undo_button_y = 265
    redo_button_x = 1080
    redo_button_y = 325
    
    # NEW: Brush size buttons (moved to avoid overlap)
    brush_size_button_x = 900
    brush_size_button_y = 25
    eraser_size_button_x = 900
    eraser_size_button_y = 85
    
    # Mode variables
    drawing_mode = True  # True for brush, False for eraser
    
    def saveCanvasState():
        """Save current canvas state for undo"""
        undoStack.append(imgCanvas.copy())
        redoStack.clear()  # Clear redo when new action is performed
    
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
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        img = cv2.flip(img, 1)  # Mirror the image
        
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
                            drawing_mode = True
                
                # Check if clicking on buttons
                if y1 < header_height:
                    # Brush button
                    if (brush_button_x < x1 < brush_button_x + button_width and 
                        brush_button_y < y1 < brush_button_y + button_height):
                        drawing_mode = True
                        print("Brush mode activated!")
                    
                    # Eraser button
                    elif (eraser_button_x < x1 < eraser_button_x + button_width and 
                          eraser_button_y < y1 < eraser_button_y + button_height):
                        drawing_mode = False
                        print("Eraser mode activated!")
                    
                    # Clear button
                    elif (clear_button_x < x1 < clear_button_x + button_width and 
                          clear_button_y < y1 < clear_button_y + button_height):
                        saveCanvasState()
                        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                        print("Canvas cleared!")
                    
                    # Save button
                    elif (save_button_x < x1 < save_button_x + button_width and 
                          save_button_y < y1 < save_button_y + button_height):
                        timestamp = int(time.time())
                        filename = f"virtual_painting_{timestamp}.png"
                        cv2.imwrite(filename, imgCanvas)
                        print(f"Painting saved as {filename}")
                    
                    # NEW: Undo button
                    elif (undo_button_x < x1 < undo_button_x + button_width and 
                          undo_button_y < y1 < undo_button_y + button_height):
                        undo()
                        print("Undo performed!")
                    
                    # NEW: Redo button
                    elif (redo_button_x < x1 < redo_button_x + button_width and 
                          redo_button_y < y1 < redo_button_y + button_height):
                        redo()
                        print("Redo performed!")
                    
                    # NEW: Brush size buttons
                    elif (brush_size_button_x < x1 < brush_size_button_x + button_width and 
                          brush_size_button_y < y1 < brush_size_button_y + button_height):
                        currentBrushSize = (currentBrushSize + 1) % len(brushSizes)
                        brushThickness = brushSizes[currentBrushSize]
                        print(f"Brush size: {brushThickness}")
                    
                    # NEW: Eraser size buttons
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
                
                if drawing_mode:
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                else:
                    # Eraser: create a mask and remove drawn content
                    mask = np.zeros(imgCanvas.shape[:2], dtype=np.uint8)
                    cv2.line(mask, (xp, yp), (x1, y1), 255, eraserThickness)
                    imgCanvas[mask == 255] = [0, 0, 0]
                
                xp, yp = x1, y1
        
        # Create header with UI
        header_with_ui = header.copy()
        
        # Draw color swatches
        for i, color in enumerate(colors):
            x1 = swatch_spacing * (i + 1) + swatch_width * i
            y1 = swatch_start_y
            cv2.rectangle(header_with_ui, (x1, y1), (x1 + swatch_width, y1 + swatch_height), color, cv2.FILLED)
            if drawColor == color:
                cv2.rectangle(header_with_ui, (x1, y1), (x1 + swatch_width, y1 + swatch_height), (0, 0, 0), 5)
        
        # Draw buttons
        # Draw brush button with active state indication
        if drawing_mode:
            cv2.rectangle(header_with_ui, (brush_button_x, brush_button_y), (brush_button_x + button_width, brush_button_y + button_height), (0, 255, 0), 3)
            cv2.putText(header_with_ui, "Brush", (brush_button_x + 10, brush_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(header_with_ui, (brush_button_x, brush_button_y), (brush_button_x + button_width, brush_button_y + button_height), (255, 255, 255), 2)
            cv2.putText(header_with_ui, "Brush", (brush_button_x + 10, brush_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw eraser button with active state indication
        if not drawing_mode:
            cv2.rectangle(header_with_ui, (eraser_button_x, eraser_button_y), (eraser_button_x + button_width, eraser_button_y + button_height), (0, 0, 255), 3)
            cv2.putText(header_with_ui, "Eraser", (eraser_button_x + 5, eraser_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(header_with_ui, (eraser_button_x, eraser_button_y), (eraser_button_x + button_width, eraser_button_y + button_height), (255, 255, 255), 2)
            cv2.putText(header_with_ui, "Eraser", (eraser_button_x + 5, eraser_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(header_with_ui, (clear_button_x, clear_button_y), (clear_button_x + button_width, clear_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, "Clear", (clear_button_x + 5, clear_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(header_with_ui, (save_button_x, save_button_y), (save_button_x + button_width, save_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, "Save", (save_button_x + 5, save_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # NEW: Undo/Redo buttons
        cv2.rectangle(header_with_ui, (undo_button_x, undo_button_y), (undo_button_x + button_width, undo_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, "Undo", (undo_button_x + 5, undo_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.rectangle(header_with_ui, (redo_button_x, redo_button_y), (redo_button_x + button_width, redo_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, "Redo", (redo_button_x + 5, redo_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # NEW: Brush/Eraser size buttons
        cv2.rectangle(header_with_ui, (brush_size_button_x, brush_size_button_y), (brush_size_button_x + button_width, brush_size_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, f"B:{brushThickness}", (brush_size_button_x + 5, brush_size_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.rectangle(header_with_ui, (eraser_size_button_x, eraser_size_button_y), (eraser_size_button_x + button_width, eraser_size_button_y + button_height), (255, 255, 255), 2)
        cv2.putText(header_with_ui, f"E:{eraserThickness}", (eraser_size_button_x + 5, eraser_size_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine header with main image
        img[0:header_height, 0:header_width] = header_with_ui
        
        # Blend canvas with image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        # Display mode indicator with better visibility
        if drawing_mode:
            mode_text = f"Brush Mode (Size: {brushThickness})"
            mode_color = (0, 255, 0)  # Green for brush
        else:
            mode_text = f"Eraser Mode (Size: {eraserThickness})"
            mode_color = (0, 0, 255)  # Red for eraser
        
        cv2.putText(img, mode_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        
        # Show the image
        cv2.imshow("Enhanced Virtual Painter", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 