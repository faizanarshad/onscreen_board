import cv2
import numpy as np
import time

def create_demo_ui():
    """Create a demo UI to show the virtual painter interface"""
    
    # Create main image
    img = np.zeros((720, 1280, 3), np.uint8)
    
    # UI Setup
    header_height = 150
    header_width = 1280
    header = np.zeros((header_height, header_width, 3), np.uint8)
    header[:] = (100, 50, 0)
    cv2.putText(header, "Virtual Painter UI - Demo Mode", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
    swatch_width = 160
    swatch_height = 75
    swatch_start_y = 37
    swatch_spacing = 20
    
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
    
    # Create header with UI
    header_with_ui = header.copy()
    
    # Draw color swatches
    for i, color in enumerate(colors):
        x1 = swatch_spacing * (i + 1) + swatch_width * i
        y1 = swatch_start_y
        cv2.rectangle(header_with_ui, (x1, y1), (x1 + swatch_width, y1 + swatch_height), color, cv2.FILLED)
        # Highlight first color as selected
        if i == 0:
            cv2.rectangle(header_with_ui, (x1, y1), (x1 + swatch_width, y1 + swatch_height), (0, 0, 0), 5)
    
    # Draw buttons
    cv2.rectangle(header_with_ui, (brush_button_x, brush_button_y), (brush_button_x + button_width, brush_button_y + button_height), (255, 255, 255), 2)
    cv2.putText(header_with_ui, "Brush", (brush_button_x + 10, brush_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(header_with_ui, (eraser_button_x, eraser_button_y), (eraser_button_x + button_width, eraser_button_y + button_height), (255, 255, 255), 2)
    cv2.putText(header_with_ui, "Eraser", (eraser_button_x + 5, eraser_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(header_with_ui, (clear_button_x, clear_button_y), (clear_button_x + button_width, clear_button_y + button_height), (255, 255, 255), 2)
    cv2.putText(header_with_ui, "Clear", (clear_button_x + 5, clear_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.rectangle(header_with_ui, (save_button_x, save_button_y), (save_button_x + button_width, save_button_y + button_height), (255, 255, 255), 2)
    cv2.putText(header_with_ui, "Save", (save_button_x + 5, save_button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Combine header with main image
    img[0:header_height, 0:header_width] = header_with_ui
    
    # Add instructions
    instructions = [
        "Virtual Painter Instructions:",
        "",
        "1. Point your index finger to draw",
        "2. Point index and middle finger to select tools/colors",
        "3. Click on color swatches to change color",
        "4. Use Brush/Eraser buttons to switch modes",
        "5. Press 'q' to quit",
        "",
        "Note: Camera access required for full functionality",
        "Enable camera access in System Preferences > Security & Privacy > Camera"
    ]
    
    y_offset = 200
    for i, instruction in enumerate(instructions):
        cv2.putText(img, instruction, (10, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add a sample drawing area
    cv2.rectangle(img, (50, 500), (1230, 680), (50, 50, 50), 2)
    cv2.putText(img, "Drawing Area - Use your hand gestures here", (60, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    return img

def main():
    print("Starting Virtual Painter Demo...")
    print("Press 'q' to quit")
    
    while True:
        # Create demo UI
        img = create_demo_ui()
        
        # Show the image
        cv2.imshow("Virtual Painter - Demo Mode", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Demo closed.")

if __name__ == "__main__":
    main() 