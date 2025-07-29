import cv2
import numpy as np
import time
import math

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
            for r in range(glow_radius, 0, -2):
                alpha = 0.6 * (1 - r / glow_radius)
                glow_color = [int(c * alpha) for c in color]
                cv2.circle(img, (swatch_x + swatch_size//2, swatch_y + swatch_size//2), r, glow_color, -1)
        
        # Main color swatch with rounded corners
        create_rounded_rectangle(img, swatch_x, swatch_y, swatch_x + swatch_size, swatch_y + swatch_size, color, 8, -1)
        
        # Subtle border
        border_color = [max(0, c - 40) for c in color]
        create_rounded_rectangle(img, swatch_x, swatch_y, swatch_x + swatch_size, swatch_y + swatch_size, border_color, 8, 2)
        
        # Selection indicator
        if color == selected_color:
            cv2.circle(img, (swatch_x + swatch_size//2, swatch_y + swatch_size//2), swatch_size//3, (255, 255, 255), 3)
            cv2.circle(img, (swatch_x + swatch_size//2, swatch_y + swatch_size//2), swatch_size//3 - 2, (0, 0, 0), 2)

def create_demo_ui():
    """Create a beautiful demo UI"""
    
    # Create main image with gradient background
    img = create_gradient_background(1280, 720, (30, 30, 60), (60, 30, 60), 'vertical')
    
    # UI Setup
    header_height = 180
    header_width = 1280
    
    # Beautiful gradient header
    header_gradient1 = (45, 45, 85)   # Dark blue
    header_gradient2 = (85, 45, 85)   # Purple
    header = create_gradient_background(header_width, header_height, header_gradient1, header_gradient2, 'horizontal')
    
    # Add animated particles to header
    current_time = time.time()
    for i in range(8):
        x = int((current_time * 50 + i * 150) % header_width)
        y = int(50 + 30 * math.sin(current_time * 2 + i))
        cv2.circle(header, (x, y), 4, (255, 255, 255, 150), -1)
    
    # Beautiful title with glow effect
    title = "Beautiful Virtual Painter"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
    title_x = (header_width - title_size[0]) // 2
    title_y = 90
    
    # Title glow effect
    for i in range(4):
        glow_color = (100 + i * 40, 100 + i * 40, 255)
        cv2.putText(header, title, (title_x + i, title_y + i), cv2.FONT_HERSHEY_SIMPLEX, 1.8, glow_color, 4)
    
    # Main title
    cv2.putText(header, title, (title_x, title_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    
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
    
    selected_color = colors[0]  # Red
    create_color_palette(header, 50, 100, colors, selected_color, 50)
    
    # Tool buttons with beautiful design
    tools = [
        {"name": "Brush", "color": (0, 255, 100), "icon": "B"},
        {"name": "Eraser", "color": (255, 100, 100), "icon": "E"},
        {"name": "Shape", "color": (100, 100, 255), "icon": "S"}
    ]
    
    tool_x = 650
    tool_y = 100
    for i, tool in enumerate(tools):
        button_x = tool_x + i * 120
        button_y = tool_y
        is_active = (i == 0)  # First tool active
        create_animated_button(header, button_x, button_y, 100, 50, 
                             f"{tool['icon']} {tool['name']}", tool["color"], is_active)
    
    # Action buttons
    actions = [
        {"name": "Clear", "color": (255, 150, 0), "icon": "C"},
        {"name": "Save", "color": (0, 200, 100), "icon": "S"},
        {"name": "Undo", "color": (200, 100, 200), "icon": "U"},
        {"name": "Redo", "color": (100, 200, 200), "icon": "R"}
    ]
    
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
    size_text = "Size: Medium (15px)"
    
    # Size indicator background with rounded corners
    create_rounded_rectangle(header, size_x, size_y, size_x + 250, size_y + 35, (0, 0, 0, 150), 18, -1)
    cv2.putText(header, size_text, (size_x + 15, size_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, size_color, 2)
    
    # Combine header with main image
    img[0:header_height, 0:header_width] = header
    
    # Beautiful instructions section
    instructions = [
        "How to Use This Beautiful Virtual Painter:",
        "",
        "Hand Gestures:",
        "   • Point your index finger to draw",
        "   • Point index + middle finger to select tools/colors",
        "   • Point 3 fingers to complete shapes (AI mode)",
        "",
        "Features:",
        "   • Beautiful gradient interface",
        "   • 10 vibrant colors with glow effects",
        "   • Multiple brush sizes and tools",
        "   • AI-powered shape detection",
        "   • Voice commands (in advanced version)",
        "   • Undo/Redo with 20-level history",
        "",
        "Tips:",
        "   • Ensure good lighting for hand detection",
        "   • Keep your hand clearly visible to the camera",
        "   • Use voice commands for hands-free operation",
        "",
        "Ready to create beautiful artwork!"
    ]
    
    y_offset = header_height + 20
    for i, instruction in enumerate(instructions):
        if instruction.startswith("How to Use"):
            # Main title
            cv2.putText(img, instruction, (20, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        elif instruction.startswith("Hand Gestures") or instruction.startswith("Features") or instruction.startswith("Tips") or instruction.startswith("Ready"):
            # Section headers
            cv2.putText(img, instruction, (20, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif instruction.startswith("   •"):
            # Bullet points
            cv2.putText(img, instruction, (20, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        elif instruction == "":
            # Empty lines
            continue
        else:
            # Regular text
            cv2.putText(img, instruction, (20, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add decorative elements
    # Floating bubbles
    for i in range(6):
        x = int((current_time * 30 + i * 200) % 1280)
        y = int(400 + 100 * math.sin(current_time * 1.5 + i))
        size = int(10 + 5 * math.sin(current_time * 2 + i))
        color = (100 + i * 25, 150 + i * 15, 255 - i * 30)
        cv2.circle(img, (x, y), size, color, -1)
        cv2.circle(img, (x, y), size + 2, (255, 255, 255), 1)
    
    # Status bar at bottom
    status_y = 680
    status_bg_color = (0, 0, 0, 180)
    cv2.rectangle(img, (0, status_y), (1280, 720), status_bg_color, -1)
    cv2.rectangle(img, (0, status_y), (1280, 720), (255, 255, 255), 2)
    
    status_text = "Beautiful Virtual Painter Demo | Press 'q' to quit | Camera access required for full functionality"
    cv2.putText(img, status_text, (20, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def main():
    print("Starting Beautiful Virtual Painter Demo...")
    print("Press 'q' to quit")
    
    while True:
        # Create beautiful demo UI
        img = create_demo_ui()
        
        # Show the image
        cv2.imshow("Beautiful Virtual Painter Demo", img)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Demo closed.")

if __name__ == "__main__":
    main() 