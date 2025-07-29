import cv2
import numpy as np
import time
import math

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

def create_rounded_rectangle(img, x1, y1, x2, y2, color, radius=10, thickness=-1):
    """Draw a rounded rectangle"""
    # Draw main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def create_animated_button(img, x, y, width, height, text, color, is_active=False, animation_frame=0):
    """Create an animated button with glow effect"""
    # Button background with gradient
    button_bg = create_gradient_background(width, height, 
                                         tuple(max(0, c - 50) for c in color), 
                                         color, 'vertical')
    
    # Add glow effect if active
    if is_active:
        glow_color = tuple(min(255, c + 50) for c in color)
        cv2.rectangle(img, (x-2, y-2), (x+width+2, y+height+2), glow_color, 3)
    
    # Place button background
    img[y:y+height, x:x+width] = button_bg
    
    # Add border
    cv2.rectangle(img, (x, y), (x+width, y+height), (255, 255, 255), 2)
    
    # Add text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def create_color_palette(img, x, y, colors, selected_color, swatch_size=60):
    """Create a beautiful color palette"""
    for i, color in enumerate(colors):
        swatch_x = x + i * (swatch_size + 10)
        swatch_y = y
        
        # Create gradient swatch
        swatch = create_gradient_background(swatch_size, swatch_size, 
                                          tuple(max(0, c - 30) for c in color), 
                                          color, 'radial')
        
        # Add glow if selected
        if color == selected_color:
            glow_color = tuple(min(255, c + 80) for c in color)
            cv2.rectangle(img, (swatch_x-3, swatch_y-3), 
                         (swatch_x+swatch_size+3, swatch_y+swatch_size+3), 
                         glow_color, 3)
        
        # Place swatch
        img[swatch_y:swatch_y+swatch_size, swatch_x:swatch_x+swatch_size] = swatch
        
        # Add border
        cv2.rectangle(img, (swatch_x, swatch_y), 
                     (swatch_x+swatch_size, swatch_y+swatch_size), 
                     (255, 255, 255), 2)

def create_demo_ui():
    """Create the beautiful demo UI with new layout"""
    # Create main image with gradient background
    img = create_gradient_background(1280, 720, (50, 50, 100), (100, 50, 150), 'vertical')
    
    # NEW LAYOUT: Colors on top, buttons on right, shapes on left
    header_height = 100  # Reduced for colors on top
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
    right_button_x = 1100  # Far right
    
    # Left side shapes (vertical)
    shape_width = 120
    shape_height = 50
    shape_spacing = 15
    left_shape_x = 20  # Far left
    
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
    
    # Tool buttons with beautiful design
    tools = [
        {"name": "Brush", "color": (0, 255, 100), "icon": "B"},
        {"name": "Eraser", "color": (255, 100, 100), "icon": "E"},
        {"name": "Shape", "color": (100, 100, 255), "icon": "S"}
    ]
    
    # Action buttons
    actions = [
        {"name": "Clear", "color": (255, 150, 0), "icon": "C"},
        {"name": "Save", "color": (0, 200, 100), "icon": "S"},
        {"name": "Undo", "color": (200, 100, 200), "icon": "U"},
        {"name": "Redo", "color": (100, 200, 200), "icon": "R"}
    ]
    
    # Shape options
    shapes = [
        {"name": "Circle", "color": (255, 100, 100), "icon": "O"},
        {"name": "Rectangle", "color": (100, 255, 100), "icon": "[]"},
        {"name": "Triangle", "color": (100, 100, 255), "icon": "/\\"},
        {"name": "Line", "color": (255, 255, 100), "icon": "|"}
    ]
    
    # Create beautiful header with colors on top
    header = create_gradient_background(header_width, header_height, (80, 40, 120), (120, 60, 180), 'horizontal')
    
    # Draw color palette on top
    for i, color in enumerate(colors):
        color_x = color_start_x + i * (color_size + color_spacing)
        # Create gradient swatch
        swatch = create_gradient_background(color_size, color_size, 
                                          tuple(max(0, c - 30) for c in color), 
                                          color, 'radial')
        
        # Add glow if selected (first color as default)
        if i == 0:
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
    
    # Draw right side buttons (vertical)
    button_y_start = header_height + 20
    for i, tool in enumerate(tools):
        button_y = button_y_start + i * (button_height + button_spacing)
        is_active = i == 0  # First tool active by default
        create_animated_button(header, right_button_x, button_y, button_width, button_height, 
                             f"{tool['icon']} {tool['name']}", tool["color"], is_active)
    
    # Draw action buttons below tool buttons
    action_y_start = button_y_start + len(tools) * (button_height + button_spacing) + 20
    for i, action in enumerate(actions):
        button_y = action_y_start + i * (button_height + button_spacing)
        create_animated_button(header, right_button_x, button_y, button_width, button_height, 
                             f"{action['icon']} {action['name']}", action["color"])
    
    # Draw left side shapes (vertical)
    shape_y_start = header_height + 20
    for i, shape in enumerate(shapes):
        shape_y = shape_y_start + i * (shape_height + shape_spacing)
        create_animated_button(header, left_shape_x, shape_y, shape_width, shape_height, 
                             f"{shape['icon']} {shape['name']}", shape["color"])
    
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
    
    # Draw instructions with beautiful formatting
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
    
    # Add decorative floating bubbles
    for i in range(8):
        bubble_x = int(200 + 300 * np.sin(time.time() + i))
        bubble_y = int(400 + 200 * np.cos(time.time() * 0.5 + i))
        bubble_size = int(5 + 3 * np.sin(time.time() * 2 + i))
        cv2.circle(img, (bubble_x, bubble_y), bubble_size, (255, 255, 255, 100), -1)
        cv2.circle(img, (bubble_x, bubble_y), bubble_size, (255, 255, 255), 1)
    
    # Status bar at bottom
    status_y = 680
    cv2.rectangle(img, (0, status_y), (1280, 720), (255, 255, 255), 2)
    
    status_text = "Beautiful Virtual Painter Demo | Press 'q' to quit | Camera access required for full functionality"
    cv2.putText(img, status_text, (20, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def main():
    print("Starting Beautiful Virtual Painter Demo...")
    print("Press 'q' to quit")
    
    while True:
        # Create the beautiful demo UI
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