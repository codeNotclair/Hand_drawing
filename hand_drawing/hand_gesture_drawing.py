import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pickle
from collections import deque

class HandGestureDrawing:
    def __init__(self):
        # Initialize MediaPipe Hand module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,  # Increased from 0.7 to 0.8
            min_tracking_confidence=0.8    # Increased from 0.7 to 0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize drawing parameters
        self.canvas = None
        self.temp_canvas = None  # For shape preview
        
        # Position tracking with smoothing
        self.position_history = deque(maxlen=5)  # Store last 5 positions for smoothing
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        
        # Gesture states
        self.is_drawing = False
        self.prev_drawing_state = False
        self.draw_color = (0, 0, 255)  # Red color by default (BGR format)
        self.thickness = 5
        self.mode = "draw"  # Modes: draw, erase, rectangle, circle
        self.hand_closed = False  # Track if the hand is closed
        
        # Gesture debouncing
        self.gesture_cooldown = False
        self.last_gesture_time = time.time()
        self.cooldown_duration = 0.5  # seconds
        
        # Color palette (in BGR format)
        self.colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (255, 255, 255) # White
        ]
        self.selected_color_index = 0
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
            
        success, img = self.cap.read()
        if success:
            self.img_height, self.img_width = img.shape[0], img.shape[1]
            self.canvas = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            self.temp_canvas = np.zeros_like(self.canvas)
            
            # Define UI regions
            # Toolbar dimensions
            self.toolbar_height = 60
            self.toolbar = np.zeros((self.toolbar_height, self.img_width, 3), dtype=np.uint8)
            
            # Setup toolbar buttons
            self.button_size = 40
            self.button_margin = 10
            self.buttons = []
            
            # Add drawing mode button
            self.buttons.append({
                "x": self.button_margin,
                "y": self.button_margin,
                "icon": cv2.FILLED,
                "action": "draw",
                "color": (0, 0, 255)  # Red (BGR)
            })
            
            # Add eraser button
            self.buttons.append({
                "x": 2 * self.button_margin + self.button_size,
                "y": self.button_margin,
                "icon": cv2.FILLED,
                "action": "erase",
                "color": (255, 255, 255)  # White (BGR)
            })
            
            # Add rectangle button
            self.buttons.append({
                "x": 3 * self.button_margin + 2 * self.button_size,
                "y": self.button_margin,
                "icon": cv2.FILLED,
                "action": "rectangle",
                "color": (0, 255, 255)  # Yellow (BGR)
            })
            
            # Add circle button
            self.buttons.append({
                "x": 4 * self.button_margin + 3 * self.button_size,
                "y": self.button_margin,
                "icon": cv2.FILLED,
                "action": "circle",
                "color": (255, 255, 0)  # Cyan (BGR)
            })
            
            # Add save button
            self.buttons.append({
                "x": 5 * self.button_margin + 4 * self.button_size,
                "y": self.button_margin,
                "icon": cv2.FILLED,
                "action": "save",
                "color": (0, 255, 0)  # Green (BGR)
            })
            
            # Add load button
            self.buttons.append({
                "x": 6 * self.button_margin + 5 * self.button_size,
                "y": self.button_margin,
                "icon": cv2.FILLED,
                "action": "load",
                "color": (255, 0, 255)  # Magenta (BGR)
            })
            
            # Add stroke eraser button
            self.buttons.append({
                "x": 7 * self.button_margin + 6 * self.button_size,
                "y": self.button_margin,
                "icon": cv2.FILLED,
                "action": "stroke_erase",
                "color": (0, 0, 0)  # Black (BGR)
            })
            
            # Rectangle and circle drawing
            self.shape_start_x, self.shape_start_y = 0, 0
            self.drawing_shape = False
            
            # Wave gesture detection
            self.wave_gesture_detected = False
            self.wave_gesture_start_time = 0
        else:
            raise IOError("Failed to capture initial frame")
        
    def detect_hands(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hands
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            return frame, hand_landmarks
        
        return frame, None
    
    def smooth_position(self, x, y):
        """Apply smoothing to input coordinates using historical positions"""
        self.position_history.append((x, y))
        
        if len(self.position_history) < 2:
            return x, y
        
        # Calculate smoothed position (average of history)
        smoothed_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
        smoothed_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)
        
        return int(smoothed_x), int(smoothed_y)
    
    def detect_gesture(self, hand_landmarks, frame_shape):
        if not hand_landmarks:
            # Reset drawing state if no hands detected
            self.is_drawing = False
            self.hand_closed = False  # Reset hand closed state
            return False
        
        # Get finger tip landmarks
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_x = int(index_tip.x * frame_shape[1])
        index_y = int(index_tip.y * frame_shape[0])
        
        # Get other finger landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_x = int(thumb_tip.x * frame_shape[1])
        thumb_y = int(thumb_tip.y * frame_shape[0])
        
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_x = int(middle_tip.x * frame_shape[1])
        middle_y = int(middle_tip.y * frame_shape[0])
        
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_x = int(ring_tip.x * frame_shape[1])
        ring_y = int(ring_tip.y * frame_shape[0])
        
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_x = int(pinky_tip.x * frame_shape[1])
        pinky_y = int(pinky_tip.y * frame_shape[0])
        
        # Get finger MCP (knuckle) landmarks for more gesture detection
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_mcp_y = int(index_mcp.y * frame_shape[0])
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        
        # Calculate distances for gesture detection
        thumb_index_distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
        thumb_middle_distance = np.sqrt((thumb_x - middle_x)**2 + (thumb_y - middle_y)**2)
        thumb_ring_distance = np.sqrt((thumb_x - ring_x)**2 + (thumb_y - ring_y)**2)
        thumb_pinky_distance = np.sqrt((thumb_x - pinky_x)**2 + (thumb_y - pinky_y)**2)

        # Detect closed hand (all fingers close to thumb)
        if (thumb_index_distance < 30 and thumb_middle_distance < 30 and
                thumb_ring_distance < 30 and thumb_pinky_distance < 30):
            self.hand_closed = True
        else:
            if self.hand_closed:
                # Hand was closed and now opened, change color
                self.selected_color_index = (self.selected_color_index + 1) % len(self.colors)
                self.draw_color = self.colors[self.selected_color_index]
                self.hand_closed = False
        
        # Apply smoothing to cursor position
        smooth_x, smooth_y = self.smooth_position(index_x, index_y)
        
        # Constrain the coordinates to the canvas area
        smooth_x = max(0, min(smooth_x, frame_shape[1] - 1))
        smooth_y = max(self.toolbar_height, min(smooth_y, frame_shape[0] - 1))
        
        # Update finger position for drawing
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        self.curr_x, self.curr_y = smooth_x, smooth_y
        
        # Check if we're in the toolbar area
        in_toolbar = index_y < self.toolbar_height
        
        # Check if we're clicking a button
        if in_toolbar and thumb_index_distance < 30:
            for button in self.buttons:
                if (abs(index_x - button["x"] - self.button_size//2) < self.button_size//2 and 
                    abs(index_y - button["y"] - self.button_size//2) < self.button_size//2):
                    if not self.gesture_cooldown:
                        self.handle_button_press(button["action"])
                        self.gesture_cooldown = True
                        self.last_gesture_time = time.time()
            return False
        
        # Handle debouncing cooldown
        current_time = time.time()
        if self.gesture_cooldown and (current_time - self.last_gesture_time) > self.cooldown_duration:
            self.gesture_cooldown = False
        
        # DRAWING MODE - Thumb and index finger pinched (close together)
        if thumb_index_distance < 40:
            if self.mode in ["draw", "erase", "stroke_erase"]:
                self.is_drawing = True
            elif self.mode in ["rectangle", "circle"]:
                if not self.drawing_shape:
                    # Starting a new shape
                    self.drawing_shape = True
                    self.shape_start_x, self.shape_start_y = self.curr_x, self.curr_y
                
            # Check additional gestures only if not in cooldown
            if not self.gesture_cooldown:
                
                # Gesture for changing mode - all fingers except index extended
                all_fingers_up = (
                    middle_y < index_mcp_y and 
                    ring_y < index_mcp_y and 
                    pinky_y < index_mcp_y
                )
                
                if all_fingers_up:
                    # Cycle through modes
                    modes = ["draw", "erase", "rectangle", "circle"]
                    current_index = modes.index(self.mode)
                    self.mode = modes[(current_index + 1) % len(modes)]
                    self.gesture_cooldown = True
                    self.last_gesture_time = time.time()
                
                # Gesture for changing color - middle finger extended
                elif middle_y < index_y - 50 and self.mode == "draw":
                    self.selected_color_index = (self.selected_color_index + 1) % len(self.colors)
                    self.draw_color = self.colors[self.selected_color_index]
                    self.gesture_cooldown = True
                    self.last_gesture_time = time.time()
                
                # Gesture for changing brush size - distance between thumb and pinky
                elif pinky_y < index_mcp_y:
                    thumb_pinky_distance = np.sqrt((thumb_x - pinky_x)**2 + (thumb_y - pinky_y)**2)
                    # Map distance to thickness (min 1, max 30)
                    self.thickness = max(1, min(30, int(thumb_pinky_distance / 7)))
                
                # Victory gesture for switching colors - index and middle fingers extended
                elif index_y < index_mcp_y and middle_y < index_mcp_y and ring_y > index_mcp_y and pinky_y > index_mcp_y:
                    self.selected_color_index = (self.selected_color_index + 1) % len(self.colors)
                    self.draw_color = self.colors[self.selected_color_index]
                    self.gesture_cooldown = True
                    self.last_gesture_time = time.time()
                
                # Gesture for changing tool - ring finger extended
                elif ring_y < index_y - 50:
                    tools = ["draw", "erase", "rectangle", "circle"]
                    current_tool_index = tools.index(self.mode)
                    self.mode = tools[(current_tool_index + 1) % len(tools)]
                    self.gesture_cooldown = True
                    self.last_gesture_time = time.time()
        else:
            if self.mode in ["draw", "erase"]:
                self.is_drawing = False
            elif self.mode in ["rectangle", "circle"] and self.drawing_shape:
                # Finish drawing the shape
                self.temp_canvas = np.zeros_like(self.temp_canvas)
                
                if self.mode == "rectangle":
                    cv2.rectangle(
                        self.canvas,
                        (self.shape_start_x, self.shape_start_y),
                        (self.curr_x, self.curr_y),
                        self.draw_color,  # Corrected - removed redundant condition
                        self.thickness
                    )
                elif self.mode == "circle":
                    radius = int(np.sqrt((self.curr_x - self.shape_start_x)**2 + 
                                          (self.curr_y - self.shape_start_y)**2))
                    cv2.circle(
                        self.canvas,
                        (self.shape_start_x, self.shape_start_y),
                        radius,
                        self.draw_color,  # Corrected - removed redundant condition
                        self.thickness
                    )
                
                self.drawing_shape = False
        
        # Detect wave gesture (left-right movement)
        if len(self.position_history) >= 5:
            x_positions = [pos[0] for pos in self.position_history]
            if max(x_positions) - min(x_positions) > 100:  # Threshold for wave gesture
                if not self.wave_gesture_detected:
                    self.wave_gesture_detected = True
                    self.wave_gesture_start_time = time.time()
                elif time.time() - self.wave_gesture_start_time > 0.5:  # Duration to confirm wave gesture
                    self.canvas = np.zeros_like(self.canvas)
                    self.temp_canvas = np.zeros_like(self.temp_canvas)
                    self.wave_gesture_detected = False
            else:
                self.wave_gesture_detected = False

        # Detect thumbs up gesture
        if (thumb_tip.y < thumb_ip.y < thumb_mcp.y and
                index_mcp.y < thumb_mcp.y):
            if not self.gesture_cooldown:
                self.gesture_cooldown = True
                self.last_gesture_time = time.time()
                self.save_canvas_delay = True  # Set flag to save canvas after delay

        # Check if we need to save the canvas after delay
        if self.save_canvas_delay and (time.time() - self.last_gesture_time) > 5:
            self.save_canvas()
            self.save_canvas_delay = False

        return self.is_drawing
    
    def handle_button_press(self, action):
        """Handle toolbar button presses"""
        if action in ["draw", "erase", "rectangle", "circle", "stroke_erase"]:
            self.mode = action
        elif action == "save":
            self.save_canvas()
        elif action == "load":
            self.load_canvas()
    
    def save_canvas(self):
        """Save the current canvas to a file"""
        # Create directory if it doesn't exist
        if not os.path.exists("saved_drawings"):
            os.makedirs("saved_drawings")
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save as image
        cv2.imwrite(f"saved_drawings/drawing_{timestamp}.png", self.canvas)
        
        # Also save as pickle for restoring exact data
        with open(f"saved_drawings/drawing_{timestamp}.pkl", 'wb') as f:
            pickle.dump(self.canvas, f)
    
    def load_canvas(self):
        """Load a previously saved canvas"""
        # Check if directory exists
        if not os.path.exists("saved_drawings"):
            return
            
        # Find the most recent pickle file
        files = [f for f in os.listdir('saved_drawings') if f.startswith('drawing_') and f.endswith('.pkl')]
        if not files:
            return
        
        # Sort files by timestamp (most recent last)
        files.sort()
        latest_file = files[-1]
        
        # Load the canvas
        try:
            with open(os.path.join("saved_drawings", latest_file), 'rb') as f:
                loaded_canvas = pickle.load(f)
                
                # Ensure loaded canvas has the same dimensions as current canvas
                if loaded_canvas.shape == self.canvas.shape:
                    self.canvas = loaded_canvas
                else:
                    # Resize if dimensions don't match
                    self.canvas = cv2.resize(loaded_canvas, (self.img_width, self.img_height))
        except Exception as e:
            print(f"Error loading canvas: {e}")
    
    def draw_on_canvas(self):
        if self.is_drawing and self.prev_drawing_state and self.mode in ["draw", "erase", "stroke_erase"]:
            color = self.draw_color if self.mode == "draw" else (0, 0, 0)
            thickness = self.thickness if self.mode == "draw" else 20
            
            # Make sure coordinates are valid
            if self.prev_x > 0 and self.prev_y > self.toolbar_height and self.curr_x > 0 and self.curr_y > self.toolbar_height:
                if self.mode == "stroke_erase":
                    # Erase the entire stroke
                    mask = np.zeros_like(self.canvas[:, :, 0])
                    cv2.line(mask, (self.prev_x, self.prev_y), (self.curr_x, self.curr_y), 255, thickness)
                    self.canvas[mask == 255] = (0, 0, 0)
                else:
                    cv2.line(
                        self.canvas, 
                        (self.prev_x, self.prev_y), 
                        (self.curr_x, self.curr_y), 
                        color, 
                        thickness
                    )
        
        # Update preview for shape drawing
        if self.mode in ["rectangle", "circle"] and self.drawing_shape:
            self.temp_canvas = np.zeros_like(self.temp_canvas)
            
            if self.mode == "rectangle":
                cv2.rectangle(
                    self.temp_canvas,
                    (self.shape_start_x, self.shape_start_y),
                    (self.curr_x, self.curr_y),
                    self.draw_color,
                    self.thickness
                )
            elif self.mode == "circle":
                radius = int(np.sqrt((self.curr_x - self.shape_start_x)**2 + 
                                      (self.curr_y - self.shape_start_y)**2))
                cv2.circle(
                    self.temp_canvas,
                    (self.shape_start_x, self.shape_start_y),
                    radius,
                    self.draw_color,
                    self.thickness
                )
        
        self.prev_drawing_state = self.is_drawing
    
    def draw_toolbar(self):
        """Draw the toolbar with buttons and color palette"""
        # Clear toolbar
        self.toolbar.fill(50)  # Dark gray background
        
        # Draw buttons
        for button in self.buttons:
            # Highlight the active mode
            border_color = (0, 255, 0) if button["action"] == self.mode else (200, 200, 200)
            border_thickness = 2 if button["action"] == self.mode else 1
            
            # Draw button background
            cv2.rectangle(
                self.toolbar,
                (button["x"], button["y"]),
                (button["x"] + self.button_size, button["y"] + self.button_size),
                border_color,
                border_thickness
            )
            
            # Draw button icon
            if button["action"] in ["draw", "erase"]:
                # Draw a dot for pen/eraser
                center_x = button["x"] + self.button_size // 2
                center_y = button["y"] + self.button_size // 2
                cv2.circle(self.toolbar, (center_x, center_y), self.button_size // 4, button["color"], -1)
            elif button["action"] == "rectangle":
                # Draw rectangle icon
                top_left = (button["x"] + 10, button["y"] + 10)
                bottom_right = (button["x"] + self.button_size - 10, button["y"] + self.button_size - 10)
                cv2.rectangle(self.toolbar, top_left, bottom_right, button["color"], 2)
            elif button["action"] == "circle":
                # Draw circle icon
                center = (button["x"] + self.button_size // 2, button["y"] + self.button_size // 2)
                cv2.circle(self.toolbar, center, self.button_size // 3, button["color"], 2)
            elif button["action"] == "save":
                # Draw save icon (floppy disk-like symbol)
                top_left = (button["x"] + 10, button["y"] + 10)
                bottom_right = (button["x"] + self.button_size - 10, button["y"] + self.button_size - 10)
                cv2.rectangle(self.toolbar, top_left, bottom_right, button["color"], 2)
                cv2.rectangle(
                    self.toolbar,
                    (button["x"] + 15, button["y"] + 15),
                    (button["x"] + self.button_size - 15, button["y"] + 25),
                    button["color"],
                    -1
                )
            elif button["action"] == "load":
                # Draw load icon (folder-like symbol)
                # Folder base
                pts = np.array([
                    [button["x"] + 10, button["y"] + 25],
                    [button["x"] + 18, button["y"] + 15],
                    [button["x"] + self.button_size - 10, button["y"] + 15],
                    [button["x"] + self.button_size - 10, button["y"] + self.button_size - 10],
                    [button["x"] + 10, button["y"] + self.button_size - 10]
                ])
                cv2.polylines(self.toolbar, [pts], True, button["color"], 2)
            elif button["action"] == "stroke_erase":
                # Draw stroke eraser icon
                center_x = button["x"] + self.button_size // 2
                center_y = button["y"] + self.button_size // 2
                cv2.line(self.toolbar, (center_x - 10, center_y - 10), (center_x + 10, center_y + 10), button["color"], 2)
                cv2.line(self.toolbar, (center_x - 10, center_y + 10), (center_x + 10, center_y - 10), button["color"], 2)
        
        # Draw color palette
        palette_start_x = 7 * self.button_margin + 6 * self.button_size
        color_square_size = 20
        spacing = 5
        
        for i, color in enumerate(self.colors):
            # Calculate position
            x = palette_start_x + i * (color_square_size + spacing)
            y = self.toolbar_height // 2 - color_square_size // 2
            
            # Draw color square
            cv2.rectangle(
                self.toolbar,
                (x, y),
                (x + color_square_size, y + color_square_size),
                color,
                cv2.FILLED
            )
            
            # Highlight selected color
            if i == self.selected_color_index:
                cv2.rectangle(
                    self.toolbar,
                    (x - 2, y - 2),
                    (x + color_square_size + 2, y + color_square_size + 2),
                    (255, 255, 255),
                    2
                )
        
        # Draw current brush size indicator
        brush_indicator_x = palette_start_x + len(self.colors) * (color_square_size + spacing) + 20
        brush_indicator_y = self.toolbar_height // 2
        
        cv2.circle(
            self.toolbar,
            (brush_indicator_x, brush_indicator_y),
            self.thickness // 2,
            self.colors[self.selected_color_index],  # Use current selected color
            -1
        )
        cv2.putText(
            self.toolbar,
            f"Size: {self.thickness}",
            (brush_indicator_x + 15, brush_indicator_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return self.toolbar
    
    def run(self):
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("Failed to capture video")
                    break
                
                # Flip the frame horizontally for a more natural feel
                frame = cv2.flip(frame, 1)
                
                # Detect hands in the frame
                frame, hand_landmarks = self.detect_hands(frame)
                
                # Detect gestures
                is_drawing = self.detect_gesture(hand_landmarks, frame.shape)
                
                # Draw on canvas if in drawing mode
                self.draw_on_canvas()
                
                # Create toolbar
                toolbar = self.draw_toolbar()
                
                # Combine the toolbar and frame
                combined_frame = np.vstack([toolbar, frame])
                
                # Display the current mode and other info
                cv2.putText(
                    combined_frame, 
                    f"Mode: {self.mode.capitalize()} | Brush Size: {self.thickness}", 
                    (10, self.toolbar_height + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Combine canvas or shape preview with the frame
                combined_canvas = cv2.addWeighted(self.canvas, 1, self.temp_canvas, 1, 0)
                mask = (combined_canvas > 0).any(axis=2)
                
                # Apply canvas only to the frame part, not the toolbar
                frame_with_canvas = combined_frame.copy()
                frame_height = frame.shape[0]
                
                # Ensure mask dimensions match the frame area (excluding toolbar)
                if np.any(mask) and mask.shape == (frame_height, self.img_width):
                    frame_with_canvas[self.toolbar_height:, :][mask] = cv2.addWeighted(
                        frame_with_canvas[self.toolbar_height:, :][mask], 
                        0.5, 
                        combined_canvas[mask], 
                        0.5, 
                        0
                    )
                
                # Show the combined frame
                cv2.imshow("Hand Gesture Drawing", frame_with_canvas)
                
                # Press 'q' to quit, 'c' to clear canvas
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.canvas = np.zeros_like(self.canvas)
                    self.temp_canvas = np.zeros_like(self.temp_canvas)
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = HandGestureDrawing()
        app.run()
    except Exception as e:
        print(f"Error initializing application: {e}")
        cv2.destroyAllWindows()