import cv2
import numpy as np
import time
from collections import deque

class HandTrackingSystem:
    def __init__(self, camera_id=0):
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Skin color range for hand detection (HSV)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Virtual object properties
        self.virtual_obj = {
            'center': (320, 240), 
            'width': 100,
            'height': 100,
            'color': (255, 0, 255),
            'thickness': 2
        }
        
        # State thresholds 
        self.safe_threshold = 150
        self.warning_threshold = 80
        self.danger_threshold = 30
        
        # System state
        self.state = "INITIALIZING"
        self.state_history = deque(maxlen=10)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Hand tracking history for smoothing
        self.hand_positions = deque(maxlen=5)
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
    def apply_skin_segmentation(self, frame):
        """Detect skin color regions using HSV color space"""
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
        
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return skin_mask, contours
    
    def detect_hand(self, contours):
        """Find the largest contour (most likely hand) and its properties"""
        if not contours:
            return None, None, None

        largest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest_contour)
 
        if area < 2000:
            return None, None, None
        
        # Calculate moments to find the centroid
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:

            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = x + w // 2
            cy = y + h // 2
        
        # Find convex hull for better hand shape representation
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        
        # Calculate convexity defects to find finger points
        defects = []
        if len(hull) > 3:
            try:
                defects = cv2.convexityDefects(largest_contour, hull)
            except:
                defects = None
        
        return (cx, cy), largest_contour, defects
    
    def calculate_distance_to_object(self, point):
        """Calculate minimum distance from point to virtual object boundary"""
        x, y = point
        obj_x, obj_y = self.virtual_obj['center']
        width, height = self.virtual_obj['width'], self.virtual_obj['height']
        
        # Calculate distance to rectangle edges
        left = obj_x - width // 2
        right = obj_x + width // 2
        top = obj_y - height // 2
        bottom = obj_y + height // 2
        
        # Check if point is inside rectangle
        if left <= x <= right and top <= y <= bottom:
            
            dist_to_left = abs(x - left)
            dist_to_right = abs(x - right)
            dist_to_top = abs(y - top)
            dist_to_bottom = abs(y - bottom)
            return min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

        # Find nearest x coordinate on rectangle
        nearest_x = max(left, min(x, right))
        # Find nearest y coordinate on rectangle
        nearest_y = max(top, min(y, bottom))
        
        # Euclidean distance to nearest point on rectangle
        distance = np.sqrt((x - nearest_x)**2 + (y - nearest_y)**2)
        
        return distance
    
    def determine_state(self, distance):
        """Determine state based on distance to virtual object"""
        if distance is None:
            return "SEARCHING"
        elif distance > self.safe_threshold:
            return "SAFE"
        elif distance > self.warning_threshold:
            return "WARNING"
        else:
            return "DANGER"
    
    def draw_virtual_object(self, frame):
        """Draw virtual object on frame"""
        x, y = self.virtual_obj['center']
        w, h = self.virtual_obj['width'] // 2, self.virtual_obj['height'] // 2
        
        # Draw rectangle
        cv2.rectangle(frame, 
                     (x - w, y - h), 
                     (x + w, y + h), 
                     self.virtual_obj['color'], 
                     self.virtual_obj['thickness'])
        
        # label
        cv2.putText(frame, "VIRTUAL OBJECT", 
                   (x - w, y - h - 10), 
                   self.font, 0.5, self.virtual_obj['color'], 1)
    
    def draw_state_overlay(self, frame, state, hand_pos=None, distance=None):
        """Draw state information and hand position on frame"""

        if state == "SAFE":
            color = (0, 255, 0)  # Green
            text = "STATE: SAFE"
        elif state == "WARNING":
            color = (0, 255, 255)  # Yellow
            text = "STATE: WARNING"
        elif state == "DANGER":
            color = (0, 0, 255)  # Red
            text = "STATE: DANGER"

            danger_text = "DANGER DANGER"
            text_size = cv2.getTextSize(danger_text, self.font, 1.2, 3)[0]
            text_x = frame.shape[1] // 2 - text_size[0] // 2
            text_y = frame.shape[0] // 2 - text_size[1] // 2
            
            # background for better visibility
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - 30), 
                         (text_x + text_size[0] + 10, text_y + text_size[1] + 10), 
                         (0, 0, 0), -1)
            
            # flashing effect 
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, danger_text, 
                           (text_x, text_y + text_size[1]), 
                           self.font, 1.2, color, 3)
        else:
            color = (255, 255, 255)  
            text = f"STATE: {state}"
        
        # Draw main state
        cv2.putText(frame, text, (10, 30), 
                   self.font, self.font_scale, color, self.font_thickness)
        
        # Draw hand position 
        if hand_pos:
           
            cv2.circle(frame, hand_pos, 10, (0, 255, 255), -1)
            cv2.circle(frame, hand_pos, 12, (255, 255, 255), 2)
            cv2.putText(frame, "HAND", 
                       (hand_pos[0] + 15, hand_pos[1]), 
                       self.font, 0.5, (255, 255, 255), 1)
            
            # Draw distance line to object if not in SEARCHING state
            if state not in ["SEARCHING", "INITIALIZING"] and distance is not None:
                cv2.line(frame, hand_pos, self.virtual_obj['center'], 
                        (255, 255, 255), 2, cv2.LINE_AA)
                
                
                distance_text = f"Distance: {int(distance)} px"
                cv2.putText(frame, distance_text, 
                           (hand_pos[0] + 20, hand_pos[1] + 40), 
                           self.font, 0.5, (255, 255, 255), 1)
        
        # Draw thresholds visualization
        self.draw_thresholds(frame)
        
        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30), 
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        #instructions
        instructions = "Move hand close to purple box"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 10), 
                   self.font, 0.5, (200, 200, 200), 1)
    
    def draw_thresholds(self, frame):
        """Visualize distance thresholds around virtual object"""
        x, y = self.virtual_obj['center']
        
        #warning zone
        cv2.circle(frame, (x, y), self.warning_threshold, 
                  (0, 255, 255, 100), 2)  # Yellow warning zone
        cv2.putText(frame, "WARNING ZONE", 
                   (x - 60, y - self.warning_threshold - 20), 
                   self.font, 0.4, (0, 255, 255), 1)
        
        #danger zone
        cv2.circle(frame, (x, y), self.danger_threshold, 
                  (0, 0, 255, 100), 2)  # Red danger zone
        cv2.putText(frame, "DANGER ZONE", 
                   (x - 50, y - self.danger_threshold - 20), 
                   self.font, 0.4, (0, 0, 255), 1)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """Main loop for the hand tracking system"""
        print("Starting Hand Tracking System...")
        print("Move your hand close to the purple virtual object.")
        print("Press 'q' to quit.")
        
        while True:
            # Read frame 
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally 
            frame = cv2.flip(frame, 1)
            
            # copy for display
            display_frame = frame.copy()
            
            # skin segmentation
            skin_mask, contours = self.apply_skin_segmentation(frame)
            
            # Detect hand position
            hand_pos, hand_contour, defects = self.detect_hand(contours)
            
            # Smooth hand position 
            if hand_pos:
                self.hand_positions.append(hand_pos)
                if len(self.hand_positions) > 1:
                    # Average the last few positions for smoothing
                    avg_x = int(np.mean([p[0] for p in self.hand_positions]))
                    avg_y = int(np.mean([p[1] for p in self.hand_positions]))
                    hand_pos = (avg_x, avg_y)
            else:
                self.hand_positions.clear()
            
            # Calculate distance
            distance = None
            if hand_pos:
                distance = self.calculate_distance_to_object(hand_pos)
                self.state = self.determine_state(distance)
            else:
                self.state = "SEARCHING"
            
         
            self.state_history.append(self.state)
            
            # virtual object
            self.draw_virtual_object(display_frame)
            
            # overlay and hand information
            self.draw_state_overlay(display_frame, self.state, hand_pos, distance)
            
            # hand contour if detected
            if hand_contour is not None:
                cv2.drawContours(display_frame, [hand_contour], -1, (0, 255, 0), 2)
            
            # skin mask in a small window
            if skin_mask is not None:
                mask_display = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
                mask_display = cv2.resize(mask_display, (160, 120))
                display_frame[10:130, 10:170] = mask_display
                cv2.putText(display_frame, "Skin Mask", (10, 140), 
                           self.font, 0.4, (255, 255, 255), 1)
            
            # Update FPS
            self.update_fps()
            
            # Display the frame
            cv2.imshow('Hand Tracking System', display_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")

def main():
    
    system = HandTrackingSystem(camera_id=0)
    system.run()

if __name__ == "__main__":
    main()