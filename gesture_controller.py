Wimport cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque

class GestureController:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture state management
        self.current_gesture = "NONE"
        self.previous_gesture = "NONE"
        self.gesture_buffer = deque(maxlen=5)  # Smoothing buffer
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Key states (prevent key spam)
        self.keys_pressed = set()
        
        # Gesture thresholds
        self.FIST_THRESHOLD = 0.15
        self.PALM_THRESHOLD = 0.6
        
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def is_fist(self, landmarks):
        """Detect closed fist - all fingers curled"""
        # Check if all fingertips are close to palm
        palm = landmarks[0]
        fingertips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
        
        distances = [self.calculate_distance(palm, tip) for tip in fingertips]
        avg_distance = np.mean(distances)
        
        return avg_distance < self.FIST_THRESHOLD
    
    def is_open_palm(self, landmarks):
        """Detect open palm - all fingers extended"""
        palm = landmarks[0]
        fingertips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
        
        distances = [self.calculate_distance(palm, tip) for tip in fingertips]
        avg_distance = np.mean(distances)
        
        return avg_distance > self.PALM_THRESHOLD
    
    def get_hand_position(self, landmarks):
        """Get hand position (left, center, right)"""
        wrist = landmarks[0]
        
        if wrist.x < 0.35:
            return "LEFT"
        elif wrist.x > 0.65:
            return "RIGHT"
        else:
            return "CENTER"
    
    def detect_gesture(self, hand_landmarks):
        """Main gesture detection logic"""
        if not hand_landmarks:
            return "NONE"
        
        landmarks = hand_landmarks.landmark
        
        # Get hand position
        position = self.get_hand_position(landmarks)
        
        # Detect gesture types
        is_fist = self.is_fist(landmarks)
        is_palm = self.is_open_palm(landmarks)
        
        # Gesture mapping
        if is_fist:
            return "BRAKE"
        elif is_palm:
            return "BOOST"
        elif position == "LEFT":
            return "STEER_LEFT"
        elif position == "RIGHT":
            return "STEER_RIGHT"
        else:
            return "FORWARD"
    
    def smooth_gesture(self, gesture):
        """Apply temporal smoothing to reduce jitter"""
        self.gesture_buffer.append(gesture)
        
        # Most common gesture in buffer wins
        if len(self.gesture_buffer) >= 3:
            gesture_counts = {}
            for g in self.gesture_buffer:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            return max(gesture_counts, key=gesture_counts.get)
        return gesture
    
    def execute_action(self, gesture):
        """Map gestures to keyboard actions"""
        # Release all keys first
        for key in list(self.keys_pressed):
            pyautogui.keyUp(key)
            self.keys_pressed.remove(key)
        
        # Press appropriate keys based on gesture
        if gesture == "FORWARD":
            pyautogui.keyDown('w')
            self.keys_pressed.add('w')
        elif gesture == "STEER_LEFT":
            pyautogui.keyDown('w')
            pyautogui.keyDown('a')
            self.keys_pressed.add('w')
            self.keys_pressed.add('a')
        elif gesture == "STEER_RIGHT":
            pyautogui.keyDown('w')
            pyautogui.keyDown('d')
            self.keys_pressed.add('w')
            self.keys_pressed.add('d')
        elif gesture == "BOOST":
            pyautogui.keyDown('w')
            pyautogui.keyDown('shift')
            self.keys_pressed.add('w')
            self.keys_pressed.add('shift')
        elif gesture == "BRAKE":
            pyautogui.keyDown('s')
            self.keys_pressed.add('s')
    
    def draw_ui(self, frame, gesture, hand_landmarks):
        """Draw UI overlay on frame"""
        h, w, _ = frame.shape
        
        # Draw hand landmarks
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # Gesture display
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Control legend
        legend_y = 100
        cv2.putText(frame, "Controls:", (20, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Gesture zone visualization
        cv2.line(frame, (int(w*0.35), 0), (int(w*0.35), h), (255, 255, 0), 2)
        cv2.line(frame, (int(w*0.65), 0), (int(w*0.65), h), (255, 255, 0), 2)
        
        cv2.putText(frame, "LEFT", (int(w*0.15), 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "CENTER", (int(w*0.45), 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "RIGHT", (int(w*0.75), 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Instructions
        instructions = [
            "Fist = BRAKE",
            "Open Palm = BOOST",
            "Hand Left = STEER LEFT",
            "Hand Right = STEER RIGHT",
            "Center = FORWARD",
            "Press 'Q' to quit"
        ]
        
        y_offset = h - 150
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (w - 280, y_offset + i*25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main control loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        print("🎮 Gesture Controller Started!")
        print("=" * 50)
        print("Position your hand in frame:")
        print("  • LEFT zone = Steer Left")
        print("  • CENTER zone = Forward")
        print("  • RIGHT zone = Steer Right")
        print("  • Make FIST = Brake")
        print("  • Open PALM = Boost")
        print("=" * 50)
        print("Press 'Q' to quit\n")
        
        # Wait for game to be focused
        print("You have 3 seconds to focus on your game window...")
        time.sleep(3)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            # Detect gesture
            gesture = "NONE"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                gesture = self.detect_gesture(hand_landmarks)
                gesture = self.smooth_gesture(gesture)
            else:
                gesture = self.smooth_gesture("NONE")
            
            # Execute action only if gesture changed
            if gesture != self.previous_gesture:
                self.execute_action(gesture)
                self.previous_gesture = gesture
            
            self.current_gesture = gesture
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
            
            # Draw UI
            if results.multi_hand_landmarks:
                frame = self.draw_ui(frame, gesture, results.multi_hand_landmarks[0])
            else:
                frame = self.draw_ui(frame, "NO HAND DETECTED", None)
            
            # Display
            cv2.imshow('Gesture Controller', frame)
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        for key in list(self.keys_pressed):
            pyautogui.keyUp(key)
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n✅ Controller stopped successfully!")

if __name__ == "__main__":
    controller = GestureController()
    controller.run()
