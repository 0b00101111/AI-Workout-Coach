import numpy as np

class AISpotter:
    def __init__(self):
        self.feedback = ""
        self.color = (0, 255, 0)

    def calculate_angle(self, a, b, c):
        """Calculates angle 0-180 degrees at vertex b"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360 - angle
        return angle

    def analyze_frame(self, pose_name, landmarks):
        self.feedback = ""
        self.color = (0, 255, 0) # Green
        is_perfect = False       # Guilty until proven innocent

        def get_xy(idx): return [landmarks[idx].x, landmarks[idx].y]

        nose  = get_xy(0)
        mouth = get_xy(9)
        shldr = get_xy(11)
        elbow = get_xy(13)
        wrist = get_xy(15)
        hip   = get_xy(23)
        knee  = get_xy(25)
        ankle = get_xy(27)

        # --- EXERCISE LOGIC ---
        if pose_name == 'barbell biceps curl':
            # Error 1: Elbows swinging
            if abs(elbow[0] - shldr[0]) > 0.12:
                self.feedback, self.color = "Fix Elbows!", (0, 0, 255)
            # Error 2: Leaning back
            elif abs(shldr[0] - hip[0]) > 0.15:
                self.feedback, self.color = "Stand Tall!", (0, 165, 255)

            # Perfect Check: Hand is high (curled) & no errors
            if not self.feedback and wrist[1] < shldr[1]:
                is_perfect = True

        elif pose_name == 'push-up':
            # Error 1: Sagging hips
            if self.calculate_angle(shldr, hip, ankle) < 160:
                self.feedback, self.color = "Straighten Back!", (0, 0, 255)

            # Perfect Check: Arms extended at top & no errors
            if not self.feedback and self.calculate_angle(shldr, elbow, wrist) > 165:
                is_perfect = True

        elif pose_name == 'shoulder press':
            # Only analyze if hands are pushing up (above the mouth)
            # This prevents it from yelling at you while you're resting at the bottom.
            hands_up = wrist[1] < mouth[1]

            if hands_up:
                elbow_angle = self.calculate_angle(shldr, elbow, wrist)

                # Error: Hands up but elbows bent (Partial Rep)
                if elbow_angle < 140:
                    self.feedback, self.color = "Extend Arms!", (0, 0, 255)

                # Perfect: Hands up AND elbows straight
                elif elbow_angle > 165:
                    is_perfect = True

            # Always check for arching back
            if abs(shldr[0] - hip[0]) > 0.2:
                self.feedback, self.color = "Don't Arch!", (0, 165, 255)

        elif pose_name == 'squat':
            knee_angle = self.calculate_angle(hip, knee, ankle)

            # Only analyze depth if user is actually descending (knee bent < 150)
            if knee_angle < 150:
                # Error: Hip above knee
                if hip[1] < knee[1]:
                    if knee_angle < 110: # Only warn if they are somewhat deep
                        self.feedback, self.color = "Go Lower!", (0, 165, 255)

                # Perfect: Hip below knee & no errors
                elif not self.feedback:
                    is_perfect = True

            # Always check for chest falling
            if abs(shldr[0] - hip[0]) > 0.3:
                self.feedback, self.color = "Chest Up!", (0, 0, 255)

        return self.feedback, self.color, is_perfect