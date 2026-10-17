from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import cv2
import mediapipe as mp
import math
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

keyboard = KeyboardController()
mouse = MouseController()

from voice_engine import VoiceEngine

voice: "VoiceEngine | None" = None
recording_on = False
is_clicking = False

model_path = './models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = vision.HandLandmarksConnections
mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles
latest_result = None

# Landmarks
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20
WRIST = 0

# Extra knuckle landmarks (PIP = mid joint, MCP = base knuckle)
THUMB_IP = 3
INDEX_PIP = 6
INDEX_MCP = 5
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18
PINKY_MCP = 17


def get_distance(point1, point2) -> float:
    return math.hypot(point1.x - point2.x, point1.y - point2.y)


def get_pix_coords(landmark, frame_shape) -> tuple[int, int]:
    h, w, _ = frame_shape
    return int(landmark.x * w), int(landmark.y * h)


def _get_screen_size() -> tuple[int, int]:
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    size = (root.winfo_screenwidth(), root.winfo_screenheight())
    root.destroy()
    return size


SCREEN_WIDTH, SCREEN_HEIGHT = _get_screen_size()


def update_mouse_position(index_tip) -> None:
    """Moves the system cursor to match the index fingertip's normalized (0..1) position."""
    x = min(max(index_tip.x, 0.0), 1.0) * SCREEN_WIDTH
    y = min(max(index_tip.y, 0.0), 1.0) * SCREEN_HEIGHT
    mouse.position = (int(x), int(y))


def is_finger_extended(landmarks, tip_idx, pip_idx) -> bool:
    wrist = landmarks[WRIST]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    return get_distance(wrist, tip) > get_distance(wrist, pip) * 1.15


def classify_gesture(landmarks) -> str:
    """
    Classifies static poses from your Must-Finish list:
      - 'PINCH'     : Thumb tip (4) and Index tip (8) pinched together
      - 'CLUMPED'   : All 5 fingertips bunched close together (start of Explainer gesture)
      - 'FIST'      : All 4 fingers curled in (Mic toggle / Brain dump)
      - 'OPEN_PALM' : All 4 fingers extended & spread (Slow down / Pause gesture)
      - 'POINTING'  : Only Index finger extended (Reading guide gesture)
    """
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    ring_tip = landmarks[RING_FINGER_TIP]
    pinky_tip = landmarks[PINKY_TIP]

    # Reference hand scale based on distance from wrist to index MCP
    hand_scale = get_distance(landmarks[WRIST], landmarks[INDEX_MCP]) + 1e-6

    index_ext = is_finger_extended(landmarks, INDEX_FINGER_TIP, INDEX_PIP)
    middle_ext = is_finger_extended(landmarks, MIDDLE_FINGER_TIP, MIDDLE_PIP)
    ring_ext = is_finger_extended(landmarks, RING_FINGER_TIP, RING_PIP)
    # FIX: Use PINKY_PIP (18) instead of PINKY_MCP (17) so fists register reliably
    pinky_ext = is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP)

    # 1. CLUMPED FINGERTIPS (Explainer bot pre-pose: all 5 tips close to each other)
    avg_x = (thumb_tip.x + index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 5
    avg_y = (thumb_tip.y + index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 5
    max_tip_spread = max(
        math.hypot(lm.x - avg_x, lm.y - avg_y)
        for lm in (thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip)
    )
    if max_tip_spread < hand_scale * 0.42:
        return "CLUMPED"

    # 2. PINCH (Thumb tip + Index tip touching, for Pinch-and-drag text selection)
    pinch_dist = get_distance(thumb_tip, index_tip)
    if pinch_dist < hand_scale * 0.35:
        return "PINCH"

    # 3. FIST (All 4 main fingers curled in -> Activate Mic / Brain Dump)
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "FIST"

    # 4. OPEN PALM (All 4 main fingers extended -> Pause TTS / Cancel Mic)
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "OPEN_PALM"

    # 5. POINTING (Only index extended -> Reading Guide gesture)
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "POINTING"

    return "NEUTRAL"


def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback,
)


def draw_landmarks(frame, hand_landmarks):
    """Draw the 21 hand points + connections onto the frame for visual debugging."""
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )

RECORDING_COOLDOWN_SECONDS = 1.0
_last_recording_trigger_time = 0.0


def _recording_trigger_ready() -> bool:
    """Debounces rapid re-detection of the same gesture (e.g. OPEN_PALM flickering frame to frame)."""
    global _last_recording_trigger_time
    now = time.time()
    if now - _last_recording_trigger_time < RECORDING_COOLDOWN_SECONDS:
        return False
    _last_recording_trigger_time = now
    return True


def start_recording():
    """Starts mic capture via VoiceEngine for brain-dump transcription."""
    global recording_on
    if voice is not None:
        voice.toggle_mic(mode="brain_dump")
    recording_on = True


def stop_recording():
    """Stops mic capture; VoiceEngine transcribes and pastes the result."""
    global recording_on
    if voice is not None:
        voice.toggle_mic(mode="brain_dump")
    recording_on = False


def run_command(key: str) -> None:
    global is_clicking

    # Release the held left click the moment the hand leaves the PINCH pose.
    if key != "PINCH" and is_clicking:
        mouse.release(Button.left)
        is_clicking = False

    match key:
        case "OPEN_PALM":
            if _recording_trigger_ready():
                if not recording_on:
                    keyboard.type('Brain dump: ')
                    start_recording()
                else:
                    stop_recording()
        case "CLUMPED":
            pass  #dosmthn
        case "NEUTRAL":
            pass  #dosmthn
        case "FIST":
            if _recording_trigger_ready():
                if not recording_on:
                    start_recording()
                else:
                    stop_recording()
        case "EXPLAIN_EXPAND":
            pass  #dosmthn
        case "WRIST_TWIST":
            pass  #dosmthn
        case "POINTING":
            pass  #dosmthn
        case "PINCH":
            if not is_clicking:
                mouse.press(Button.left)
                is_clicking = True

def main():
    global voice
    cap = cv2.VideoCapture(0)
    voice = VoiceEngine()

    with HandLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        last_timestamp_ms = -1

        # FIX: Keep state tracking variables OUTSIDE the while loop!
        last_clumped_time = 0.0
        prev_palm_z_diff = None
        last_twist_time = 0.0
        prev_pose = "NEUTRAL"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Ensure timestamp is strictly monotonically increasing for LIVE_STREAM
            timestamp_ms = int((time.time() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            landmarker.detect_async(mp_image, timestamp_ms)

            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

                    # 1. Access any specific coordinate (normalized 0..1 and pixel x, y):
                    index_tip = hand_landmarks[INDEX_FINGER_TIP]
                    ix_px, iy_px = get_pix_coords(index_tip, frame.shape)
                    update_mouse_position(index_tip)

                    # 2. Classify the current static pose:
                    pose = classify_gesture(hand_landmarks)
                    now = time.time()

                    # 3. Check for CLUMPED -> EXPANDED transition (Explainer Bot trigger):
                    if pose == "CLUMPED":
                        last_clumped_time = now
                    elif pose == "OPEN_PALM" and (now - last_clumped_time) < 0.6:
                        pose = "EXPLAIN_EXPAND"
                        last_clumped_time = 0.0

                    # 4. Check for WRIST TWIST (palm Z-depth flip between Index MCP #5 and Pinky MCP #17):
                    palm_z_diff = hand_landmarks[INDEX_MCP].z - hand_landmarks[PINKY_MCP].z
                    if prev_palm_z_diff is not None:
                        if (palm_z_diff * prev_palm_z_diff < 0) and abs(palm_z_diff - prev_palm_z_diff) > 0.08:
                            if now - last_twist_time > 1.2:
                                pose = "WRIST_TWIST"
                                last_twist_time = now
                    prev_palm_z_diff = palm_z_diff

                    # 5. Trigger actions ONLY when the gesture changes (edge trigger):
                    if pose != prev_pose:
                        print(f"[Gesture Triggered] {prev_pose} -> {pose}")
                        # Uncomment these lines to trigger your VoiceEngine:
                        # if pose == "FIST":
                        #     voice.toggle_mic(mode="brain_dump")
                        # elif pose == "EXPLAIN_EXPAND":
                        #     voice.read_highlighted_selection(explain_first=True)
                        # elif pose == "OPEN_PALM":
                        #     voice.pause_all()
                        prev_pose = pose

                    # 6. Display the live gesture & Index Tip coordinates on the camera feed:
                    cv2.putText(
                        frame,
                        f"Gesture: {pose} | Index: ({index_tip.x:.2f}, {index_tip.y:.2f})",
                        (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )

                    run_command(pose)

            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()