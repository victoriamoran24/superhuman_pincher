from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils, hands as mp_hands_styles

model_path = './models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

#Landmarks

THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20
WRIST = 0


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
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in hand_landmarks
    ])
    drawing_utils.draw_landmarks(
        frame, proto, mp_hands_styles.HAND_CONNECTIONS
    )


def main():
    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.time() - start_time) * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()