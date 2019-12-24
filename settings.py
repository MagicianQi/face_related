# Model
FACE_DETECTION_SERVER_URL = 'http://face_detection:6902/v1/models/face_detection:predict'
FACE_VERIFICATION_SERVER_URL = 'http://face_verification:8541/v1/models/face_verification:predict'
FACE_CALIBRATION_MODEL_PATH = 'models/shape_predictor_68_face_landmarks.dat'
FACE_DETECTION_THRESHOLD = 0.6
IMAGE_SIZE = 160

# GIF
GIF_FRAME_INTERVAL = 5
GIF_MAX_FRAME = 5

# Test Image Config
FONT_FILE_PATH = "static/simhei.ttf"
SAVE_TEMP_IMAGE_PATH = "static/temporary/"
FONT_SIZE = 12
FONT_COLOR = (0, 255, 255)
LINE_COLOR = (255, 0, 0)
TEMP_IMAGE_RANDOM_NAME_LENGTH = 8

# Matching Threshold

MATCH_THRESHOLD = 0.80
