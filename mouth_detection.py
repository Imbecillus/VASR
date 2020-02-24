import dlib
import imageio
import sys
from PIL import Image

predictor_path = 'dlib_stuff/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_face_bounding_box(img):
    dets = detector(img, 1)
    return dets[0]

def crop_face(img):
    try:
        face = get_face_bounding_box(img)
    except:
        print('No face detected.')
        return None, True

    # Get face bounding box
    left = face.left()
    right = face.right()
    top = face.top()
    bottom = face.bottom()

    # Add 10% padding
    width = right - left
    height = bottom - top

    x_padding = 650 - width
    y_padding = 650 - height

    left = left - int(x_padding / 2)
    right = right + int(x_padding / 2)
    top = top - int(y_padding / 2)
    bottom = bottom + int(y_padding / 2)

    # Ensure box doesn't extend outside the image
    if left < 0:
        left = 0
    if right > 1000:
        right = 1000
    if top < 0:
        top = 0
    if bottom > 1000:
        bottom = 1000

    # Crop and return image
    face_img = img[top:bottom, left:right]
    return face_img, False

def crop_mouth(img, m_top = None, m_bottom = None, m_left = None, m_right = None):
    # Get bounding box if none is given
    if m_top is None or m_bottom is None or m_left is None or m_right is None:
        m_top, m_bottom, m_left, m_right = get_mouth_bounding_box(img)
        if m_top == -1:
            return None, False

    # Crop and return image
    mouth_img = img[m_top:m_bottom, m_left:m_right]
    return mouth_img, True

def get_mouth_bounding_box(img):
    # Get face bounding box
    try:
        face = get_face_bounding_box(img)
    except:
        print('No face detected.')
        return None, None, None, None

    # Get mouth bounding box
    shape = predictor(img, face)
    m_left = shape.part(48).x
    m_right = shape.part(54).x
    m_top = shape.part(31).y
    m_bottom = shape.part(57).y

    # Add padding
    m_width = m_right - m_left
    m_height = m_bottom - m_top
    m_left = m_left - int(m_width / 2)
    m_right = m_right + int(m_width / 2)
    m_top = m_top
    m_bottom = m_bottom + int(m_height * 0.33)

    return m_top, m_bottom, m_left, m_right

if sys.argv[0] == 'mouth_detection.py':
    # Script run directly; execute test code

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print('Testing ROI detection...')

        img = imageio.imread(path).astype('uint8')

        face_img, _ = crop_face(img)
        mouth_img, test = crop_mouth(img)

        imageio.imwrite('test_face.jpg', face_img)
        imageio.imwrite('test_mouth.jpg', mouth_img)

        print('Done.')