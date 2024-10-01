import cv2
import dlib
import numpy as np

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor.dat")

# Function to extract landmarks as NumPy array
def get_landmarks(image, face):
    landmarks = predictor(image, face)
    points = []
    for i in range(68):
        points.append((landmarks.part(i).x, landmarks.part(i).y))
    return np.array(points)

# Read the image with two faces
image = cv2.imread(r"C:\Users\oshika\Documents\Python Scripts\FaceSwap\image2.jpg")

# Detect faces
faces = detector(image)
if len(faces) < 2:
    print("Error: The image must contain at least two faces for swapping.")
else:
    # Get landmarks for both faces
    landmarks1 = get_landmarks(image, faces[0])
    landmarks2 = get_landmarks(image, faces[1])

    # Create convex hulls for both sets of landmarks
    hull1 = cv2.convexHull(landmarks1)
    hull2 = cv2.convexHull(landmarks2)

    # Calculate bounding rectangles
    x1, y1, w1, h1 = cv2.boundingRect(hull1)
    x2, y2, w2, h2 = cv2.boundingRect(hull2)

    # Create masks for both faces
    mask1 = np.zeros_like(image)
    mask2 = np.zeros_like(image)
    cv2.fillConvexPoly(mask1, hull1, (255, 255, 255))
    cv2.fillConvexPoly(mask2, hull2, (255, 255, 255))

    # Get centers for seamless cloning
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)

    # Perform seamless cloning to swap faces
    swapped_face1 = cv2.seamlessClone(image, image, mask1, center2, cv2.NORMAL_CLONE)
    swapped_face2 = cv2.seamlessClone(image, swapped_face1, mask2, center1, cv2.NORMAL_CLONE)

    # Display the result
    cv2.imshow("Face Swap", swapped_face2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
