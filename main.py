import cv2
import os
import sys
import numpy as np

from argparse import ArgumentParser


def nothing(x):
    pass


# Directory to store output images
BASE_DIR = os.path.join(os.getcwd(), 'data', 'output')

# Current frame and frequency to read the frames
FRAME_COUNT = 0
FRAME_FREQ = 1

# Object area range
MIN_AREA = 9000
MAX_AREA = 50000

# Lower and upper HSV values for object color detection
LOW_H = 0
LOW_S = 0
LOW_V = 45
HIGH_H = 179
HIGH_S = 255
HIGH_V = 255

# HSV trackbars
# cv2.namedWindow('Thresh')
# cv2.createTrackbar('lowH', 'Thresh', LOW_H, 179, nothing)
# cv2.createTrackbar('lowS', 'Thresh', LOW_S, 255, nothing)
# cv2.createTrackbar('lowV', 'Thresh', LOW_V, 255, nothing)
# cv2.createTrackbar('highH', 'Thresh', HIGH_H, 179, nothing)
# cv2.createTrackbar('highS', 'Thresh', HIGH_S, 255, nothing)
# cv2.createTrackbar('highV', 'Thresh', HIGH_V, 255, nothing)

# Lower and upper value of color Range of the object
# for color thresholding to detect the object
LOWER_COLOR_RANGE = (0, 0, 0)
UPPER_COLOR_RANGE = (174, 73, 255)

# Keep count of objects
COUNT_OBJECT = 0
OBJECT_COUNT = "Object number: {}".format(COUNT_OBJECT)

# Array to store all defects found
OBJ_DEFECT = []

SHOW_STEPS = False


def build_argparser():
    """
    Parse the command line arguments.

    :return: command line arguments.
    """
    parser = ArgumentParser()

    parser.add_argument('-i', '--input',
                        required=False,
                        help="Path to the video file. Leave it blank to use default camera.")

    parser = parser.parse_args()

    return parser


def detect_color(frame, cnt):
    """
    Identifies if the current object has a color defect.

    1. Increase the brightness of the image.
    2. Convert the image to HSV color space format, which gives more information about the colors of the image.
    3. Threshold the image based on the color using "inRange" function.
    4. Morphological opening and closing is done on the mask to remove noises and fill the gaps.
    5. Find the contours on the mask image

    :param frame: Input frame from video
    :param cnt: Contours of object
    :return: color_flag
    """
    color_flag = False

    # Increase the brightness of the image
    cv2.convertScaleAbs(frame, frame, 1, 20)

    if SHOW_STEPS:
        cv2.imshow('Bright', frame)

    # Convert the captured frame from BGR to HSV color space
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if SHOW_STEPS:
        cv2.imshow('Color hsv', img_hsv)

    # Threshold the image
    img_threshold = cv2.inRange(img_hsv, LOWER_COLOR_RANGE, UPPER_COLOR_RANGE)

    if SHOW_STEPS:
        cv2.imshow('Color thresh 1', img_threshold)

    # Morphological opening (remove small objects from the foreground)
    img_threshold = cv2.erode(img_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    img_threshold = cv2.dilate(img_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    if SHOW_STEPS:
        cv2.imshow('Color thresh 2', img_threshold)

    # Find the contours of the image
    image_contours, contours, hierarchy = cv2.findContours(img_threshold,
                                                           cv2.RETR_LIST,
                                                           cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])

        if 2000 < area < 10000:
            cv2.drawContours(frame, contours[i], -1, (0, 0, 255), 2)
            color_flag = True

    if SHOW_STEPS:
        cv2.imshow('Color out', frame)

    if color_flag:
        x, y, w, h = cv2.boundingRect(cnt)
        print("Color defect detected in object {}".format(COUNT_OBJECT))
        cv2.imwrite("{}/color/color_{}.png".format(BASE_DIR, COUNT_OBJECT), frame[y:y+h, x:x+w])
        cv2.putText(frame, OBJECT_COUNT, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, "Defect: Color", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow("Out", frame)
        cv2.waitKey(2000)

    return color_flag


def detect_cracks(frame, cnt):
    """
    Identify possible cracks in object.

    1. Convert the image to gray scale.
    2. Blur the gray image to remove the noises.
    3. Find the edges on the blurred image to get the contours of possible cracks.
    4. Filter the contours to get the contour of the crack.
    5. Draw the contour on the original image for visualization.

    :param frame: Input frame from video
    :param cnt: Contours of the object
    :return: crack_flag
    """
    crack_flag = False
    low_threshold = 130
    kernel_size = 3
    ratio = 3

    # Convert the captured frame from BGR to gray scale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if SHOW_STEPS:
        cv2.imshow('Crack gray', img)

    img = cv2.blur(img, (7, 7))

    if SHOW_STEPS:
        cv2.imshow('Crack blur', img)

    # Find the edges
    edges = cv2.Canny(img, low_threshold, low_threshold * ratio, kernel_size)

    if SHOW_STEPS:
        cv2.imshow('Crack edges', edges)

    # Find the contours
    image_contours, contours, hierarchy = cv2.findContours(edges,
                                                           cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            cv2.drawContours(frame, contours, i, (0, 255, 0), 2)
            if area > 20 or area < 9:
                crack_flag = True

        if SHOW_STEPS or 1:
            cv2.imshow('Cracks', frame)

        if crack_flag:
            x, y, w, h = cv2.boundingRect(cnt)
            print("Crack defect detected in object {}".format(COUNT_OBJECT))
            cv2.imwrite("{}/crack/crack_{}.png".format(BASE_DIR, COUNT_OBJECT), frame[y:y+h, x:x+w])
            cv2.putText(frame, OBJECT_COUNT, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, "Defect: Crack", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow("Out", frame)
            cv2.waitKey(2000)

    return crack_flag


def detect_flaws(cap):
    """
    Detect color variation or cracks in objects and store a picture of them in their respective directory.

    :return: None
    """
    global FRAME_COUNT
    global FRAME_FREQ
    global COUNT_OBJECT
    global OBJECT_COUNT

    while cap.isOpened():
        # Read the frame from the capture
        ret, frame = cap.read()

        if not ret:
            break

        FRAME_COUNT += 1

        cv2.imshow('Original', frame)

        frame_h, frame_w, frame_c = frame.shape

        black = np.zeros((frame_h, frame_w, 3), np.uint8)
        roi = cv2.rectangle(black,
                            (int(frame_w / 6), 0),
                            (int(frame_w / 6 + 50), frame_h),
                            (255, 255, 255),
                            -1)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 127, 255, 0)

        # Check every given frame number, based on the frequency of object on conveyor belt
        if FRAME_COUNT % FRAME_FREQ == 0:
            FRAME_FREQ = 1
            # Convert BGR image to HSV color space
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            if SHOW_STEPS:
                cv2.imshow('HSV', img_hsv)

            # LOW_H = cv2.getTrackbarPos('lowH', 'Thresh')
            # LOW_S = cv2.getTrackbarPos('lowS', 'Thresh')
            # LOW_V = cv2.getTrackbarPos('lowV', 'Thresh')
            # HIGH_H = cv2.getTrackbarPos('highH', 'Thresh')
            # HIGH_S = cv2.getTrackbarPos('highS', 'Thresh')
            # HIGH_V = cv2.getTrackbarPos('highV', 'Thresh')

            # Threshold of image in a specified color range
            img_threshold = cv2.inRange(img_hsv, (LOW_H, LOW_S, LOW_V), (HIGH_H, HIGH_S, HIGH_V))

            if SHOW_STEPS:
                cv2.imshow('Thresh', img_threshold)

            # Morphological opening (remove small objects from the foreground)
            img_threshold = cv2.erode(img_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            img_threshold = cv2.dilate(img_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            # Morphological closing (fill small objects from the foreground)
            img_threshold = cv2.dilate(img_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            img_threshold = cv2.erode(img_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            # if SHOW_STEPS:
            cv2.imshow('Closing', img_threshold)

            mask2 = cv2.bitwise_and(img_threshold, img_threshold, mask=mask)

            image_contours, mask_contours, hierarchy = cv2.findContours(mask2,
                                                                        cv2.RETR_LIST,
                                                                        cv2.CHAIN_APPROX_NONE)

            if len(mask_contours) != 0:
                print("Possible object found...")
                FRAME_FREQ = FRAME_COUNT + 40

                # Find the contours on the image
                image_contours, contours, hierarchy = cv2.findContours(img_threshold,
                                                                       cv2.RETR_LIST,
                                                                       cv2.CHAIN_APPROX_NONE)

                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)

                    if MAX_AREA > w * h > MIN_AREA:
                        COUNT_OBJECT += 1

                        frame_color = frame.copy()
                        frame_crack = frame.copy()
                        frame_flawless = frame.copy()

                        OBJECT_COUNT = "Object number: {}".format(COUNT_OBJECT)

                        # Check for the color defect of the object
                        color_flag = detect_color(frame_color, cnt)

                        if color_flag:
                            OBJ_DEFECT.append(str("Color"))

                        # Check for cracks in object
                        crack_flag = detect_cracks(frame_crack, cnt)

                        if crack_flag:
                            OBJ_DEFECT.append(str("Crack"))

                        # Check if no defects were found
                        if len(OBJ_DEFECT) < COUNT_OBJECT:
                            OBJ_DEFECT.append(str("No Defect"))
                            print("No defect detected in object {}".format(COUNT_OBJECT))
                            cv2.putText(frame_flawless, OBJECT_COUNT, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (255, 255, 255), 2)
                            cv2.putText(frame_flawless, "No defect detected", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (255, 255, 255), 2)
                            cv2.imwrite("{}/no_defect/no_defect_{}.png".format(BASE_DIR, COUNT_OBJECT),
                                        frame[y:y+h, x:x+w])

                if not OBJ_DEFECT:
                    continue

        key = cv2.waitKey(40)
        if key == 113 or key == 81:  # if key pressed is 'q' or 'Q', exit program
            break
        elif key == 112:  # if key pressed is 'p', pause program
            while 1:
                key2 = cv2.waitKey(0)
                if key2 == 111:  # if key pressed is 'o' while program is paused, resume program
                    break

    cv2.destroyAllWindows()
    cap.release()


def main():
    args = build_argparser()

    if args.input:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print("\nVideo file not found. Aborting...\n")
            sys.exit(0)
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("\nCamera not plugged in or not working. Aborting...\n")
            sys.exit(0)

    detect_flaws(cap)


if __name__ == '__main__':
    main()
