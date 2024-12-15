import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def filter_boxes_by_class(boxes, cls_id):
    """
    Filters the bounding boxes based on the specified class ID e.g. cls_id=2 for cars.

    Args:
        boxes: A YOLO boxes object.
        cls_id: The class ID to filter the boxes by.

    Returns:
        A list of xyxy tensors corresponding to the specified class ID.
    """
    mask = boxes.cls == cls_id
    filtered_xyxy = boxes.xyxy[mask]
    return filtered_xyxy.cpu().tolist()

def straighten_skewed_rectangle(img):
    # written by chatGPT

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) < 2:
        print("Not enough lines found. returning the original image")
        return img

    # Find the longest lines
    longest_lines = sorted(lines, key=lambda l: np.linalg.norm((l[0][2]-l[0][0], l[0][3]-l[0][1])), reverse=True)[:2]

    # Calculate angles of the longest lines
    angles = []
    for line in longest_lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Average the angles to get a more stable rotation estimate
    average_angle = np.mean(angles)

    # Rotate the image by the negative of the average angle
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, average_angle, 1.0)

    # Calculate the size of the new image to include padding
    cos_theta = abs(M[0, 0])
    sin_theta = abs(M[0, 1])
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]

    # Perform the rotation and padding
    rotated_img = cv2.warpAffine(img, M, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated_img

def show_resized_images(resized_imgs, cropped_license_image):
    # draw the cropped license plate image
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(cropped_license_image)
    plt.show()

    plt.figure(figsize=(6,1/8*6))
    plt.title("License Plate Digits")
    plt.axis('off')
    for i in range(len(resized_imgs)):
        plt.subplot(1, len(resized_imgs), i+1)
        plt.axis('off')
        plt.imshow(resized_imgs[i], cmap='gray')
    plt.show()

def extract_digits(image_name, min_area=0.005, max_area=0.05, debug=False, show=False, prefix="image"):
    # find the biggest car in the image
    yolo_model = YOLO("yolo11n.pt")
    car_results = yolo_model(image_name)
    car_boxes = filter_boxes_by_class(car_results[0].boxes, 2) # 2: car id

    if len(car_boxes) == 0:
        raise ValueError("No cars detected in the image.")

    # sort by area
    sorted_car_boxes = sorted(car_boxes, key=lambda x: abs(x[2] - x[0]) * abs(x[3] - x[1]), reverse=True)

    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # draw the biggest car's bbox
    car_bbox = sorted_car_boxes[0]
    img_drawn = cv2.rectangle(img.copy(), (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (255, 255, 0), 6)

    if debug:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(img_drawn)
        plt.title("Detect the Car")
        plt.savefig(f"{prefix}_car_bbox.png", bbox_inches='tight')
        if show:
            plt.show()

    # crop the car from the image
    cropped_car_image = img[int(car_bbox[1]):int(car_bbox[3]), int(car_bbox[0]):int(car_bbox[2])]

    if debug:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(cropped_car_image)
        plt.title("Cropped Car Image")
        plt.savefig(f"{prefix}_cropped_car_image.png", bbox_inches='tight')
        if show:
            plt.show()

    # detect the license plate in the cropped car image
    model = YOLO("yolo11_anpr_ghd.pt")
    results = model(cropped_car_image)
    license_box = results[0].boxes[0].xyxy[0].cpu().numpy()

    img_drawn = cv2.rectangle(cropped_car_image.copy(), (int(license_box[0]), int(license_box[1])), (int(license_box[2]), int(license_box[3])), (255, 255, 0), 3)

    if debug:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(img_drawn)
        plt.title("Detect the License Plate")
        plt.savefig(f"{prefix}_license_plate_bbox.png", bbox_inches='tight')
        if show:
            plt.show()
    
    # Crop the image based on the bounding box
    cropped_license_image = cropped_car_image[int(license_box[1]):int(license_box[3]), int(license_box[0]):int(license_box[2])]

    if debug:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(cropped_license_image)
        plt.title("Cropped License Plate Image")
        plt.savefig(f"{prefix}_cropped_license_plate_image.png", bbox_inches='tight')
        if show:
            plt.show()

    # straighten the license plate image
    straight_license_plate_img = straighten_skewed_rectangle(cropped_license_image)

    if debug:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(straight_license_plate_img)
        plt.title("Aligned License Plate Image")
        plt.savefig(f"{prefix}_straight_license_plate_image.png", bbox_inches='tight')
        if show:
            plt.show()


    # thresholding
    img_gray = cv2.cvtColor(straight_license_plate_img, cv2.COLOR_RGB2GRAY)

    # improve the contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)

    # thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # opening
    # kernel = np.ones((3,3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # thresh = opening
    # ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)

    # plt.title("img_gray")
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()
    # plt.title("Thresholding")
    # plt.imshow(thresh, cmap='gray')
    # plt.show()

    if debug:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(thresh, cmap='gray')
        plt.title("Thresholded License Plate Image")
        plt.savefig(f"{prefix}_thresh_license_plate_image.png", bbox_inches='tight')
        if show:
            plt.show()

    img = thresh.copy()

    # connected components
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    digits_imgs = []
    digits_x = []
    for i in range(0, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        # TODO: change these area thresholds
        img_area = img.shape[0] * img.shape[1]
        if area > min_area * img_area and area < max_area * img_area:
            # the bbox should be a vertical rectangle
            if w <= h or abs(w-h) < 10:
                componentMask = (labels == i).astype("uint8") * 255
                digits_imgs.append(componentMask[y:y + h, x:x + w])
                digits_x.append(x)
    
    # sort the digits based on x
    indices = np.argsort(digits_x).tolist()

    sorted_digit_imgs = []
    for i in indices:
        sorted_digit_imgs.append(digits_imgs[i])

    resized_imgs = []
    for i in range(len(sorted_digit_imgs)):
        # resize to 26x26
        resized_img = cv2.resize(sorted_digit_imgs[i], (26, 26), interpolation=cv2.INTER_AREA)
        # add a 2 pixel padding
        resized_img = np.pad(resized_img, (2,2), 'constant', constant_values=0)
        resized_imgs.append(resized_img)

        # from 0-255 to 0-1
        resized_imgs[i] = resized_imgs[i].astype(float) / 255
    
    if debug:
        show_resized_images(resized_imgs, cropped_license_image)
    
    print(f"{len(resized_imgs)}/8 digits detected")
    
    return resized_imgs, cropped_license_image

