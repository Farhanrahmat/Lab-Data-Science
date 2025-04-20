import cv2
import numpy as np
import matplotlib.pyplot as plt

print("This is running")


def show_overlay(img1, img2, alpha=0.5):
    """Overlay two grayscale images."""
    if img1.shape != img2.shape:
        raise ValueError("Images must be the same shape for overlay.")
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    plt.imshow(blended, cmap='gray')
    plt.title("Overlay (Check Pattern Alignment)")
    plt.axis("off")
    plt.show()
    

def get_central_circle_radius(image, weight_radius=1.0):
    """
    Detects the most central circle with a slight penalty for radius size.
    
    Parameters:
        image (np.ndarray): Grayscale or BGR image
        weight_radius (float): Penalize larger radii when selecting circle
    
    Returns:
        (x, y, r): Circle center and radius
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=20, maxRadius=400
    )

    if circles is None:
        raise ValueError("No circles detected.")

    circles = np.uint16(np.around(circles[0]))
    h, w = gray.shape
    center_img = np.array([w / 2, h / 2])

    # Score based on proximity to center and slight radius penalty
    def score(c):
        dist = np.linalg.norm(center_img - np.array([c[0], c[1]]))
        return dist + weight_radius * c[2]

    best = min(circles, key=score)
    print(f"[circle] Center: ({best[0]}, {best[1]}), Radius: {best[2]}")
    return best[0], best[1], best[2]  # x, y, r



def scale_to_reference_circle(img_to_scale, reference_img):
    """
    Scales `img_to_scale` so that its central circle matches that of `reference_img`,
    and then crops it to the same size as `reference_img`, centered on the resized circle.

    Returns:
        final_crop (np.ndarray): Scaled and center-cropped version of img_to_scale.
    """
    # --- Step 1: Get circle info ---
    x_ref, y_ref, r_ref = get_central_circle_radius(reference_img)
    x_target, y_target, r_target = get_central_circle_radius(img_to_scale)

    # --- Step 2: Resize to match circle size ---
    scale_factor = r_ref / r_target
    new_size = (int(img_to_scale.shape[1] * scale_factor), int(img_to_scale.shape[0] * scale_factor))
    scaled_img = cv2.resize(img_to_scale, new_size, interpolation=cv2.INTER_CUBIC)

    # --- Step 3: Find new resized circle center ---
    x_scaled = int(x_target * scale_factor)
    y_scaled = int(y_target * scale_factor)

    # --- Step 4: Crop around the new circle center, matching ref size ---
    h_ref, w_ref = reference_img.shape[:2]
    x_start = max(0, x_scaled - w_ref // 2)
    y_start = max(0, y_scaled - h_ref // 2)
    x_end = x_start + w_ref
    y_end = y_start + h_ref

    # Ensure it stays within image bounds
    x_end = min(x_end, scaled_img.shape[1])
    y_end = min(y_end, scaled_img.shape[0])
    x_start = max(0, x_end - w_ref)
    y_start = max(0, y_end - h_ref)

    final_crop = scaled_img[y_start:y_end, x_start:x_end]
    return final_crop


#def background_remove(target_img, reference_path="C:/Users/farha/OneDrive/Documents/Images"):
def background_remove(img, ref_path):
    # Funtion removes background having a reference image where the 
    #  background was removed manually, then identifies the circle in 
    #  the middle ofcthe piece and masks it

    # --- Load images ---
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    # --- Validate loading ---
    if ref_img is None:
        raise FileNotFoundError(f"Could not load reference: {reference_path}")

    # --- Step 1: Template Matching ---
    result = cv2.matchTemplate(img, ref_img, cv2.TM_CCOEFF_NORMED)
    _, _, _, top_left = cv2.minMaxLoc(result)

    h, w = ref_img.shape
    bottom_right = (top_left[0] + w, top_left[0] + h)
    matched_crop = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # --- Step 2: Detect and Mask a Single Central Circle ---
    circle_masked_crop = matched_crop.copy()
    circle_mask = np.zeros_like(matched_crop)

    circles = cv2.HoughCircles(
        matched_crop, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=100, maxRadius=400
        )

    if circles is not None:
        circles = np.uint16(np.around(circles))[0]

        # Choose the most central circle
        (h, w) = matched_crop.shape
        center_img = np.array([w / 2, h / 2])
        distances = [np.linalg.norm(np.array([x, y]) - center_img) for (x, y, r) in circles]
        best_idx = int(np.argmin(distances))

        # Use just that one
        x, y, r = circles[best_idx]
        cv2.circle(circle_mask, (x, y), r, 255, -1)  # Draw on mask
        cv2.circle(circle_masked_crop, (x, y), r, 0, -1)  # Black-out the circle

    # --- Step 3: Display Results ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(matched_crop, cmap='gray')
    plt.title("Matched Crop")

    plt.subplot(1, 3, 2)
    plt.imshow(circle_mask, cmap='gray')
    plt.title("Detected Circle Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(circle_masked_crop, cmap='gray')
    plt.title("Circle Masked Out")

    plt.tight_layout()
    plt.show()

    return circle_masked_crop, circle_mask



def extract_part(image, return_hole_mask=True):
    """
    Extracts the main 3D printed part from an image and optionally identifies internal holes.
    
    Parameters:
        image (np.ndarray): Grayscale or color image containing the 3D printed part.
        return_hole_mask (bool): Whether to return a mask of the detected hole(s).
    
    Returns:
        final (np.ndarray): Image with background removed.
        part_mask (np.ndarray): Binary mask of the main part.
        hole_mask (np.ndarray): Binary mask of the hole(s), if enabled.
    """
    
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold to binary
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours with hierarchy (to get holes)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    part_mask = np.zeros_like(gray)
    hole_mask = np.zeros_like(gray)

    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # If it's an outer contour (no parent)
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(part_mask, contours, i, 255, -1)
            # If it's a child contour (a hole inside the part)
            elif return_hole_mask:
                cv2.drawContours(hole_mask, contours, i, 255, -1)

    # Subtract hole from part if needed
    combined_mask = cv2.subtract(part_mask, hole_mask) if return_hole_mask else part_mask

    # Apply mask to original image
    final = cv2.bitwise_and(gray, gray, mask=combined_mask)
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(final, cmap='gray')
    plt.title("Matched Crop")

    plt.subplot(1, 3, 2)
    plt.imshow(part_mask, cmap='gray')
    plt.title("Detected Circle Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(hole_mask, cmap='gray')
    plt.title("Circle Masked Out")

    if return_hole_mask:
        return final, part_mask, hole_mask
    else:
        return final, part_mask
