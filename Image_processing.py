import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any,Optional,Tuple
import numpy as np
import cv2
import pywt
from typing import Any, Dict
import os
import cv2
import numpy as np
from PIL import Image
from skimage.restoration import wiener
from skimage.restoration import richardson_lucy
from skimage.util import img_as_float
from skimage import exposure
from skimage.color import rgb2gray
from skimage.restoration import estimate_sigma
from skimage.restoration import unsupervised_wiener
import cv2
import numpy as np
import imutils
from paddleocr import PaddleOCR
import cv2

def get_image_metadata(image_input) -> Dict[str, Any]:
    """
    Extracts comprehensive metadata from an image file or numpy array.
    Args:
        image_input: Either a path (str) to the input image or a numpy array containing image data.
    Returns:
        Dictionary containing all metadata fields.
    """
    
    metadata = {}
    
    # Case 1: Input is a file path (string)
    if isinstance(image_input, str):
        # Get file size
        file_size_bytes = os.path.getsize(image_input)
        file_size_kb = file_size_bytes / 1024
        metadata['file_size_kb'] = file_size_kb
        metadata['image_type'] = os.path.splitext(image_input)[1].lower()

        # Open image with PIL for basic attributes
        with Image.open(image_input) as img:
            resolution = img.size
            mode = img.mode
            img_format = img.format
            color_profile = img.info.get('icc_profile', 'None')
            aspect_ratio = resolution[0] / resolution[1]
            has_alpha = 'A' in img.getbands()
            dpi = img.info.get('dpi', (72, 72))  # Default to (72, 72) if not set
            metadata['dpi'] = dpi

        # Open image with OpenCV for further analysis
        image_cv = cv2.imread(image_input)

    # Case 2: Input is a numpy array
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

        resolution = img.size
        mode = img.mode
        img_format = None  # Format not known for numpy input
        color_profile = 'None'
        aspect_ratio = resolution[0] / resolution[1]
        has_alpha = 'A' in img.getbands()
        dpi = (72, 72)  # Default DPI for numpy arrays
        metadata['dpi'] = dpi

        image_cv = image_input

    else:
        raise ValueError("Input must be either a file path (str) or numpy array")

    # Bit depth logic
    bit_depth = None
    if mode == '1':
        bit_depth = 1
    elif mode in ('L', 'P'):
        bit_depth = 8
    elif mode == 'RGB':
        bit_depth = 24
    elif mode in ('RGBA', 'CMYK', 'I', 'F'):
        bit_depth = 32

    # Get number of channels
    num_channels = image_cv.shape[2] if len(image_cv.shape) == 3 else 1

    # Compile metadata
    metadata.update({
        'resolution': resolution,
        'width': resolution[0],
        'height': resolution[1],
        'mode': mode,
        'format': img_format,
        'color_profile': color_profile,
        'aspect_ratio': aspect_ratio,
        'has_alpha': has_alpha,
        'bit_depth': bit_depth,
        'num_channels': num_channels,
    })

    return metadata

def binarize(image: np.ndarray) -> np.ndarray:
    """Converts image to binary using thresholding."""
    
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def resize(image: np.ndarray, width: int = None, height: int = None, keep_aspect: bool = True,interpolation: str = 'bilinear') -> np.ndarray:
    """Resizes image with optional aspect ratio preservation."""
    h, w = image.shape[:2]
    if keep_aspect and width and height:
        scale = min(width / w, height / h)
        new_w, new_h = int(w * scale), int(h * scale)
    elif width:
        new_w, new_h = width, int(h * (width / w))
    elif height:
        new_w, new_h = int(w * (height / h)), height
    else:
        return image
    
     # Map interpolation methods to OpenCV flags
    interp_methods = {
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    interp_flag = interp_methods.get(interpolation.lower(), cv2.INTER_LINEAR)

    return cv2.resize(image, (new_w, new_h), interpolation=interp_flag)


def grayscalee(image: np.ndarray,threshold_value: Optional[int] = None) -> np.ndarray:
    """Converts image to grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if threshold_value is not None:
        _,  gray = cv2.threshold(gray, threshold_value,255, cv2.THRESH_BINARY)
    return gray   
def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes a BGR image to the 0-255 range for each channel."""
    # Ensure the image is in BGR format
    normalized_image = cv2.normalize(
    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print("Image normalized successfully")

    return normalized_image

def remove_noisee(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies Gaussian blur for noise removal."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

#Noise removal
def anscombe_transform(img):
    return 2.0 * np.sqrt(img + 3/8.0)

def inverse_anscombe_transform(trans):
    return ((trans / 2.0) ** 2) - 3/8.0

def detect_gaussian_or_speckle_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    high_freq_energy = np.sum(magnitude_spectrum > np.mean(magnitude_spectrum) + 3*np.std(magnitude_spectrum))
    return high_freq_energy

def detect_salt_pepper_histogram(img):
    hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
    total_pixels = img.shape[0] * img.shape[1]
    edge_count = int(hist[0] + hist[-1])
    salt_pepper_ratio = edge_count / total_pixels
    return salt_pepper_ratio > 0.01  # adjust threshold if needed

def detect_poisson_wavelet(img):
    trans = anscombe_transform(img)
    coeffs = pywt.wavedec2(trans, 'db1', level=2)
    detail_energy = sum(np.sum(np.abs(detail)) for detail in coeffs[1])
    return detail_energy > 1000  # adjust threshold

def gaussian_denoise(img):
    return cv2.GaussianBlur(img, (3, 3), 0)

def salt_pepper_denoise(img):
    return cv2.medianBlur(img.astype(np.uint8), 5)

def speckle_denoise(img):
    return cv2.medianBlur(img.astype(np.uint8), 3)

def poisson_denoise(img):
    trans = anscombe_transform(img)
    coeffs = pywt.wavedec2(trans, 'db1', level=2)
    threshold = 10
    coeffs_thresh = [coeffs[0]] + [(pywt.threshold(c, threshold, 'soft') for c in detail) for detail in coeffs[1:]]
    denoised_trans = pywt.waverec2(coeffs_thresh, 'db1')
    return inverse_anscombe_transform(denoised_trans)

def detect_and_denoise(img):
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    print("\n[DEBUG] Starting noise detection...")

    # --- Noise Detection ---
    high_freq_energy = detect_gaussian_or_speckle_fft(img)
    gaussian_threshold = 15000
    speckle_threshold = 10000
    salt_pepper_detected = detect_salt_pepper_histogram(img)
    poisson_detected = detect_poisson_wavelet(img)

    print(f"[DEBUG] Noise Metrics:")
    print(f"  - High-Freq Energy (FFT): {high_freq_energy}")
    print(f"  - Salt-and-Pepper Ratio: {'High' if salt_pepper_detected else 'Low'}")
    print(f"  - Poisson Noise (Wavelet): {'Detected' if poisson_detected else 'Not Detected'}")

    # --- Noise Classification ---
    if high_freq_energy > gaussian_threshold:
        noise_type = "Gaussian"
        print(f"[DEBUG] Noise Type: {noise_type}")
        denoised = gaussian_denoise((img * 255).astype(np.uint8))
    elif high_freq_energy > speckle_threshold:
        noise_type = "Speckle"
        print(f"[DEBUG] Noise Type: {noise_type}")
        denoised = speckle_denoise((img * 255).astype(np.uint8))
    elif salt_pepper_detected:
        noise_type = "Salt-and-Pepper"
        print(f"[DEBUG] Noise Type: {noise_type}")
        denoised = salt_pepper_denoise((img * 255).astype(np.uint8))
    elif poisson_detected:
        noise_type = "Poisson"
        print(f"[DEBUG] Noise Type: {noise_type}")
        denoised = poisson_denoise(img)
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
    else:
        noise_type = "None/Unclassified"
        print(f"[DEBUG] Noise Type: {noise_type}")
        denoised = (img * 255).astype(np.uint8)

    print(f"[DEBUG] Denoising complete. Noise Type: {noise_type}\n")
    return denoised


def blind_deblur(image, kernel_size=15):
    """Applies blind deconvolution assuming unknown kernel."""
    img_float = img_as_float(image)
    if image.ndim == 3:
        deblurred = np.zeros_like(img_float)
        for c in range(3):
            # Initialize dummy PSF (e.g., flat or Gaussian)
            init_psf = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            deblur_c, est_psf = unsupervised_wiener(img_float[:, :, c], init_psf)
            deblurred[:, :, c] = deblur_c
    else:
        init_psf = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        deblurred, est_psf = unsupervised_wiener(img_float, init_psf)
    return np.clip(deblurred * 255, 0, 255).astype(np.uint8)



def create_motion_psf(length=15, angle=0):
    """Creates a motion blur PSF kernel."""
    kernel = np.zeros((length, length))
    kernel[length // 2, :] = 1  # Horizontal line
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    return kernel / np.sum(kernel)

def apply_hybrid_fallback(image):
    """Fallback deblurring when blur type is unknown."""
    img_float = img_as_float(image)
    if image.ndim == 3:  # Color
        deblurred = np.zeros_like(img_float)
        for c in range(3):
            deblurred[:, :, c] = wiener(img_float[:, :, c], mysize=(5, 5))
    else:  # Grayscale
        deblurred = wiener(img_float, mysize=(5, 5))
    return np.clip(deblurred * 255, 0, 255).astype(np.uint8)


def analyze_blur_type(gray_image):
    """
    Identifies blur type by analyzing frequency domain, edge characteristics,
    and Laplacian variance.
    
    Returns: "motion", "gaussian", "uniform", or "unknown"
    """
    # Ensure image is grayscale
    if gray_image.ndim != 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    # 1. Check Laplacian variance for overall blur
    lap_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # 2. Check for motion blur using frequency domain
    fft = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)  # Avoid log(0)

    # Threshold for strong frequency components (lines in motion blur)
    thresh = 0.3 * np.max(magnitude_spectrum)
    mask = (magnitude_spectrum > thresh).astype(np.uint8)

    # Try to detect straight lines in frequency spectrum
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None and len(lines) > 2:
        return "motion"

    # 3. Edge width analysis for Gaussian blur
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edge_widths = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        edge_widths.append(w)
        edge_widths.append(h)
    avg_edge_width = np.mean(edge_widths) if edge_widths else 0

    # 4. Noise level for uniform blur clue
    noise_sigma = estimate_sigma(gray_image, average_sigmas=True)

    # Decision logic
    if avg_edge_width > 3.5:
        return "gaussian"
    elif noise_sigma < 0.5 and avg_edge_width > 2:
        return "uniform"
    elif lap_var < 10:
        return "motion"  # Fallback motion if too blurry
    else:
        return "unknown"

def hybrid_deblur(image, blur_threshold=2000):
    """
    Smart deblurring based on detected blur type.
    Applies Gaussian, motion, or uniform PSF deconvolution,
    and blind deconvolution when the blur type is unknown.
    """
    def is_blurry(img, threshold):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Laplacian Variance: {lap_var} (Blurry? {lap_var < threshold})")
        return lap_var < threshold

    if not is_blurry(image, blur_threshold):
        print("Image not blurry — returning original.")
        return image

    # Convert to grayscale for blur type analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur_type = analyze_blur_type(gray)
    print(f"Detected Blur Type: {blur_type}")

    img_float = img_as_float(image)

    if blur_type == "gaussian":
        
        psf = np.ones((5, 5)) / 25
        if image.ndim == 3:
            deblurred = np.stack([wiener(img_float[:, :, c], psf, balance=0.01) for c in range(3)], axis=2)
        else:
            deblurred = wiener(img_float, psf, balance=0.01)

    elif blur_type == "motion":
        
        psf = create_motion_psf(length=15, angle=45)  # You may estimate angle in future
        if image.ndim == 3:
            deblurred = np.stack([richardson_lucy(img_float[:, :, c], psf, num_iter=10) for c in range(3)], axis=2)
        else:
            deblurred = richardson_lucy(img_float, psf, num_iter=10)

    elif blur_type == "uniform":
        
        psf = np.ones((5, 5)) / 25
        if image.ndim == 3:
            deblurred = np.stack([wiener(img_float[:, :, c], psf, balance=0.01) for c in range(3)], axis=2)
        else:
            deblurred = blind_deblur(image, kernel_size=15)
            deblurred = wiener(img_float, psf, balance=0.01)

    else:  # Blind deconvolution fallback
        print("Applying blind deconvolution for unknown blur.")
        deblurred = blind_deblur(image, kernel_size=15)
        return deblurred  # Already converted to uint8 inside

    # Final clipping and conversion
    deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
    return deblurred

def enhance_contrast_with_clahe(image: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    """
    Detects low contrast using is_low_contrast and applies CLAHE if needed.
    
    Parameters:
        image (np.ndarray): Input RGB or grayscale image.
        threshold (float): Threshold for low contrast detection (default is 0.35).
    
    Returns:
        np.ndarray: Contrast-enhanced image (same shape as input).
    """
    # Convert image to grayscale float for contrast detection
    gray_float = rgb2gray(img_as_float(image))

    if exposure.is_low_contrast(gray_float, fraction_threshold=threshold):
        # Apply CLAHE using OpenCV (works on 8-bit images)
        img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if len(img_uint8.shape) == 2:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_uint8)
        else:  # Color image (apply CLAHE to each channel in LAB)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            print("[DEBUG] Contrast enhancement complete\n")
        return enhanced
    else:
        return image  # Return original if contrast is sufficient

#Brightness correction
# Add to Image_processing.py
def detect_brightness(image: np.ndarray) -> float:
    """
    Detects the overall brightness of an image using histogram analysis.
    Returns a brightness score between 0 (dark) and 1 (bright).
    """
    if len(image.shape) == 3:
        # Convert to grayscale if color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute histogram (256 bins for pixel values 0-255)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel()

    # Normalize histogram to get probability distribution
    hist_normalized = hist / hist.sum()

    # Calculate brightness as the weighted average of intensity values
    brightness_score = np.sum(hist_normalized * np.arange(256)) / 255.0

    return brightness_score


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Applies gamma correction to an image.
    gamma > 1: darker
    gamma < 1: brighter
    gamma = 1: no change
    """
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def auto_gamma_correction(image: np.ndarray) -> np.ndarray:
    """
    Automatically adjusts gamma based on image brightness.
    Dark images get brighter (gamma < 1)
    Bright images get darker (gamma > 1)
    """
    brightness = detect_brightness(image)
    
    # Determine gamma value based on brightness
    if brightness < 0.4:  # Dark image
        gamma = 0.7  # Brighten
    elif brightness > 0.6:  # Bright image
        gamma = 1.3  # Darken
    else:  # Normal brightness
        gamma = 1.0  # No change
    
    # Apply gamma correction
    corrected_image = apply_gamma_correction(image, gamma)
    
    print(f"[DEBUG] Brightness: {brightness:.2f}, Applied Gamma: {gamma:.2f}")
    return corrected_image

#Cropping

def hard_crop(image: np.ndarray, point1: Tuple[int, int], point2: Tuple[int, int]) -> np.ndarray:
    x1, y1 = point1
    x2, y2 = point2
    x_start, x_end = sorted([x1, x2])
    y_start, y_end = sorted([y1, y2])
    return image[y_start:y_end, x_start:x_end]

#CROPPING

def crop_with_paddleocr(image: np.ndarray, padding=20, det_only=True):
    ocr = PaddleOCR(use_angle_cls=False, lang='en', det_limit_side_len=1280)

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input — expected NumPy array.")

    result = ocr.ocr(image, det=True, rec=False, cls=False)

    if not result or not result[0]:
        return image  # return original image if no detection

    boxes = []
    for line in result[0]:
        x_coords = [point[0] for point in line]
        y_coords = [point[1] for point in line]
        boxes.append((min(x_coords), min(y_coords), max(x_coords), max(y_coords)))

    min_x = max(0, int(min(box[0] for box in boxes) - padding))
    min_y = max(0, int(min(box[1] for box in boxes) - padding))
    max_x = min(image.shape[1], int(max(box[2] for box in boxes) + padding))
    max_y = min(image.shape[0], int(max(box[3] for box in boxes) + padding))

    cropped = image[min_y:max_y, min_x:max_x]
    return cropped





def detect_and_straighten_by_moments(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return image

    # Find the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Get orientation using image moments
    moments = cv2.moments(largest)
    if abs(moments["mu02"]) < 1e-2:
        print("No meaningful rotation detected.")
        return image

    angle = 0.5 * np.arctan2(2 * moments["mu11"], (moments["mu20"] - moments["mu02"]))
    angle_degrees = np.degrees(angle)
    print(f"Detected angle (moments): {angle_degrees:.2f}°")
    if abs(angle_degrees) < 2.0:
        print("Detected angle is too small to correct.")
        return image

    return imutils.rotate_bound(image, -angle_degrees)

