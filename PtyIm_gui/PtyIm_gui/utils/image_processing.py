import numpy as np
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift, shift


def rescale_image(image, pixel_size, reference_pixel_size):
    scale = pixel_size / reference_pixel_size
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return resize(image, new_size, preserve_range=True, anti_aliasing=True).astype(
        image.dtype
    )


def crop_center_stack(stack, crop_size):
    h, w = stack.shape[1:]
    sx = (w - crop_size) // 2
    sy = (h - crop_size) // 2
    return stack[:, sy : sy + crop_size, sx : sx + crop_size]


def register_images(stack):
    # Normalize stack just for shift detection
    normalized_stack = [normalize_image(img) for img in stack]
    ref = normalized_stack[0]

    registered = [stack[0]]  # First image is unchanged (reference)

    for img, norm_img in zip(stack[1:], normalized_stack[1:]):
        shift_pixels, _, _ = phase_cross_correlation(ref, norm_img)

        # Shift original (non-normalized) image without changing value scale
        shifted = shift(img, shift=shift_pixels, mode="constant", cval=0)

        # Ensure dtype and range match the original
        registered.append(shifted.astype(stack[0].dtype))
    my_min_size_image = crop_black_borders(registered, threshold=1e-3)
    return np.stack(my_min_size_image)


def load_text_image(filename):
    return np.loadtxt(filename, delimiter=" ")


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        # Return a blank (mid-gray) image if the input is flat
        return np.full_like(image, 127, dtype=np.uint8)
    normalized = (image - min_val) / (max_val - min_val) * 255
    return normalized.astype(np.uint8)


def crop_black_borders(stack, threshold=1e-3):
    """
    Crop the registered image stack to remove black borders.

    Parameters:
        stack (list or np.ndarray): Image stack of shape (N, H, W)
        threshold (float): Minimum pixel intensity considered non-black

    Returns:
        np.ndarray: Cropped image stack (N, h, w)
    """
    stack = np.array(stack)  # Convert list to NumPy array

    masks = stack > threshold  # Binary masks for non-black pixels
    combined_mask = np.all(masks, axis=0)

    if not np.any(combined_mask):
        raise ValueError("No overlapping non-black region found across the stack.")

    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    cropped_stack = stack[:, row_min : row_max + 1, col_min : col_max + 1]
    return cropped_stack
