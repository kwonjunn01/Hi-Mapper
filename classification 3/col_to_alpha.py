'''
We are going to use Python's Pillow library to implement GIMP's color to alpha algorithm.
https://docs.gimp.org/en/gimp-filter-color-to-alpha.html

Maverick Reynolds
07.15.2023

'''

import numpy as np

# Different scalar interpolation functions
def interpolate(x, interpolation=None):
    if interpolation == 'power':
        return x**2
    elif interpolation == 'root':
        return np.sqrt(x)
    elif interpolation == 'smooth':
        return (np.sin(np.pi/2*x))**2
    elif interpolation == 'inverse-sin':
        return np.arcsin(2*x-1)/np.pi + 0.5
    else:
        return x


def rgb_distance(pixels: np.array, color: np.array, shape='cube'):
    '''
    If shape is 'cube', the radius is the maximum orthogonal distance between the two colors
    If shape is 'sphere', the radius is the distance between the two colors in 3D space
    
    Returned value should always be between 0 and 255
    '''

    # Ensure parameters are RGB (three channels)
    pixels = pixels[:,:,:3]

    # Take advantage of numpy's vectorization here
    if shape == 'cube':
        return np.amax(abs(pixels - color), axis=2)
    elif shape == 'sphere':
        return np.linalg.norm(pixels - color, axis=2)


def color_to_alpha(pixels, color, transparency_threshold, opacity_threshold, shape='cube', interpolation=None):
    '''
    this function takes in the image and performs the GIMP color to alpha algorithm
    Colors within the transparency_threshold are converted to transparent
    Colors within the opacity_threshold are unchanged
    Colors between the two thresholds smoothly transition between transparent and opaque

    Takes advantage of np vectorization
    '''
    color = np.array(color)

    # Make new pixels and th channel for RGBA
    pixels = pixels[:,:,:3]
    new_pixels = np.copy(pixels)
    new_pixels = np.append(new_pixels, np.zeros((new_pixels.shape[0], new_pixels.shape[1], 1), dtype=np.uint8), axis=2)

    # Get the distance matrix
    distances = rgb_distance(pixels, color, shape=shape)

    # Create masks for pixels that are transparent and opaque
    transparency_mask = distances <= transparency_threshold
    opacity_mask = distances >= opacity_threshold

    # Calculate alpha values for pixels between the thresholds
    threshold_difference = opacity_threshold - transparency_threshold
    alpha = (distances - transparency_threshold) / threshold_difference
    alpha = np.clip(alpha, 0, 1)

    # Interpolate based on method provided
    alpha = interpolate(alpha, interpolation=interpolation)

    # Extrapolate along line passing through color and pixel onto the opacity threshold
    # This is the RGB value that will be used for the pixel
    proportion_to_opacity = distances / opacity_threshold
    extrapolated_colors = (pixels - color) / proportion_to_opacity[:, :, np.newaxis] + color
    
    extrapolated_colors = np.nan_to_num(extrapolated_colors, nan=0)
    extrapolated_colors = np.clip(np.around(extrapolated_colors), 0, 255).astype(np.uint8)

    # Reassign color values of intermediate pixels
    new_pixels[~transparency_mask & ~opacity_mask, :3] = extrapolated_colors[~transparency_mask & ~opacity_mask]
    # Reassign the alpha values of intermediate pixels
    new_pixels[:, :, 3] = alpha * 255
    
    return new_pixels