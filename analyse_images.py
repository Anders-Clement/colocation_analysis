import os
import copy
from numba import jit
import numpy as np
import cv2
from PIL import Image

import roifile

# Jit gives a huge speedup of nested for loops
@jit
def count_image(img):

    # Thresholds over which pixels are considered red, green, or red-green
    RED_THRESHOLD = 40
    GREEN_THRESHOLD = 40

    # Counters for red, green, and red-green pixels
    reds = 0
    greens = 0
    red_greens = 0

    red_green_image = np.zeros_like(img)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # Current pixel, ordered as RGB
            pixel = img[y, x]
            # Completely black pixels can be skipped.
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                continue
            
            if pixel[0] > RED_THRESHOLD:
                reds += 1
            if pixel[1] > GREEN_THRESHOLD:
                greens += 1
            if pixel[0] > RED_THRESHOLD and pixel[1] > GREEN_THRESHOLD:
                red_greens += 1
                red_green_image[y, x] = pixel
                # Set blue channel to 0, for visualization purposes
                red_green_image[y, x, 2] = 0
            else:
                red_green_image[y, x] = (0, 0, 0)
    return reds, greens, red_greens, red_green_image

# Load tif image and convert to RGB, assumes TIF image with 3 channels ordered as RGB
def get_image(img_path):
    pil_image = Image.open(img_path)
    red = np.array(pil_image)
    pil_image.seek(1)
    green = np.array(pil_image)
    pil_image.seek(2)
    blue = np.array(pil_image)

    rgb_image = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
    rgb_image[:, :, 0] = red
    rgb_image[:, :, 1] = green
    rgb_image[:, :, 2] = blue

    return rgb_image

# Load ROI file as made from ImageJ, and use it to mask the image
def mask_from_roi(img, roi_path):
    
    roi = roifile.ImagejRoi.fromfile(roi_path)
    roi_coords = roi.coordinates(multi=True)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_img = copy.deepcopy(img)
    for coords in roi_coords:
        np_coords = np.array(coords, dtype=np.int32)
        cv2.drawContours(mask, [np_coords], -1, (255,255,255), -1, cv2.LINE_4)
        cv2.drawContours(roi_img, [np_coords], -1, (255,255,255), 10, cv2.LINE_4)

    masked = copy.deepcopy(img)
    masked = cv2.bitwise_and(masked, masked, mask=mask)

    return roi_img, mask, masked

# Get a full RGB channel, but only with a certain channel
def get_one_channel_as_rgb(img, channel):
    rgb_image = np.zeros_like(img)
    rgb_image[:, :, channel] = img[:, :, channel]
    return rgb_image

def main():
    # Assumes all images are in the images folder relative to the script.
    for root, dirs, files in os.walk('images'):
        for file in files:
            if file.endswith('.tif'):
                img_path = os.path.join(root, file)
                # Split by naming scheme: "Ax NR y ... .tif"
                name_tokens = file.split(' ')
                first = name_tokens[0]
                second = name_tokens[2]
                # ROI naming scheme: "Ax.y+....roi"
                roi_name = first + '.' + second
                found_roi = False
                for roi_file in os.listdir(root):
                    if roi_file.endswith('.roi') and roi_name in roi_file and '+' in roi_file:
                        roi_path = os.path.join(root, roi_file)
                        
                        rgb_image = get_image(img_path)

                        roi_img, mask, masked = mask_from_roi(rgb_image, roi_path)

                        pixel_count = np.count_nonzero(mask)

                        reds, greens, red_greens, red_green_image = count_image(masked)

                        red_image =  get_one_channel_as_rgb(rgb_image, 0)
                        green_image = get_one_channel_as_rgb(rgb_image, 1)
                        blue_image = get_one_channel_as_rgb(rgb_image, 2)

                        red_image_masked = get_one_channel_as_rgb(masked, 0)
                        green_image_masked = get_one_channel_as_rgb(masked, 1)
                        blue_image_masked = get_one_channel_as_rgb(masked, 2)

                        result_folder_name = file[:-4] # File name without extension
                        result_folder_path = os.path.join(root, result_folder_name)
                        if not os.path.exists(result_folder_path):
                            os.mkdir(result_folder_path)

                        # Save r,g,b and masked r,g,b, red_green_image, masked and roi_img
                        # Color conversion due to OpenCV using BGR instead of RGB
                        cv2.imwrite(os.path.join(root, result_folder_name, 'roi_img.png'), cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'mask.png'), cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name, 'masked.png'), cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'red_green_image.png'), cv2.cvtColor(red_green_image, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'red_image.png'), cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'green_image.png'), cv2.cvtColor(green_image, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'blue_image.png'), cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'red_image_masked.png'), cv2.cvtColor(red_image_masked, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'green_image_masked.png'), cv2.cvtColor(green_image_masked, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(os.path.join(root, result_folder_name,'blue_image_masked.png'), cv2.cvtColor(blue_image_masked, cv2.COLOR_BGR2RGB))
                        result_string = f'Reds: {reds}, Greens: {greens}, Red-Greens: {red_greens}, Pixel count in roi: {pixel_count}'
                        print(file + result_string)
                        # Also output result to file
                        result_file_path = os.path.join(root, result_folder_name,'result.txt')
                        if os.path.exists(result_file_path) and os.path.isfile(result_file_path):
                            os.remove(result_file_path)
                        with open(result_file_path, 'w') as result_file:
                            result_file.write(result_string)
                    
                        found_roi = True

                if not found_roi:
                    print(f'No ROI found for file: {os.path.join(root, file)}')

if __name__ == '__main__':
    main()