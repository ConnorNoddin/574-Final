from PIL import Image
import numpy as np
import sys
import copy
import hashlib
import math
import time

def convolve(image_matrix, image_filter):
    height, width, channels = image_matrix.shape
    image_matrix_copy = copy.copy(image_matrix)
    image_matrix_copy = np.zeros_like(image_matrix_copy)
    for d in range(0, channels, 1):
        for y in range(1, height - 1, 1):
            for x in range(1, width - 1, 1):
                # Get the RGB values of the pixel
                sum = 0
                sum += image_matrix[y - 1, x - 1, d] * image_filter[0]
                sum += image_matrix[y - 1, x, d] * image_filter[1]
                sum += image_matrix[y - 1, x + 1, d] * image_filter[2]
                sum += image_matrix[y, x - 1, d] * image_filter[3]
                sum += image_matrix[y, x, d] * image_filter[4]
                sum += image_matrix[y, x + 1, d] * image_filter[5]
                sum += image_matrix[y + 1, x - 1, d] * image_filter[6]
                sum += image_matrix[y + 1, x, d] * image_filter[7]
                sum += image_matrix[y + 1, x + 1, d] * image_filter[8]
                if sum > 255:
                    sum = 255
                if sum < 0:
                    sum = 0
                image_matrix_copy[y, x, d] = sum
                # Do something with the pixel values, for example, print them
    return image_matrix_copy

def combine(image_matrix1, image_matrix2):
    height, width, channels = image_matrix1.shape
    image_matrix_copy = copy.copy(image_matrix1)
    image_matrix_copy = np.zeros_like(image_matrix_copy)
    for d in range(0, channels, 1):
        for y in range(1, height - 1, 1):
            for x in range(1, width - 1, 1):
                # Get the RGB values of the pixel
                result = 0
                result = image_matrix1[y, x, d]**2 + image_matrix2[y, x, d]**2
                result = int(math.sqrt(result))
                if result > 255:
                    result = 255
                if result < 0:
                    result = 0
                image_matrix_copy[y, x, d] = result
                # Do something with the pixel values, for example, print them
    return image_matrix_copy

def load_jpeg(file_path):
    # Load the JPEG file as a 2D matrix
    try:
        # Open the JPEG file
        img = Image.open(file_path)

        # Convert the image to a numpy array
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

def store_jpeg(image_matrix, filename):
    # Create a PIL image from the modified matrix
    new_img = Image.fromarray(image_matrix.astype("uint8"))

    # Save the modified image as a new JPEG file
    try:
        new_img.save(filename, quality=90)
        print("New JPEG file saved successfully.")
    except Exception as e:
        print("Error saving image:", e)

def md5sum(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_file>")
        sys.exit(1)

    sobel_x_filter = [-1, 0, +1, -2, 0, +2, -1, 0, +1]
    sobel_y_filter = [-1, -2, -1, 0, 0, 0, 1, 2, +1]

    output_name = "out.jpg"

    start_time = time.time()

    # Get the filename from command-line argument
    image_matrix = load_jpeg(sys.argv[1])

    load_time = time.time()

    x_convolve = convolve(image_matrix, sobel_x_filter)

    # X convolve for debugging
    #store_jpeg(x_convolve, f"{output_name.replace(".jpg", "")}_x.jpg")

    y_convolve = convolve(image_matrix, sobel_y_filter)

    # Y convolve for debugging
    #store_jpeg(y_convolve, f"{output_name.replace(".jpg", "")}_y.jpg")

    convolve_time = time.time()

    combined = combine(x_convolve, y_convolve)

    combine_time = time.time()

    store_jpeg(combined, output_name)

    store_time = time.time()

    print(f"MD5SUM of butterfinger_sobel.jpg: {md5sum("butterfinger_sobel.jpg")}")
    print(f"MD5SUM of {output_name}: {md5sum(output_name)}")

    print(f"Load Time: {load_time-start_time}")
    print(f"Load Time: {convolve_time-load_time}")
    print(f"Load Time: {combine_time-convolve_time}")
    print(f"Load Time: {store_time-combine_time}")

# Ensures main() is only run when the program is run as a script
if __name__ == "__main__":
    main()
