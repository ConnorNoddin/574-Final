# Connor Noddin
# ECE 574
# Final Project
# sobel_serial

from PIL import Image
import numpy as np
import sys
import copy
import hashlib
import math
import time
from scipy import signal


def convolve(image_matrix, image_filter):
    height, width, channels = image_matrix.shape
    grad = np.zeros_like(image_matrix).astype(np.double)
    # Do one convolution for each color
    for d in range(0, channels, 1):
        grad[1 : height - 1, 1 : width - 1, d] = signal.convolve2d(
            image_matrix[:, :, d].astype(np.double),
            image_filter.astype(np.double),
            boundary="fill",
            mode="valid",
            fillvalue=0,
        )
        # Ensures values are in valid range
        grad[:, :, d] = np.clip(grad[:, :, d], 0, 255).astype(np.double)

    return grad


def combine(image_matrix1, image_matrix2):
    # Square and sum the pixel values of each image
    sum = image_matrix1.astype(np.double) ** 2 + image_matrix2.astype(np.double) ** 2

    # Sqrt of each value in entire array
    output_matrix = np.sqrt(sum)

    # Ensures values are in the correct range
    output_matrix = np.clip(output_matrix, 0, 255).astype(np.double)

    return output_matrix


def load_jpeg(file_path):
    try:
        # Open the JPEG file using PIL
        img = Image.open(file_path)
        # Convert the image to a 3D numpy array
        img_array = np.array(img)
        return img_array
    except Exception as error:
        print(f"Error loading image {file_path}:", error)
        sys.exit(1)


def store_jpeg(image_matrix, filename):
    # Save image using 8 bit jpegs as in 574 homeworks
    new_img = Image.fromarray(image_matrix.astype("uint8"))

    try:
        # Save image using quality 90 as in 574 homeworks
        new_img.save(filename, quality=90)
    # Error handling if save failed
    except Exception as error:
        print(f"Error saving image {filename}:", error)


def md5sum(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
        # Returns MD5 hash as hex
        return hashlib.md5(data).hexdigest()


def main():
    # Allows one command line argument for the image input file
    if len(sys.argv) != 2:
        print("Usage: python sobel_serial.py <image_file>")
        sys.exit(1)

    # X and Y filter as used in the ECE 574 homeworks
    # sobel_x_filter = np.array(
    #     [
    #         [-1, 0, +1],
    #         [-2, 0, +2],
    #         [-1, 0, +1],
    #     ],
    # )
    # sobel_y_filter = np.array(
    #     [
    #         [-1, -2, -1],
    #         [0, 0, 0],
    #         [1, 2, +1],
    #     ]
    # )

    # Mirror the kernel
    sobel_x_filter = np.array(
        [
            [+1, 0, -1],
            [2, 0, -2],
            [+1, 0, -1],
        ]
    )

    sobel_y_filter = np.array(
        [
            [+1, 2, +1],
            [0, 0, 0],
            [-1, -2, -1],
        ]
    )

    # Name for the output image
    output_name = "out.jpg"

    start_time = time.time()

    # Get the input filename from command-line argument
    image_matrix = load_jpeg(sys.argv[1])

    load_time = time.time()

    # Convolve image and using X filter
    x_convolve = convolve(image_matrix, sobel_x_filter)

    # X convolve for debugging
    store_jpeg(x_convolve, f"{output_name.replace('.jpg', '')}_x.jpg")

    # Convolve image and using Y filter
    y_convolve = convolve(image_matrix, sobel_y_filter)

    # Y convolve for debugging
    store_jpeg(y_convolve, f"{output_name.replace('.jpg', '')}_y.jpg")

    convolve_time = time.time()

    # Combine both images
    combined = combine(x_convolve, y_convolve)

    combine_time = time.time()

    # Store JPEG: Note, quality is 90
    store_jpeg(combined, output_name)

    store_time = time.time()

    # MD5Sums to ensure convolve performed correctly
    # print(f"MD5SUM of butterfinger_sobel.jpg: {md5sum("butterfinger_sobel.jpg")}")
    print(f"MD5SUM of {output_name}: {md5sum(output_name)}")

    # Timing calculations for each major component
    print(f"Load Time: {load_time-start_time}")
    print(f"Convolve Time: {convolve_time-load_time}")
    print(f"Combine Time: {combine_time-convolve_time}")
    print(f"Store Time: {store_time-combine_time}")


# Ensures main() is only run when the program is run as a script
if __name__ == "__main__":
    main()
