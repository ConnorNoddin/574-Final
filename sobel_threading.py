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
import threading


def convolve(image_matrix, image_filter, image_output, ystart, yend, thread_num):
    height, width, channels = image_matrix.shape
    # Debugging for convolve
    print(
        f"Thread {thread_num} is doing convolve for the follow y values: {int(ystart)} to {int(yend)}"
    )
    # Loop through each color of pixel
    for d in range(0, channels, 1):
        # Loop through width
        for x in range(1, width-1, 1):
            # Loop through height
            for y in range(ystart, yend, 1): 
                # Get the RGB values of the pixel
                sum = image_matrix[y - 1, x - 1, d] * image_filter[0]
                sum += image_matrix[y - 1, x, d] * image_filter[1]
                sum += image_matrix[y - 1, x + 1, d] * image_filter[2]
                sum += image_matrix[y, x - 1, d] * image_filter[3]
                sum += image_matrix[y, x, d] * image_filter[4]
                sum += image_matrix[y, x + 1, d] * image_filter[5]
                sum += image_matrix[y + 1, x - 1, d] * image_filter[6]
                sum += image_matrix[y + 1, x, d] * image_filter[7]
                sum += image_matrix[y + 1, x + 1, d] * image_filter[8]
                # Check for saturation
                if sum > 255:
                    sum = 255
                elif sum < 0:
                    sum = 0
                # Save convolution to output array
                image_output[y, x, d] = sum


def combine(image_matrix1, image_matrix2, image_output, ystart, yend, thread_num):
    height, width, channels = image_matrix1.shape
    # Debugging for combine
    print(
        f"Thread {thread_num} is doing combine for the follow y values: {int(ystart)} to {int(yend)}"
    )
    # Loop through each color of pixel
    for d in range(0, channels, 1):
        # Loop through height
        for x in range(1, width - 1, 1):
            # Loop through width
            for y in range(ystart, yend, 1):
                result = image_matrix1[y, x, d] ** 2 + image_matrix2[y, x, d] ** 2
                # Save the results as int() to replicate the C behavior
                result = int(math.sqrt(result))
                # Chek for saturation
                if result > 255:
                    result = 255
                elif result < 0:
                    result = 0
                # Save result to putput matrix
                image_output[y, x, d] = result


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
    if len(sys.argv) != 3:
        print("Usage: python sobel_serial.py <image_file> <num_threads>")
        sys.exit(1)

    # Get number of threads as second argument
    num_threads = int(sys.argv[2])

    # X and Y filter as used in the ECE 574 homeworks
    sobel_x_filter = [-1, 0, +1, -2, 0, +2, -1, 0, +1]
    sobel_y_filter = [-1, -2, -1, 0, 0, 0, 1, 2, +1]

    # List to hold threads
    threads = []
    # Start and end values fo each thread
    ystarts = [0] * num_threads
    yends = [0] * num_threads

    # Name for the output image
    output_name = "out.jpg"

    start_time = time.time()

    # Get the input filename from command-line argument
    image_matrix = load_jpeg(sys.argv[1])

    load_time = time.time()

    # Makes an empty copy for output
    x_convolve = np.zeros_like(image_matrix)
    y_convolve = np.zeros_like(image_matrix)

    # Width to calculate work for each thread
    height, _, _ = image_matrix.shape

    # Get start and end x values for each thread
    chunk_size = height // num_threads  # // floors the division
    ystarts = [i * chunk_size for i in range(num_threads)]
    yends = [start + chunk_size for start in ystarts]
    ystarts[0] = 1  # First index is always 1
    yends[-1] = height- 1  # Last index does not do border

    # Create and start threads to perform the x convolution
    for i in range(num_threads):
        thread = threading.Thread(
            target=convolve,
            args=(image_matrix, sobel_x_filter, x_convolve, ystarts[i], yends[i], i),
        )
        threads.append(thread)
        thread.start()

    # Join each thread
    for thread in threads:
        thread.join()

    # Clear threads
    threads = []

    # Create and start threads to perform the y convolution
    for i in range(num_threads):
        thread = threading.Thread(
            target=convolve,
            args=(image_matrix, sobel_y_filter, y_convolve, ystarts[i], yends[i], i),
        )
        threads.append(thread)
        thread.start()

    # Join each y convolve threads
    for thread in threads:
        thread.join()

    # Clear threads
    threads = []

    convolve_time = time.time()

    # Output matrix for combine
    combined = np.zeros_like(image_matrix)

    # Create and start threads to combine the images
    for i in range(num_threads):
        thread = threading.Thread(
            target=combine,
            args=(x_convolve, y_convolve, combined, ystarts[i], yends[i], i),
        )
        threads.append(thread)
        thread.start()

    # Join each thread
    for thread in threads:
        thread.join()

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
