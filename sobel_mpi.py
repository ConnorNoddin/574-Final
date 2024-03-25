from mpi4py import MPI
import numpy as np
import sys
import copy
import hashlib
import math
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def convolve(image_matrix, image_filter, image_output, xstart, xend):
    height, width, channels = image_matrix.shape
    # Loop through each color of pixel
    for d in range(0, channels, 1):
        # Loop through height
        for y in range(1, height - 1, 1):
            # Loop through width
            for x in range(int(xstart), int(xend), 1):
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


def combine(image_matrix1, image_matrix2, image_output, xstart, xend):
    height, width, channels = image_matrix1.shape
    # Loop through each color of pixel
    for d in range(0, channels, 1):
        # Loop through height
        for y in range(1, height - 1, 1):
            # Loop through width
            for x in range(int(xstart), int(xend), 1):
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
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        data = file.read()
        # Returns MD5 hash as hex
        return hashlib.md5(data).hexdigest()


def main():
    # Allows one command line argument for the image input file
    if len(sys.argv) != 3:
        if rank == 0:
            print("Usage: python sobel_mpi.py <image_file> <output_file>")
        sys.exit(1)

    # Get the input filename from command-line argument
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # X and Y filter as used in the ECE 574 homeworks
    sobel_x_filter = [-1, 0, +1, -2, 0, +2, -1, 0, +1]
    sobel_y_filter = [-1, -2, -1, 0, 0, 0, 1, 2, +1]

    # Load image on rank 0
    if rank == 0:
        image_matrix = load_jpeg(input_file)
    else:
        image_matrix = None

    # Broadcast image_matrix to all processes
    image_matrix = comm.bcast(image_matrix, root=0)

    # Split image width among processes
    width = image_matrix.shape[1]
    chunk_size = width // size
    start_col = rank * chunk_size
    end_col = start_col + chunk_size

    # Adjust start_col and end_col to ensure they are within image bounds
    if rank == 0:
        start_col = 1
    if rank == size - 1:
        end_col = width - 1

    # Perform convolution on each process
    x_convolve = np.zeros_like(image_matrix)
    y_convolve = np.zeros_like(image_matrix)
    convolve(image_matrix, sobel_x_filter, x_convolve, start_col, end_col)
    convolve(image_matrix, sobel_y_filter, y_convolve, start_col, end_col)

    # Combine results on rank 0
    if rank == 0:
        combined = np.zeros_like(image_matrix)
    else:
        combined = None

    # Gather results to rank 0
    comm.Gather(y_convolve, combined, root=0)

    # Save JPEG on rank 0
    if rank == 0:
        store_jpeg(combined, output_file)
        # Print MD5 sum
        print(f"MD5SUM of {output_file}: {md5sum(output_file)}")


# Run main function
main()
