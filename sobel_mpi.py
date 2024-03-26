from mpi4py import MPI
import numpy as np
import sys
import copy
import hashlib
import math
import time
from PIL import Image


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def convolve(image_matrix, image_filter, image_output, ystart, yend):
    height, width, channels = image_matrix.shape
    print(f"Node is {rank}... doing convolve from {ystart} to {yend}")
    # Loop through each color of pixel
    for d in range(0, channels, 1):
        # Loop through height
        for x in range(1, width - 1, 1):
            # Loop through width
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


def combine(image_matrix1, image_matrix2, image_output, ystart, yend):
    height, width, channels = image_matrix1.shape
    print(f"Node is {rank}... doing combine from {ystart} to {yend}")
    # Loop through each color of pixel
    for d in range(0, channels, 1):
        # Loop through width
        for x in range(1, width - 1, 1):
            # Loop through height
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
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: python sobel_mpi.py <image_file>")
        sys.exit(1)

    # Get the input filename from command-line argument
    input_file = sys.argv[1]
    output_file = "out.jpg"

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
    height = image_matrix.shape[0]
    chunk_size = height // size
    start_y = rank * chunk_size
    end_y = start_y + chunk_size

    # First rank does not do edge
    if rank == 0:
        start_y = 1
    # Last rank is does not do edge
    if rank == size - 1:
        end_y = height - 1

    # Perform convolution on each process
    x_convolve = np.zeros_like(image_matrix)
    y_convolve = np.zeros_like(image_matrix)
    combined_result = np.zeros_like(image_matrix)
    convolve(image_matrix, sobel_x_filter, x_convolve, start_y, end_y)
    convolve(image_matrix, sobel_y_filter, y_convolve, start_y, end_y)
    combine(x_convolve, y_convolve, combined_result, start_y, end_y)

    # Flatten the 3D NumPy array to a 1D list to help with MPI gather
    combined_1d = combined_result.flatten().tolist()

    final_result_1d = comm.gather(combined_1d, root=0)

    # Save JPEG on rank 0
    if rank == 0:
        recovery_3d = np.array(final_result_1d)
        numpy_array_tmp = np.zeros_like(image_matrix)
        numpy_array_back = np.zeros_like(image_matrix)
        starts = []
        ends = []
        for i in range(size):
            starts.append(i * chunk_size)
            ends.append(starts[i] + chunk_size)
        starts[0] = 1
        ends[-1] = height - 1

        for i in range(size):
            numpy_array_tmp = recovery_3d[i, :].reshape(image_matrix.shape)
            numpy_array_back[starts[i] : ends[i], :] = numpy_array_tmp[
                starts[i] : ends[i], :
            ]
        # Store final 3d result
        store_jpeg(numpy_array_back, output_file)
        # Print MD5 sum
        print(f"MD5SUM of {output_file}: {md5sum(output_file)}")


# Always run main function MPI way
main()
