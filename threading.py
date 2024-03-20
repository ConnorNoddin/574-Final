import threading
import time

# Define a function that will be executed by each thread
def thread_function(name, delay):
    print("Thread", name, "is starting")
    counter = 0
    while counter < 5:
        time.sleep(delay)
        print("Thread", name, "count:", counter)
        counter += 1
    print("Thread", name, "is exiting")

# Create two threads
thread1 = threading.Thread(target=thread_function, args=(1, 1))  # Thread with delay of 1 second
thread2 = threading.Thread(target=thread_function, args=(2, 2))  # Thread with delay of 2 seconds

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both threads have finished")