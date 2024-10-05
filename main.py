import threading
import warnings
from face_detect import faceDetection
from voice_match import trainVoice, listen_and_match
from multiprocessing import Process
import tkinter as tk

warnings.filterwarnings("ignore", category=UserWarning, message=".*GetPrototype.*")

# Initialize processes as None
listen_thread = None
face_thread = None

trainVoice()

def startProcess():
    global listen_thread, face_thread
    try:
        # Initialize the processes
        listen_thread = Process(target=listen_and_match)
        face_thread = Process(target=faceDetection)

        # Start the processes
        listen_thread.start()
        face_thread.start()

    except Exception as e:
        print(f"Exception occurred: {e}")
        terminate()  # Make sure to terminate in case of an error

def terminate():
    global listen_thread, face_thread
    if listen_thread and listen_thread.is_alive():
        listen_thread.terminate()
        listen_thread.join()  # Ensure proper cleanup
    if face_thread and face_thread.is_alive():
        face_thread.terminate()
        face_thread.join()  # Ensure proper cleanup
    print("Processes terminated!")

if __name__ == "__main__":

    # Create the main window
    window = tk.Tk()
    window.title("Simple Tkinter Example")

    # Create a label
    label = tk.Label(window, text="Hello, Tkinter!")
    label.pack(pady=20)

    # Define a function to start the processes
    def on_start_click():
        startProcess()
        label.config(text="Processes Started!")

    # Define a function to terminate the processes
    def on_terminate_click():
        terminate()
        label.config(text="Processes Terminated!")

    # Create a button to start the processes
    start_button = tk.Button(window, text="Start", command=on_start_click)
    start_button.pack(pady=10)

    # Create a button to terminate the processes
    terminate_button = tk.Button(window, text="Terminate", command=on_terminate_click)
    terminate_button.pack(pady=10)

    # Run the application (Tkinter's event loop remains responsive)
    window.mainloop()
