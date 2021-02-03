
import sys
from typing import List

from ptmlib.time import Stopwatch

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager.context import LogicalDevice

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def print_device_info():
    my_devices: List[LogicalDevice] = tf.config.list_logical_devices()
    for item in my_devices:
        print(f"NAME: {item.name}    TYPE: {item.device_type}")


def main():
    stopwatch = Stopwatch()
    stopwatch.start()

    print_hi('PyCharm')

    print('python version:', sys.version)
    print('tf version:', tf.__version__)
    print('keras version:', keras.__version__)

    print_device_info()

    stopwatch.stop()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
