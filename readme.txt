# tensorflow runtime for Raspberry pi zero (Bullseye)

This is my first trial of building tensorflow lite wheel for Raspberry pi zero "1" with Bullseye and Python 3.9.

tflite_runtime-2.11.0-cp39-cp39-linux_armv6l.whl


# INSTALLATION


pi@raspberrypi7:~ $ pip install tflite_runtime-2.11.0-cp39-cp39-linux_armv6l.whl --force-reinstall

Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
Processing ./tflite_runtime-2.11.0-cp39-cp39-linux_armv6l.whl
Requirement already satisfied: numpy>=1.19.2 in /usr/lib/python3/dist-packages (from tflite-runtime==2.11.0) (1.19.5)
Installing collected packages: tflite-runtime
Successfully installed tflite-runtime-3.11.0

Note) 3.11.0 is typo. Should read 2.11.0

# TEST

pi@raspberrypi7:~ $ export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0
pi@raspberrypi7:~ $ python3 
Python 3.9.2 (default, Mar 12 2021, 04:06:34)
[GCC 10.2.1 20210110] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tflite_runtime.interpreter as tflite
>>>

Note: If you get an error "undefined symbol: __atomic_compare_exchange_8", 
you may run as follows:
LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3
If you set environment variable in advance as 
export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0
you can run such as 
python3 tflite_mnist.py


pi@raspberrypi7:~/mnist $ python3 tflite_mnist.py
2.40 sec for preparation
9 nine
0.39 sec for inference



