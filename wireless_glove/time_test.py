import bluetooth
import RPi.GPIO as GPIO
import numpy as np
from time import sleep
import time

ble_address = '98:D3:31:F9:6B:67'
port = 1

sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
try: 
    sock.connect((ble_address, port))
    print("Connection Successful!!!")
except:
    print("Connection Fail!!!")

GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

signal_channel = []

# start time
time_start = time.time()
# end time
time_end = 0

p_time_stamp = 0

num = 1

while True:
    time_end = time.time()
    data_ = sock.recv(1024)
    if len(data_) >=17:
        print(chr(data_[15]))
#     print(np.shape(data_))
#     signal_channel.append(data_[0])
#     print(int(time_end-time_start), ':', data_)
    sleep(0.1)