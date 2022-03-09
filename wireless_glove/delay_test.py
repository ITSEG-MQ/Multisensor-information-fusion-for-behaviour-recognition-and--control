import bluetooth
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

signal_channel = [[] for i in range(14)]

ble_address = '98:D3:31:F9:6B:67'
port = 1

sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
try: 
    sock.connect((ble_address, port))
    print("Connection Successful!!!")
except:
    print("Connection Fail!!!")

signal_channel = [[] for i in range(14)]

fig, axs = plt.subplots(8)

while True:
    ble_data = sock.recv(1024)
    print(len(ble_data))
    if len(ble_data) >= 16:
        if chr(ble_data[0])=='s' and chr(ble_data[15]=='e'):            
            print("Length:", len(ble_data), " Value:", ble_data[1])
        sleep(0.1)
    

