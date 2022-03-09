import bluetooth
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

sensors_value = []


ble_address = '98:D3:31:F9:6B:67'
port = 1

sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
try: 
    sock.connect((ble_address, port))
    print("Connection Successful!!!")
except:
    print("Connection Fail!!!")

signal_channel = [[] for i in range(8)]

fig, axs = plt.subplots(8)

while True:
    ble_data = sock.recv(1024)
    
    if len(ble_data) >= 10:
        if chr(ble_data[0])=='s' and chr(ble_data[9]=='e'):            

            
            for i in range(8):
                signal_channel[i].append(ble_data[i+1])
            
# start real-time plot
            plt.ion()
# pressure sensors plot
            for i in range(8):
                axs[i].plot(signal_channel[i])    
        plt.pause(0.001)        
        sleep(0.001)
    
plt.off()
