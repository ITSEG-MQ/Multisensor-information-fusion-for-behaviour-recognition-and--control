import bluetooth
import numpy as np
from time import sleep
import time
import torch
import numpy as np
from data_loader import data_generator_np

ble_address = '98:D3:31:F9:6B:67'
port = 1

sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
try: 
    sock.connect((ble_address, port))
    print("Connection Successful!!!")
except:
    print("Connection Fail!!!")

signal_channel = [[] for i in range(14)]

while True:
    device = torch.device("cpu")
#     get ble data
    ble_data = sock.recv(1024)
#     if data length bigger than 10
    if len(ble_data) >= 16:
#         make sure the ble data's start and end met the check symble
#         make sure the data is correct
        if chr(ble_data[0])=='s' and chr(ble_data[15]=='e'):
            for i in range(14):
                signal_channel[i].append(ble_data[i+1])
                
            
            if np.shape(signal_channel)[-1] >= 251:
                input_data = np.array(signal_channel)
                data = torch.FloatTensor(input_data[:, -251:-1])
                data = data.unsqueeze(0)
                data = data.unsqueeze(0)
        
                model = torch.load('net_3.pt', map_location="cpu")
                model = model.to(device)
                model.eval()
                with torch.no_grad():
                    result = model(data)
                    print(result)
                    result = result.detach().numpy()[0]
                    result = np.argmax(result)
    
                print(result)
    

    sleep(0.005)

