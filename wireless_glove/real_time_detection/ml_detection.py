import pickle
import numpy as np
import bluetooth

def feature_extraction(x):
    data_feature = []
    data_ave = np.mean(x, 1)
    data_std = np.std(x, 1)
    data_max = np.max(x, 1)
    data_min = np.min(x, 1)
    data_energy = np.sum(np.abs(x), 1)
    data_max_min = data_max - data_min
    list_feature = [data_ave, data_std, data_max, data_min, data_energy, data_max_min]
    data_feature.append(list_feature)

    data_feature = np.array(data_feature)
    return data_feature


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

    
                print(result)
    

    sleep(0.005)


