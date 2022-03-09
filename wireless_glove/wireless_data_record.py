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

signal_channel = [[] for i in range(14)]

# button current state and previous state
button_state = 0
p_button_state = 0
start_record = False

# current file number

# previous file number
p_num = 0
num = 0

# start time
time_start = 0
# end time
time_end = 0
# set recording length to 60 seconds
recording_len = 60

p_time_stamp = 0



while True:
    with open("file_num.txt", 'r') as f:
        num = int(f.readline())
#     get current button state
    button_state = GPIO.input(7)
#     get current record state and file number
    if(button_state != p_button_state):
        p_button_state = button_state
        if(button_state == 1):
            if(start_record == True):
                start_record = False
            else:
                start_record = True
                num+=1
                with open("file_num.txt", 'w+') as f:
                    f.write(str(num))
                    
                

#     get ble data
    ble_data = sock.recv(1024)
#     if data length bigger than 10
    if len(ble_data) >= 16:
#         make sure the ble data's start and end met the check symble
#         make sure the data is correct
        if chr(ble_data[0])=='s' and chr(ble_data[15]=='e'):
            
#             set data_file name
            if(num != p_num):
                p_num = num
                data_file = str(num) + ".txt"               
                print("Current file:", data_file)
                time_start = time.time()
            
#             start record, save data to signal_channel
            if start_record is True: 
                for i in range(14):
                    signal_channel[i].append(ble_data[i+1])
                    time_end = time.time()
                    time_stamp = int(time_end-time_start)

                    # record 1 min, stop recording
                if time_end-time_start >= recording_len:
                    start_record = False
                        
#           if stop record save signal_channel data to txt files
            if start_record is False:
                if(len(signal_channel[0]) > 0 and len(signal_channel[1]) > 0): 
                    np.savetxt(str(data_file), signal_channel, delimiter=',') 
                    print("Current File Shape:", np.shape(signal_channel)) 
                    signal_channel = [[] for i in range(14)] 
                    print("File:", data_file, "saved!!!") 
                    p_time_stamp = 0

    sleep(0.005)
