#include <SoftwareSerial.h>
#include <Adafruit_MCP3008.h>
#include <Wire.h>
#include <I2Cdev.h>
#include <MPU6050.h>

SoftwareSerial ble_module(6, 7); // RX, TX
Adafruit_MCP3008 adc; 
MPU6050 mpu;



// variables to store mpu value
int16_t ax, ay, az;
int16_t gx, gy, gz;

// variables to store mpu value 0-255
int valax; 
int valay;
int valaz;
int valgx; 
int valgy;
int valgz;

int p_valax; 
int p_valay;
int p_valaz;
int p_valgx; 
int p_valgy;
int p_valgz;

// set green led pin number
#define green_led 2

// variable to store receiver command
char r_command = 'a';

// array to store sending data 
byte sensors_data[14];

// a variable to decide send sensors data
boolean if_send = true; 


unsigned long previousTime = 0; 
//const long breakTime = 100;

void setup() {
//  set led pin mode
  pinMode(green_led, OUTPUT);
  
//  set led to low
  digitalWrite(green_led, LOW);
  
//  set bluetooth baud rate
  ble_module.begin(115200);
  
//  set select pin to D3 for MCP3008
  adc.begin(10); 

// MPU6050 setup
  IMU_setup(); 

}

void loop() {

  send_data();
  delay(5);
  
  mpu_data(); 
  p_valax = valax;
  p_valay = valay;
  p_valaz = valaz;
  p_valgx = valgx;
  p_valgy = valgy;
  p_valgz = valgz;
  if(p_valax == valax && p_valay == valay && p_valaz == valaz) { 
    digitalWrite(green_led, LOW);
    IMU_setup(); 
  }
}

// function to send data
void send_data() {
  mpu_data(); 
// data start with 's'
  ble_module.print('s');
  for(byte i=0; i<8; i++){
    sensors_data[i] = map(adc.readADC(i), 0, 1023, 0, 255);
  }
  sensors_data[8] = valax;
  sensors_data[9] = valay;
  sensors_data[10] = valaz;
  sensors_data[11] = valgx;
  sensors_data[12] = valgy;
  sensors_data[13] = valgz;
//  sensors_data[14] = map(analogRead(emg_sensor), 0, 1023, 0, 255);
  ble_module.write(sensors_data, 14);
//  data end with 'e'
  ble_module.print('e');
}

void mpu_data(){
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  valax = map(ax, -17000, 17000, 0, 255);
  valay = map(ay, -17000, 17000, 0, 255);
  valaz = map(az, -17000, 17000, 0, 255);
  valgx = map(gx, -17000, 17000, 0, 255);
  valgy = map(gy, -17000, 17000, 0, 255);
  valgz = map(gz, -17000, 17000, 0, 255); 
}

void IMU_setup() {
  Wire.begin();
  Serial.begin(38400);
  mpu.initialize();
  if(mpu.testConnection()){
    digitalWrite(green_led, HIGH);
    Serial.print("yes");
  }
  else {
    digitalWrite(green_led, LOW);
  }
}
