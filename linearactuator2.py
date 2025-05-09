'''
-----------------------------------------------------
-----------------------------------------------------
Lunabotics Linear Actuator Testing
Author: NotAWildernessExplorer
Date:04/11/2025 
-----------------------------------------------------
-----------------------------------------------------
Connections for BST7960 to Pi4 b
Pin 1:	GPIO12 (pin 32 on board)
Pin 2:	GPIO13 (pin 33 on board)
Pin 3:	3V
Pin 4:	3V
Pin 5:	NC
Pin 6: 	NC
Pin 7:	3V
Pin 8:	GND
-----------------------------------------------------
-----------------------------------------------------
'''

## import library 
import time                 # What time is it? Well, this library will tell you!!! 
from gpiozero import PWMOutputDevice

## Luna linear actuators start here!
class LinearActuator:
    def __init__(self):
        # In gpiozero, we use BCM pin numbers (GPIO12 and GPIO13) instead of board pins (32 and 33)
        self.r_pwm = PWMOutputDevice(12, frequency=125000)  # GPIO12 for forward
        self.l_pwm = PWMOutputDevice(13, frequency=125000)  # GPIO13 for reverse
        # Note: Maximum PWM frequency in gpiozero might be lower than 125MHz
        # Actual supported frequency depends on hardware
        
    def move(self, qty):
        '''
        Changes motor controller duty cycle\n
        qty > 0: extend \n
        qty < 0: retract \n
        qty = 0: stop
        '''
        if qty > 0:
            self.stop()                  # Stop motors
            time.sleep(0.001)            # wait
            self.r_pwm.value = 1.0       # Full speed forward (1.0 = 100%)
        elif qty < 0:
            self.stop()                  # Stop motors
            time.sleep(0.001)            # wait
            self.l_pwm.value = 1.0       # Full speed reverse (1.0 = 100%)
        else:
            self.stop()                  # Stop motors
            time.sleep(0.001)            # wait
        
    def stop(self):
        '''stops the motors'''
        self.r_pwm.value = 0             # Set forward pwm to zero
        self.l_pwm.value = 0             # Set reverse pwm to zero
    
    def close(self):
        '''Clean up resources'''
        self.stop()
        self.r_pwm.close()
        self.l_pwm.close()

# Main program
try:
    # init the actuator
    LA = LinearActuator()
    
    # move forward
    print("Start")
    LA.move(+1)
    time.sleep(3)
    
    # move in reverse
    print("Rev")
    LA.move(-1)
    print("Done")
    time.sleep(3)
    
finally:
    # Clean up properly when done or if an exception occurs
    if 'LA' in locals():
        LA.close()
