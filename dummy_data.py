import time
import math
import random
import threading
import json
import numpy as np


# Sensor enable/disable configuration
sensor_config = {
   "pMMG": True,
   "IMU": True,
   "EMG": True,
   "Controller": True
}


# Number of sensors for each type
num_pmmg = 8
num_imu = 6
num_emg = 8
num_buttons = 4


# Return initial sensor configuration (sent once at startup)
def get_sensor_info_packet():
   info = {}
   if sensor_config["pMMG"]:
       info["pMMG"] = num_pmmg
   if sensor_config["IMU"]:
       info["IMU"] = num_imu
   if sensor_config["EMG"]:
       info["EMG"] = num_emg
   if sensor_config["Controller"]:
       info["Controller"] = {"buttons": num_buttons, "joystick": "2D"}
   return {"info": info}


# Quaternion helper: Hamilton product
def quaternion_multiply(q1, q2):
   w1, x1, y1, z1 = q1
   w2, x2, y2, z2 = q2
   return [
       w1*w2 - x1*x2 - y1*y2 - z1*z2,
       w1*x2 + x1*w2 + y1*z2 - z1*y2,
       w1*y2 - x1*z2 + y1*w2 + z1*x2,
       w1*z2 + x1*y2 - y1*x2 + z1*w2
   ]


# Generate 3D rotation quaternion
def generate_quaternion(t):
   roll  = 0.1 * np.sin(0.2 * t)
   pitch = 0.1 * np.sin(0.17 * t + 1.0)
   yaw   = 0.1 * np.sin(0.13 * t + 2.0)


   qx = np.array([np.cos(roll/2), np.sin(roll/2), 0, 0])
   qy = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
   qz = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])


   q = quaternion_multiply(qz, quaternion_multiply(qy, qx))
   return [round(x, 5) for x in q]


# pMMG: Sine wave
def generate_pmmg(t):
   return [round(110 + (10 + i) * math.sin(t + i * 0.3), 2) for i in range(num_pmmg)]


# EMG: realistic microvolt range
def generate_emg():
   emg = []
   for _ in range(num_emg):
       if random.random() < 0.2:
           val = random.uniform(-1e-4, 1e-4)
       elif random.random() < 0.3:
           val = random.uniform(-1e-6, 1e-6)
       else:
           val = random.gauss(0, 2e-5)
       emg.append(round(val, 7))
   return emg


# Game controller: joystick and 4 buttons
class ControllerState:
   def __init__(self):
       self.last_press_time = [0.0] * num_buttons
       self.next_press_delay = [random.uniform(1.0, 2.0) for _ in range(num_buttons)]
       self.joystick_phase = random.uniform(0, 2 * math.pi)
       self.motion_start_time = None


   def generate(self, t):
       # Buttons: random button presses at sparse intervals
       buttons = []
       
       for i in range(num_buttons):
           if t - self.last_press_time[i] > self.next_press_delay[i]:
               buttons.append(True)
               self.last_press_time[i] = t
               self.next_press_delay[i] = random.uniform(1.0, 2.0)
           else:
               buttons.append(False)


       # Joystick: occasionally initiate movement
       if self.motion_start_time is None and random.random() < 0.01:
           self.motion_start_time = t
           self.joystick_phase = random.uniform(0, 2 * math.pi)


       if self.motion_start_time is not None:
           dt = t - self.motion_start_time
           if dt > 2.0:
               self.motion_start_time = None
               joy_x, joy_y = 128, 128
           else:
               joy_x = int(128 + 60 * math.sin(0.5 * dt + self.joystick_phase))
               joy_y = int(128 + 60 * math.sin(0.4 * dt + self.joystick_phase + 1))
       else:
           joy_x, joy_y = 128, 128


       return {
           "buttons": buttons,
           "joystick": {"x": joy_x, "y": joy_y}
       }


controller_state = ControllerState()


# Generate sensor packet
def generate_sensor_packet(t):
   packet = {"timestamp": round(t, 3)}
   if sensor_config["pMMG"]:
       packet["pMMG"] = generate_pmmg(t)
   if sensor_config["IMU"]:
       packet["IMU"] = [generate_quaternion(t + i) for i in range(num_imu)]
   if sensor_config["EMG"]:
       packet["EMG"] = generate_emg()
   if sensor_config["Controller"]:
       packet["Controller"] = controller_state.generate(t)
   return packet


# Stream sensor data until Enter pressed
def stream_dummy_data(stop_flag):
   print("\nStarted streaming dummy data... (press Enter to stop this trial)\n")
   t_start = time.time()
   while not stop_flag.is_set():
       t = time.time() - t_start
       data = generate_sensor_packet(t)
       print(json.dumps(data))
       time.sleep(0.04)


if __name__ == "__main__":
   # Print initial sensor info
   sensor_info = get_sensor_info_packet()
   print(">>> Sensor Configuration Packet:")
   print(json.dumps(sensor_info))


   print("\nPress Enter to start a trial. Type 'q' or 'quit' to exit.\n")


   while True:
       user_input = input(">> Start next trial? (Enter = yes / q = quit): ").strip().lower()
       if user_input in ['q', 'quit']:
           print("Exiting dummy data generator.")
           break


       # Create stop flag and thread
       stop_flag = threading.Event()
       thread = threading.Thread(target=stream_dummy_data, args=(stop_flag,))
       thread.start()


       # Wait for Enter to stop
       input()
       stop_flag.set()
       thread.join()
       print("\nTrial ended.\n") 