#! /usr/bin/env python 

import rclpy
import csv
import numpy as np
import time
import os

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist

import torch

from sac_agent_pkg.sac_agent import SACAgent

MAX_SPEED = 2
MAX_STEER = 0.4
NOISE_FACTOR = 0.002

class SACAgentNode (Node):
    def __init__(self):
        super().__init__('sac_agent_node')

        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.start_time = None
        nn_path = '/home/hammaad/sim_ws/src/sac_agent_pkg/Data/Main_Agent/gamma_0.99.pth'
        self.scan_buffer = np.zeros((2, 20))
  
        self.lap_complete = False
        self.collisions = 0
        self.prev_collision_state = False
        self.start_position = None

        self.agent = SACAgent(nn_path)
        self.position_history = []
        self.speed_history = []

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = self.add_noise(ranges, NOISE_FACTOR)
        
        # Normalize ranges
        scaled_ranges = ranges/30
        # get 20 beams out of the 1080 by averaging every 54 beams
        ranges = np.clip(scaled_ranges, 0, 1)
        mean_ranges = np.mean(ranges.reshape(-1, 54), axis=1)
        # self.min_lidar_reading = np.min(mean_ranges)

        self.check_collision(mean_ranges)

        ###########
        #   END   #
        ###########
        
        scan = mean_ranges

        if self.scan_buffer.all() ==0: # first reading
            for i in range(2):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        lidar_state = np.reshape(self.scan_buffer, (20 * 2))

        self.lidar_state = lidar_state

    def timer_callback(self):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'Attempt_1_Lesgooo'

        nn_state = self.lidar_state
        nn_action = self.agent.act(nn_state)

        steering_angle = nn_action[0] * MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED)

        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed

        self.drive_pub.publish(msg)

    # # NEW #
    # def timer_callback(self):
    #     msg = AckermannDriveStamped()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.frame_id = 'Attempt_1_Lesgooo'

    #     nn_state = self.lidar_state
    #     nn_action = self.agent.act(nn_state)

    #     steering_angle = nn_action[0] * MAX_STEER
    #     speed = (nn_action[1] + 1) * (MAX_SPEED / 2 - 0.5) + 1
    #     speed = min(speed, MAX_SPEED)

    #     # Adjust speed and steering angle based on min_lidar_reading
    #     speed, steering_angle = self.adjust_speed_and_steering(speed, steering_angle)

    #     msg.drive.steering_angle = steering_angle
    #     msg.drive.speed = speed

    #     self.drive_pub.publish(msg)

    # # NEW #
    # def adjust_speed_and_steering(self, speed, steering_angle):
    #     # Adjust speed and steering angle based on min_lidar_reading
    #     # Decrease speed and increase steering sensitivity as the min_lidar_reading gets smaller
    #     min_reading_threshold = 0.05  # Adjust this value based on your requirement

    #     if self.min_lidar_reading < min_reading_threshold:
    #         speed_factor = self.min_lidar_reading / min_reading_threshold
    #         speed *= speed_factor

    #         steering_sensitivity = 2.5  # Adjust this value based on how sensitive you want the steering to be
    #         steering_angle *= steering_sensitivity

    #     return speed, steering_angle

    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        self.twist = msg.twist.twist

        self.position_history.append((self.pose.position.x, self.pose.position.y))
        self.speed_history.append(self.twist.linear.x)

        current_position = (self.pose.position.x, self.pose.position.y)
        if self.start_time is None:
            self.start_time = self.get_clock().now()

        if self.check_lap_complete(current_position):
            time_since_start = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            self.get_logger().info(f'Lap completed in {time_since_start} seconds.')
            self.start_time = self.get_clock().now()

        self.position_history.append(current_position)
        self.speed_history.append(self.twist.linear.x)


        

    # NEW #
    def add_noise(self, ranges, noise_factor):
        noise = np.random.normal(0, noise_factor, ranges.shape)
        noisy_ranges = ranges + noise
        return noisy_ranges

    def save_data_to_file(self):
        with open('position_history.csv', 'w', newline='') as position_file, open('speed_history.csv', 'w', newline='') as speed_file:
            position_writer = csv.writer(position_file)
            speed_writer = csv.writer(speed_file)
        
            position_writer.writerow(['x', 'y'])
            speed_writer.writerow(['speed'])

            for pos, speed in zip(self.position_history, self.speed_history):
                position_writer.writerow(pos)
                speed_writer.writerow([speed])

    def save_data_to_file(self):
        folder_name = f"PLOTS/speed_{MAX_SPEED}_{NOISE_FACTOR}_data"
        os.makedirs(folder_name, exist_ok=True)

        position_file_path = os.path.join(folder_name, 'position_history.csv')
        speed_file_path = os.path.join(folder_name, 'speed_history.csv')

        with open(position_file_path, 'w', newline='') as position_file, open(speed_file_path, 'w', newline='') as speed_file:
            position_writer = csv.writer(position_file)
            speed_writer = csv.writer(speed_file)

            position_writer.writerow(['x', 'y'])
            speed_writer.writerow(['speed'])

            for pos, speed in zip(self.position_history, self.speed_history):
                position_writer.writerow(pos)
                speed_writer.writerow([speed])

    def check_lap_complete(self, current_position):
        if self.start_position is None:
            self.start_position = current_position
            return False

        distance_to_start = np.linalg.norm(np.array(self.start_position) - np.array(current_position))
        if not self.lap_complete and distance_to_start < 1.0:
            self.lap_complete = True
        elif self.lap_complete and distance_to_start > 2.0:
            self.lap_complete = False
            return True

        return False

    def check_collision(self, current_ranges):
        collision = min(current_ranges) < 0.1
        if collision and not self.prev_collision_state:
            self.collisions += 1
        self.prev_collision_state = collision


# def main(args = None):
#     rclpy.init(args = args)
#     sac_agent_node = SACAgentNode()
#     rclpy.spin(sac_agent_node)
#     rclpy.shutdown()

# NEW #
def main(args = None):
    rclpy.init(args = args)
    sac_agent_node = SACAgentNode()
    
    try:
        rclpy.spin(sac_agent_node)
    except KeyboardInterrupt:
        pass
    finally:
        # sac_agent_node.save_data_to_file()
        sac_agent_node.destroy_node()
        rclpy.shutdown()



