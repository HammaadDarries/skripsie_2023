#! /usr/bin/env python 

import rclpy

import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist

import torch

from sac_agent import SACAgent, PolicyNetworkSAC

MAX_SPEED = 4
MAX_STEER = 0.4

class SACAgentNode (Node):
    def __init__(self):
        super().__init__('sac_agent_node')

        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        nn_path = '/home/hammaad/sim_ws/src/sac_agent_pkg/Data/myFavouriteAgent_SAC_aut'
        self.scan_buffer = np.zeros((2, 20))

        self.model = PolicyNetworkSAC()
        self.model.load_state_dict(torch.load(nn_path))

        self.agent = SACAgent(self.model)

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

    def lidar_callback(self, msg):
        ranges = msg.ranges
        ranges = np.array(ranges)

        # Normalize ranges
        scaled_ranges = ranges/30
        ranges = np.clip(scaled_ranges, 0, 1)

        # use only 180 degree FOV instead of 270
        ranges_180 = ranges[180:-180]

        # average every 36 scans to get 20 beams instead of 720
        mean_ranges = np.mean(ranges_180.reshape(-1, 36), axis=1)

        scan = mean_ranges

        if self.scan_buffer.all() ==0: # first reading
            for i in range(2):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        lidar_state = np.reshape(self.scan_buffer, (20 * 2))

        self.lidar_state = lidar_state


def main(args = None):
    rclpy.init(args = args)
    sac_agent_node = SACAgentNode()
    rclpy.spin(sac_agent_node)
    rclpy.shutdown()



