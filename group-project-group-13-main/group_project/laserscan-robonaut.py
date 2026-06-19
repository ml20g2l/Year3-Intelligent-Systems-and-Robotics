import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan, Imu
import numpy as np
import time
import signal
from group_project import coordinates
import math
import os
from nav_msgs.msg import Odometry
import shutil
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration
import threading
import queue
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kP, kI, kD, kS):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_dif = 0
        self.err_prev = 0
        self.err_hist = queue.Queue(maxsize=self.kS)
        self.t_prev = 0

    def control(self, err, t):
        dt = t - self.t_prev
        if dt > 0.0:
            if self.err_hist.full():
                self.err_int -= self.err_hist.get()
            self.err_hist.put(err)
            self.err_int += err
            self.err_dif = (err - self.err_prev)
            u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt)
            self.err_prev = err
            self.t_prev = t
            return u
        return 0

class RoboNaut(Node):
    def __init__(self):
        super().__init__('robotnaut')

        # Initialize the CvBridge and the subscription to the camera images
        self.bridge = CvBridge()
        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10  # Increased depth to ensure we don't miss image frames
        )
        self.create_subscription(Image, '/camera/image_raw', self.camera_callback, image_qos_profile)

        # Parameters and state variables
        self.sign_detected = False
        self.entrance_attempts = 0
        self.coordinates_file_path = self.declare_parameter('coordinates_file_path', '').get_parameter_value().string_value
        self.coordinates = coordinates.get_module_coordinates(self.coordinates_file_path)

        # Action client for navigation
        self.navigation_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)  # 10 Hz

        self.navigation_goal_handle = None
        self.latest_image_data = None
        self.current_entrance = 1
        self.sign_detected_post_rotation = False

        # For window implementation
        self.in_green_room = False
        self.window_screenshot_counter = 0
        self.robot_stopped = False
        self.window_seen = False
        self.flag = False
        self.robot_close = False
        self.screenshot_taken = False
        self.contour = 0
        self.flag_idk = False
        self.latest_cv_image = None
        self.window_coordinates = None  # Store the window coordinates for cropping
        self.window_angle = None  # Store the window angle for alignment

        self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        self.last_movement_time = time.time()  # Timestamp of the last detected movement
        self.total_rotation_radians = 0
        self.full_rotation = False
        self.windows_ss_dir = '/uolstore/home/users/sc21ar/ros2_ws/src/group-project-group-13/windows_ss'
        self.prepare_capture_called = False

        # Wall-following parameters
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.robot_scan_sub = self.create_subscription(LaserScan, '/scan', self.robot_laserscan_callback, qos_profile)
        self.robot_ctrl_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)
        timer_period = 0.1  # Node execution time period (seconds)
        self.timer = self.create_timer(timer_period, self.robot_controller_callback)
        self.laserscan = []
        self.ctrl_msg = Twist()
        self.start_time = self.get_clock().now()
        self.in_room = False  # New flag to indicate when the robot is in the room
        self.wall_following_active = False  # New flag to indicate when wall-following should start
        self.pid_lat = PIDController(kP=0.5, kI=0.0, kD=0.2, kS=10)  # Adjusted gains
        self.pid_lon = PIDController(kP=0.1, kI=0.001, kD=0.05, kS=10)
        self.turning = False  # Flag to indicate if the robot is currently turning
        self.approaching_window = False  # Flag to indicate if the robot is approaching a window

        self.image_lock = threading.Lock()  # Lock for image data
        self.captured_windows = set()  # Track captured windows

        # For heuristic map
        self.heuristic_map = []
        self.map_resolution = 0.05  # meters per cell
        self.map_size = 200  # 10m x 10m area
        self.map_center = self.map_size // 2
        self.init_map()

        # Predicted window locations
        self.predicted_windows = []

        # Position tracking
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_orientation = 0.0

    def init_map(self):
        self.heuristic_map = np.zeros((self.map_size, self.map_size))

    def update_map(self, x, y, value):
        grid_x = int(x / self.map_resolution) + self.map_center
        grid_y = int(y / self.map_resolution) + self.map_center
        if 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size:
            self.heuristic_map[grid_x, grid_y] = value

    def save_heuristic_map(self, filename):
        plt.imshow(self.heuristic_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Heuristic Map')
        plt.savefig(filename)
        self.get_logger().info(f'Heuristic map saved as {filename}')

    def navigate_to_point(self, x, y):
        self.get_logger().info(f'Navigating to point: {x}, {y}')
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.w = 1.0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.navigation_client.wait_for_server()
        self.navigation_goal_handle = self.navigation_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.navigation_goal_handle.add_done_callback(self.goal_response_callback)

    def camera_callback(self, data):
        with self.image_lock:
            self.latest_image_data = data  # Update with the latest received image data

        if self.sign_detected:
            return

        cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        with self.image_lock:
            self.latest_cv_image = cv_image
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', cv_image)
        cv2.resizeWindow('camera_Feed', 320, 240)
        cv2.waitKey(3)
        red_mask = self.color_filter(cv_image, 'red')
        green_mask = self.color_filter(cv_image, 'green')

        red_detected = self.detect_sign(red_mask)
        green_detected = self.detect_sign(green_mask)

        if red_detected or green_detected:
            self.sign_detected = True
            self.handle_sign_detection('green' if green_detected else 'red')

        if self.in_green_room:
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 9000:
                    self.robot_close = True

            for contour in contours:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)

                    border_thickness = 10
                    x_start, y_start = max(x - border_thickness, 0), max(y - border_thickness, 0)
                    x_end, y_end = min(x + w + border_thickness, cv_image.shape[1]), min(y + h + border_thickness, cv_image.shape[0])

                    roi_with_border = gray[y_start:y_end, x_start:x_end]

                    _, white_thresh = cv2.threshold(roi_with_border, 220, 255, cv2.THRESH_BINARY)

                    has_white_border = np.any(white_thresh[:border_thickness, :] == 255) and \
                                       np.any(white_thresh[-border_thickness:, :] == 255) and \
                                       np.any(white_thresh[:, :border_thickness] == 255) and \
                                       np.any(white_thresh[:, -border_thickness:] == 255)

                    roi = gray[y:y + h, x:x + w]
                    _, black_thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
                    black_area = np.sum(black_thresh == 255)
                    total_area = roi.size

                    black_ratio = black_area / total_area

                    if has_white_border and black_ratio > 0.5:
                        if (x, y, w, h) not in self.captured_windows:  # Check if this window is already captured
                            self.window_seen = True
                            self.window_coordinates = (x, y, w, h)
                            self.window_angle = cv2.minAreaRect(contour)[-1]  # Get the angle of the window
                            self.get_logger().info("Window seen.")
                            self.predicted_windows.append((x, y))  # Save the window coordinates
                        else:
                            self.window_seen = False
                            self.get_logger().info("No new window seen.")
                        break

        if self.in_green_room and self.robot_stopped and not self.full_rotation:
            self.get_logger().info("Handling robot stop for window detection")
            self.handle_robot_stop()
        elif self.full_rotation:
            self.get_logger().info("Robot finished taking screenshots of the windows.")
            rclpy.shutdown()

    def handle_robot_stop(self):
        if self.in_green_room:
            if self.window_seen:
                self.get_logger().info("Window seen, attempting to capture...")
                self.approaching_window = True
            else:
                self.get_logger().info("No window seen, starting search rotation...")
                self.search_for_window()

    def approach_window(self):
        self.get_logger().info("Approaching the window.")
        self.approaching_window = True
        if self.window_angle and abs(self.window_angle) > 5:  # If the angle is significant
            self.adjust_orientation(self.window_angle)
        self.align_and_move_towards_window()

    def align_and_move_towards_window(self):
        self.get_logger().info("Aligning and moving towards the window.")
        self.screenshot_taken = False

        def align():
            while self.window_seen and self.get_valid_scan(min(self.laserscan[0:15] + self.laserscan[-15:])) > 2.0:
                twist = Twist()
                twist.linear.x = 0.05  # Slow down to improve accuracy
                twist.angular.z = 0.0
                self.publisher.publish(twist)
                time.sleep(0.1)
            self.stop()
            time.sleep(1)  # Allow time for the robot to stabilize
            self.adjust_to_perfect_alignment()
            self.check_distance_and_take_screenshot()

        threading.Thread(target=align).start()

    def adjust_orientation(self, angle):
        self.get_logger().info(f"Adjusting orientation by {angle} degrees.")
        self.smooth_turn(angle, 0.05, 0.01)  # Smooth turn with ramp up and down

    def adjust_to_perfect_alignment(self):
        self.get_logger().info("Adjusting to perfect alignment with the window.")
        angle_to_rotate = self.calculate_angle_to_window()
        if abs(angle_to_rotate) > 0.01:
            self.smooth_turn(angle_to_rotate, 0.05, 0.01)  # Smooth turn with ramp up and down

    def calculate_angle_to_window(self):
        # Heuristic to calculate angle to window based on the window's coordinates
        x, y, w, h = self.window_coordinates
        window_center_x = x + w / 2
        frame_center_x = self.latest_cv_image.shape[1] / 2
        angle_to_rotate = (window_center_x - frame_center_x) * 0.01  # Adjust the factor as needed
        self.get_logger().info(f"Calculated angle to rotate: {angle_to_rotate} degrees")
        return -angle_to_rotate  # Invert the angle to face the window

    def check_distance_and_take_screenshot(self):
        x, y, w, h = self.window_coordinates
        current_position = self.get_current_position()
        distance = self.calculate_distance_to_window(current_position, self.window_coordinates)
        self.get_logger().info(f"Distance check: distance from window {distance} meters")
        if abs(distance - 2.0) > 0.1:
            self.maintain_distance(distance)
        elif self.is_rectangle(x, y, w, h):
            self.stop()
            time.sleep(1)  # Pause for a second before taking the screenshot
            self.take_screenshot()

    def maintain_distance(self, current_distance):
        if current_distance > 2.0:
            move_speed = 0.05
        else:
            move_speed = -0.05

        def maintain(current_distance):
            while abs(current_distance - 2.0) > 0.1:
                twist = Twist()
                twist.linear.x = move_speed
                twist.angular.z = 0.0
                self.publisher.publish(twist)
                time.sleep(0.1)
                current_distance = self.calculate_distance_to_window(self.get_current_position(), self.window_coordinates)
                self.get_logger().info(f"Maintaining distance, current distance: {current_distance}")
            self.stop()
            time.sleep(1)  # Allow time for the robot to stabilize
            self.take_screenshot()

        threading.Thread(target=maintain, args=(current_distance,)).start()

    def calculate_distance_to_window(self, current_position, window_coordinates):
        x, y, w, h = window_coordinates
        window_center_x = x + w / 2
        window_center_y = y + h / 2
        current_x, current_y, _ = current_position
        return math.sqrt((window_center_x - current_x) ** 2 + (window_center_y - current_y) ** 2)

    def is_rectangle(self, x, y, w, h):
        # Check if the detected window is rectangular based on its aspect ratio
        aspect_ratio = float(w) / h
        self.get_logger().info(f"Checking if window is rectangular with aspect ratio: {aspect_ratio}")
        return 0.8 <= aspect_ratio <= 1.2  # Assuming a rectangle has an aspect ratio close to 1

    def take_screenshot(self):
        x, y, w, h = self.window_coordinates
        if not self.screenshot_taken and self.window_seen and self.window_coordinates:
            with self.image_lock:
                window_image = self.latest_cv_image[y:y + h, x:x + w]  # Crop the window area
            self.get_logger().info("Taking screenshot.")
            try:
                if not os.path.exists(self.windows_ss_dir):
                    os.makedirs(self.windows_ss_dir)
                filename = f'{self.windows_ss_dir}/window_{self.window_screenshot_counter}.jpg'
                cv2.imwrite(filename, window_image)
                self.get_logger().info(f"Screenshot saved as {filename}.")
            except Exception as e:
                self.get_logger().error(f"Failed to save screenshot: {str(e)}")
            self.window_screenshot_counter += 1
            self.screenshot_taken = True
            self.window_seen = False
            self.approaching_window = False
            self.captured_windows.add((x, y, w, h))  # Mark this window as captured
            self.resume_wall_following()

    def resume_wall_following(self):
        self.align_back_to_wall()
        self.wall_following_active = True  # Ensure wall-following is reactivated

    def smooth_turn(self, angle, max_speed, ramp_step):
        self.get_logger().info(f'Smooth turning by {angle} degrees.')
        turn_direction = np.sign(angle)
        current_speed = 0.0
        ramp_up_duration = abs(angle) / 2
        ramp_down_duration = abs(angle) / 2
        ramp_up_steps = int(ramp_up_duration / ramp_step)
        ramp_down_steps = int(ramp_down_duration / ramp_step)
        for step in range(ramp_up_steps):
            current_speed += max_speed / ramp_up_steps
            twist = Twist()
            twist.angular.z = current_speed * turn_direction
            self.publisher.publish(twist)
            time.sleep(ramp_step)
        for step in range(ramp_down_steps):
            current_speed -= max_speed / ramp_down_steps
            twist = Twist()
            twist.angular.z = current_speed * turn_direction
            self.publisher.publish(twist)
            time.sleep(ramp_step)
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        self.get_logger().info('Smooth turn completed.')

    def get_current_position(self):
        # Return the current position of the robot
        return (self.current_x, self.current_y, self.current_orientation)

    def align_back_to_wall(self):
        self.get_logger().info('Aligning back to wall.')
        left_scan = self.get_valid_scan(min(self.laserscan[75:105]))
        right_scan = self.get_valid_scan(min(self.laserscan[255:285]))
        error = right_scan - left_scan
        if abs(error) > 0.1:  # Only adjust if the error is significant
            turn_duration = abs(error) / 0.5  # Adjust this value based on robot's turning speed
            turn_speed = 0.05 if error < 0 else -0.05  # Slow down the turn speed
            twist = Twist()
            twist.angular.z = turn_speed
            self.publisher.publish(twist)
            time.sleep(turn_duration)
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            self.get_logger().info('Alignment back to wall completed.')

    def prepare_capture(self):
        if not self.prepare_capture_called:
            if os.path.exists(self.windows_ss_dir) and os.listdir(self.windows_ss_dir):
                for filename in os.listdir(self.windows_ss_dir):
                    file_path = os.path.join(self.windows_ss_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        self.get_logger().info(f"Failed to delete {file_path}. Reason: {e}")
            self.get_logger().info("windows_ss folder cleared for new captures.")
            self.prepare_capture_called = True

    def walk_forward_window(self, target_speed, ramp_duration):
        start_speed = 0.0
        ramp_step = 0.02
        ramp_interval = 0.1

        current_speed = start_speed
        steps = int(ramp_duration / ramp_interval)
        speed_increment = (target_speed - start_speed) / steps

        self.get_logger().info("Starting to walk forward.")
        for step in range(steps):
            current_speed += speed_increment
            desired_velocity = Twist()
            desired_velocity.linear.x = current_speed
            self.publisher.publish(desired_velocity)
            time.sleep(ramp_interval)

        desired_velocity.linear.x = target_speed
        self.publisher.publish(desired_velocity)
        self.get_logger().info(f"Reached target speed of {target_speed} m/s.")

    def search_for_window(self):
        self.rotate_for_duration(-0.2, 2.0)  # Slower rotation speed and longer duration for more controlled rotation
        time.sleep(0.5)

    def rotate_for_duration(self, rotation_speed, duration):
        self.get_logger().info('Starting rotation.')
        if self.contour == 0:
            self.total_rotation_radians = 0

        rotation_this_call = rotation_speed * duration
        self.total_rotation_radians += rotation_this_call

        if abs(self.total_rotation_radians) >= 2 * math.pi:
            self.get_logger().info("Cumulative rotation has reached or exceeded 360 degrees.")
            self.full_rotation = True

        rotation_velocity = Twist()
        rotation_velocity.angular.z = rotation_speed

        self.publisher.publish(rotation_velocity)
        time.sleep(duration)

        rotation_velocity.angular.z = 0.0
        self.publisher.publish(rotation_velocity)
        self.get_logger().info('Rotation completed.')

    def odometry_callback(self, msg):
        current_speed = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2 + msg.twist.twist.linear.z**2)
        current_angular_speed = abs(msg.twist.twist.angular.z)

        movement_threshold = 0.01
        if current_speed > movement_threshold or current_angular_speed > movement_threshold:
            self.last_movement_time = time.time()

        # Update current position and orientation
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, self.current_orientation = self.euler_from_quaternion(orientation_q)

        current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.update_map(current_position[0], current_position[1], 1)  # Mark the robot's path on the map

    def euler_from_quaternion(self, q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x, y, z, w = q.x, q.y, q.z, q.w
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z

    def walk_forward(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.2
        self.publisher.publish(desired_velocity)

    def stop(self):
        twist_msg = Twist()
        self.publisher.publish(twist_msg)

    def color_filter(self, img, color):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if color == 'red':
            lower_bound1 = np.array([0, 120, 70])
            upper_bound1 = np.array([10, 255, 255])
            lower_bound2 = np.array([170, 120, 70])
            upper_bound2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
            mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
            mask = mask1 + mask2
        elif color == 'green':
            lower_bound = np.array([36, 25, 25])
            upper_bound = np.array([86, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            mask = np.zeros(img.shape[:2], dtype="uint8")
        return mask

    def detect_sign(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if area > 500 and 0.7 <= circularity <= 1.2:
                return True
        return False

    def rotate_robot(self):
        self.get_logger().info('Initiating rotation.')
        rotation_velocity = Twist()
        rotation_velocity.angular.z = math.pi / 6

        rotate_time = 6
        end_time = time.time() + rotate_time

        while rclpy.ok() and time.time() < end_time:
            self.publisher.publish(rotation_velocity)
            time.sleep(0.1)

        rotation_velocity.angular.z = 0.0
        self.publisher.publish(rotation_velocity)
        self.get_logger().info('Rotation completed.')

    def handle_sign_detection(self, sign_color):
        self.get_logger().info(f'{sign_color.capitalize()} sign detected.')

        if sign_color == 'red':
            self.rotate_robot()
            self.sign_detected_post_rotation = True
            self.process_latest_image_for_green()

        elif sign_color == 'green':
            if self.sign_detected_post_rotation:
                self.get_logger().info('Green sign detected post rotation. Navigating into the room.')
                self.navigate_to_point_and_start_wall_following(self.current_entrance)
                self.sign_detected_post_rotation = False
                self.in_room = True  # Robot has entered the room
            else:
                if self.current_entrance == 1:
                    self.get_logger().info('Green sign detected at entrance 1. Navigating into the room.')
                    self.navigate_to_point_and_start_wall_following(1)
                    self.in_green_room = True
                    self.in_room = True  # Robot has entered the room
                elif self.current_entrance == 2:
                    self.get_logger().info('Green sign detected at entrance 2. Navigating into the room.')
                    self.navigate_to_point_and_start_wall_following(2)
                    self.in_green_room = True
                    self.in_room = True  # Robot has entered the room

        self.sign_detected = False

    def process_latest_image_for_green(self):
        time.sleep(2)
        green_detected = self.detect_green_circle(self.latest_image_data)

        if green_detected:
            self.get_logger().info('Green sign detected after rotation. Navigating into the room.')
            self.navigate_to_point_and_start_wall_following(self.current_entrance)
            self.current_entrance = 2
        else:
            self.get_logger().info('No green sign detected after rotation. Moving to the next entrance.')
            self.current_entrance = 2 if self.current_entrance == 1 else 1
            self.get_logger().info(f'Navigating to entrance: {self.current_entrance}')
            self.navigate_to_entrance(self.current_entrance)

    def detect_green_circle(self, image_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
            return False

        green_mask = self.color_filter(cv_image, 'green')
        return self.detect_sign(green_mask)

    def navigate_to_point_and_start_wall_following(self, entrance_number):
        if entrance_number == 1:
            inside_x = (2 * self.coordinates.module_1.entrance.x + self.coordinates.module_1.center.x) / 7
            inside_y = (2 * self.coordinates.module_1.entrance.y + self.coordinates.module_1.center.y) / 7
            self.get_logger().info('Navigating to point inside the room from entrance 1.')
            self.navigate_to_point(inside_x, inside_y)
        elif entrance_number == 2:
            inside_x = (2 * self.coordinates.module_2.entrance.x + self.coordinates.module_2.center.x) / 3
            inside_y = (2 * self.coordinates.module_2.entrance.y + self.coordinates.module_2.center.y) / 3
            self.get_logger().info('Navigating to point inside the room from entrance 2.')
            self.navigate_to_point(inside_x, inside_y)
        else:
            self.get_logger().info('Invalid entrance number for inside room navigation.')

        self.get_logger().info("Predicting window locations.")
        self.predict_window_locations()
        self.get_logger().info("Navigating to predicted window locations.")
        self.navigate_to_predicted_windows()

    def predict_window_locations(self):
        # Example heuristic to predict window locations
        # Assuming windows are placed uniformly along the walls
        room_center = self.coordinates.module_1.center if self.current_entrance == 1 else self.coordinates.module_2.center
        room_size = 5.0  # Assuming a room size of 5x5 meters

        # Predict four window locations (one on each wall)
        self.predicted_windows = [
            (room_center.x - room_size / 2, room_center.y),  # Left wall
            (room_center.x + room_size / 2, room_center.y),  # Right wall
            (room_center.x, room_center.y - room_size / 2),  # Bottom wall
            (room_center.x, room_center.y + room_size / 2)   # Top wall
        ]

    def navigate_to_predicted_windows(self):
        for window in self.predicted_windows:
            self.get_logger().info(f"Navigating to predicted window location: {window}")
            self.navigate_to_point(window[0], window[1])
            time.sleep(5)  # Allow time for navigation and window detection
            if self.window_seen:
                self.get_logger().info("Window detected at predicted location.")
                break

    def navigate_to_entrance(self, entrance_number):
        if entrance_number == 1:
            self.navigate_to_point(self.coordinates.module_1.entrance.x, self.coordinates.module_1.entrance.y + 0.5) # Adjusting to right wall
        elif entrance_number == 2:
            self.navigate_to_point(self.coordinates.module_2.entrance.x, self.coordinates.module_2.entrance.y + 0.5) # Adjusting to right wall
        else:
            self.get_logger().info('Invalid entrance number for entrance navigation.')

    def goal_response_callback(self, future):
        self.navigation_goal_handle = future.result()
        if not self.navigation_goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        self.get_logger().info('Goal accepted')
        self.navigation_goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')
        if self.in_room:
            self.get_logger().info("Robot has entered the room. Starting wall-following behavior.")
            self.start_wall_following()

    def start_wall_following(self):
        self.in_room = True
        self.start_time = self.get_clock().now()
        self.wall_following_active = True  # Activate wall-following

    def feedback_callback(self, feedback_msg):
        pass

    def robot_laserscan_callback(self, msg):
        self.laserscan = msg.ranges

    def robot_controller_callback(self):
        DELAY = 4.0
        if self.wall_following_active and self.get_clock().now() - self.start_time > Duration(seconds=DELAY):
            front_scan = self.get_valid_scan(min(self.laserscan[0:15] + self.laserscan[-15:]))
            left_scan = self.get_valid_scan(min(self.laserscan[75:105]))
            right_scan = self.get_valid_scan(min(self.laserscan[255:285]))
            left_error = left_scan - 2.0  # Desired distance from the wall
            right_error = right_scan - 2.0  # Desired distance from the wall
            cte = left_error - right_error  # Cross-track error
            tstamp = time.time()

            self.get_logger().info(f"front_scan: {front_scan}, left_scan: {left_scan}, right_scan: {right_scan}")
            self.get_logger().info(f"left_error: {left_error}, right_error: {right_error}, cte: {cte}")

            if self.window_seen and not self.approaching_window:
                self.get_logger().info("Approaching window to take screenshot.")
                self.approach_window()
            elif self.approaching_window:
                self.align_and_move_towards_window()
            elif not self.turning and front_scan < 1.0:  # Obstacle detected ahead, avoid obstacle if closer than 1 meter
                self.avoid_obstacle()
            else:
                self.ctrl_msg.linear.x = 0.2  # Ensure a minimum forward movement
                self.ctrl_msg.angular.z = self.pid_lat.control(cte, tstamp)
                self.get_logger().info("No obstacle ahead, moving forward.")

            self.robot_ctrl_pub.publish(self.ctrl_msg)
            self.get_logger().info(f"Published LIN_VEL: {self.ctrl_msg.linear.x}, ANG_VEL: {self.ctrl_msg.angular.z}")
            self.get_logger().info(f'Relative distance error from walls is {round(cte, 4)} m')
        else:
            self.get_logger().info('Initializing or waiting for the delay period to complete...')

    def avoid_obstacle(self):
        self.get_logger().info('Avoiding obstacle.')
        self.turning = True

        # Backup slightly
        backup_velocity = Twist()
        backup_velocity.linear.x = -0.2
        self.publisher.publish(backup_velocity)
        time.sleep(1.0)

        # Turn to avoid obstacle
        def turn_to_avoid():
            left_scan = self.get_valid_scan(min(self.laserscan[75:105]))
            right_scan = self.get_valid_scan(min(self.laserscan[255:285]))
            if left_scan < right_scan:
                # Turn right
                turn_velocity = Twist()
                turn_velocity.angular.z = -math.pi / 6
                self.publisher.publish(turn_velocity)
                time.sleep(1.0)
                self.publisher.publish(Twist())  # Stop turning
            else:
                # Turn left
                turn_velocity = Twist()
                turn_velocity.angular.z = math.pi / 6
                self.publisher.publish(turn_velocity)
                time.sleep(1.0)
                self.publisher.publish(Twist())  # Stop turning

        turn_to_avoid()

        # Move forward
        forward_velocity = Twist()
        forward_velocity.linear.x = 0.2
        self.publisher.publish(forward_velocity)
        time.sleep(2.0)

        self.turning = False
        self.get_logger().info('Obstacle avoided.')

    def get_valid_scan(self, scan):
        if scan == float('inf') or scan == float('-inf') or scan != scan:  # Check for inf and NaN
            return 3.5  # Some high value that represents no obstacle
        return scan

def signal_handler(sig, frame):
    global robotnaut  # Make sure robotnaut is accessible here
    robotnaut.save_heuristic_map('/uolstore/home/users/sc21ar/ros2_ws/src/group-project-group-13/heuristic_map.png')
    rclpy.shutdown()
    os._exit(0)

def main(args=None):
    rclpy.init(args=args)
    global robotnaut
    robotnaut = RoboNaut()
    
    signal.signal(signal.SIGINT, signal_handler)

    robotnaut.navigate_to_entrance(1)

    try:
        rclpy.spin(robotnaut)
    except KeyboardInterrupt:
        pass

    robotnaut.save_heuristic_map('/uolstore/home/users/sc21ar/ros2_ws/src/group-project-group-13/heuristic_map.png')
    robotnaut.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
