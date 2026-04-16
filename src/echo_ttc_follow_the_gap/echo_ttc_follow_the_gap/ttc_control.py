#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Joy

class TTCControl(Node):
    def __init__(self):
        super().__init__('ttc_control')

        # -------- PARÁMETROS ROS (leídos desde launch) --------
        self.declare_parameter('ttc_min',              0.6)
        self.declare_parameter('v_max',                1.6)
        self.declare_parameter('v_min',                0.15)   # velocidad mínima garantizada
        self.declare_parameter('kp_steering',          1.2)
        self.declare_parameter('max_steering',         2.5)
        self.declare_parameter('steering_slowdown',    1.5)
        self.declare_parameter('angle_deadband',       0.05)

        self.ttc_ref        = self.get_parameter('ttc_min').value
        self.v_max          = self.get_parameter('v_max').value
        self.v_min          = self.get_parameter('v_min').value
        self.kp             = self.get_parameter('kp_steering').value
        self.max_steering   = self.get_parameter('max_steering').value
        self.slow_gain      = self.get_parameter('steering_slowdown').value
        self.deadband       = self.get_parameter('angle_deadband').value
        self.rb_pressed = False  # Estado del botón RB del joystick

        # -------- ESTADO --------
        self.gap_angle = 0.0
        self.min_ttc   = float('inf')

        # -------- SUBS --------
        self.create_subscription(Float32, '/gap_angle', self.gap_callback, 10)
        self.create_subscription(Float32, '/min_ttc',   self.ttc_callback, 10)
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        # -------- PUB --------
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_ctrl', 10)

        # -------- TIMER 20 Hz --------
        self.create_timer(0.05, self.compute_and_publish)

        self.get_logger().info(
            f"TTCControl OK | v_max={self.v_max} v_min={self.v_min} "
            f"kp={self.kp} ttc_ref={self.ttc_ref}"
        )

    # -------------------------------------------------------
    def gap_callback(self, msg: Float32):
        self.gap_angle = msg.data

    def ttc_callback(self, msg: Float32):
        self.min_ttc = msg.data

    def joy_callback(self, msg):
        # Handle joystick messages if needed
        self.rb_pressed = (msg.buttons[5] == 1)


    # -------------------------------------------------------
    def compute_and_publish(self):

        # -------- STEERING --------
        angle = self.gap_angle if abs(self.gap_angle) >= self.deadband else 0.0

        steering = self.kp * angle
        steering = steering / (1.0 + abs(steering))*self.max_steering                    # suavizado no lineal → (-1, 1)
        steering = max(-self.max_steering, min(self.max_steering, steering))

        if steering > 0.1 and steering < 1.1:
            steering = 1.1
        elif steering < -0.1 and steering > -1.1:
            steering = -1.1

        # -------- VELOCIDAD BASE (TTC) --------
        if self.min_ttc <= 0.01:
            scale = 0.0
        else:
            scale = min(self.min_ttc / self.ttc_ref, 1.0)

        base_vel = self.v_max * scale

        # -------- ACOPLAMIENTO GIRO–VELOCIDAD --------
        velocity = base_vel / (1.0 + self.slow_gain * abs(steering))

        # -------- VELOCIDAD MÍNIMA GARANTIZADA --------
        # Colisión gestionada por nodo externo; aquí solo garantizamos avance.
        velocity = max(self.v_max, velocity)
        velocity = min(self.v_min, velocity)


        # -------- PUBLICAR --------
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        if(not self.rb_pressed):
                velocity = 0.0
                steering = 0.0
                self.get_logger().info("Dead man activado")
        cmd.twist.linear.x  = float(velocity)
        cmd.twist.angular.z = float(steering)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TTCControl())
    rclpy.shutdown()

if __name__ == '__main__':
    main()