#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class CmdScaler(Node):

    def __init__(self):
        super().__init__('cmd_scaler')

        # Subscriber
        self.sub = self.create_subscription(
            TwistStamped,
            '/diffdrive_controller/cmd_vel',
            self.cmd_callback,
            10
        )

        # Publisher
        self.pub = self.create_publisher(
            TwistStamped,
            '/cmd_scalated',
            10
        )

    def cmd_callback(self, msg: TwistStamped):
        out = TwistStamped()

        # Mantener header (importante para sincronización)
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id

        # ---- Escalamiento ----
        x = msg.twist.linear.x
        w = msg.twist.angular.z

        # Ecuación lineal: y ≈ 2.11x - 2.21
        y = 2.11 * x - 2.21

        # Ángulo: y = x (por ahora passthrough)
        w_out = w

        # Asignación
        out.twist.linear.x = y
        out.twist.angular.z = w_out

        # Publicar
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdScaler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()