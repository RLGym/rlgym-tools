import socket
import struct
from typing import Dict, Any

import numpy as np
from rlgym.api import Renderer
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.api import PhysicsObject

DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 4577  # Default RocketSimVis port

class BakkesRocketSimPlayer(Renderer[GameState]):
    """
    A renderer that sends game state information to BakkesRocketSimPlayer.

    This is just the client side, you need to run BakkesRocketSimPlayer to see the visualization.
    Code is here: https://github.com/ZealanL/BakkesRocketSimPlayer
    """
    def __init__(self, udp_ip=DEFAULT_UDP_IP, udp_port=DEFAULT_UDP_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.udp_ip = udp_ip
        self.udp_port = udp_port

    @staticmethod
    def write_physobj(physobj: PhysicsObject):
        forward = physobj.forward.tolist()
        right = physobj.right.tolist()
        up = physobj.up.tolist()
        return (
            BakkesRocketSimPlayer.pack_vec(physobj.position.tolist()) +
            BakkesRocketSimPlayer.pack_vec(forward) +
            BakkesRocketSimPlayer.pack_vec(right) +
            BakkesRocketSimPlayer.pack_vec(up) +
            BakkesRocketSimPlayer.pack_vec(physobj.linear_velocity.tolist()) +
            BakkesRocketSimPlayer.pack_vec(physobj.angular_velocity.tolist())
        )

    @staticmethod
    def pack_vec(vec: list):
        return struct.pack("<f", vec[0]) + struct.pack("<f", vec[1]) + struct.pack("<f", vec[2])

    @staticmethod
    def write_car(car: Car, controls=None):
        bytes = b""

        # Team number
        bytes += struct.pack("B", int(car.team_num))

        # car physics
        bytes += BakkesRocketSimPlayer.write_physobj(car.physics)

        # Car boost
        bytes += struct.pack("<f", car.boost_amount)

        # Car state bools
        bytes += struct.pack("B", car.on_ground)
        bytes += struct.pack("B", car.on_ground)  # has jump
        bytes += struct.pack("B", car.can_flip)  # has flip
        bytes += struct.pack("B", car.is_demoed)

        if controls is not None:
            if isinstance(controls, np.ndarray):
                controls = [struct.pack("<f", item) for item in controls.tolist()]
            else:
                controls = [struct.pack("<f", item) for item in controls]

        return bytes

    def render(self, state: GameState, shared_info: Dict[str, Any]) -> Any:
        msg = b""

        tick_count = 0 # not implemented yet

        msg += struct.pack("I", tick_count)
        
        # Send cars
        msg += struct.pack("I", len(state.cars))
        
        i = 0
        for player, action in zip(state.cars.values(), shared_info.get("actions", [])):
            msg += BakkesRocketSimPlayer.write_car(player, action)
            i += 1

        # Send ball
        msg += BakkesRocketSimPlayer.write_physobj(state.ball)
        self.sock.sendto(msg, (self.udp_ip, self.udp_port))

    def close(self):
        pass
