import json
import socket
from typing import Dict, Any

import numpy as np
from rlgym.api import Renderer
from rlgym.rocket_league.api import GameState, Car

DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 9273  # Default RocketSimVis port

BUTTON_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")


class RocketSimVisRenderer(Renderer[GameState]):
    """
    A renderer that sends game state information to RocketSimVis.

    This is just the client side, you need to run RocketSimVis to see the visualization.
    Code is here: https://github.com/ZealanL/RocketSimVis
    """
    def __init__(self, udp_ip=DEFAULT_UDP_IP, udp_port=DEFAULT_UDP_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
        self.udp_ip = udp_ip
        self.udp_port = udp_port

    @staticmethod
    def write_physobj(physobj):
        j = {
            'pos': physobj.position.tolist(),
            'forward': physobj.forward.tolist(),
            'up': physobj.up.tolist(),
            'vel': physobj.linear_velocity.tolist(),
            'ang_vel': physobj.angular_velocity.tolist()
        }

        return j

    @staticmethod
    def write_car(car: Car, controls=None):
        j = {
            'team_num': int(car.team_num),
            'phys': RocketSimVisRenderer.write_physobj(car.physics),
            'boost_amount': car.boost_amount,
            'on_ground': bool(car.on_ground),
            "has_flipped_or_double_jumped": bool(car.has_flipped or car.has_double_jumped),
            'is_demoed': bool(car.is_demoed),
            'has_flip': bool(car.can_flip)
        }

        if controls is not None:
            if isinstance(controls, np.ndarray):
                controls = {
                    k: float(v)
                    for k, v in zip(BUTTON_NAMES, controls)
                }
            j['controls'] = controls

        return j

    def render(self, state: GameState, shared_info: Dict[str, Any]) -> Any:
        if "controls" in shared_info:
            controls = shared_info["controls"]
        else:
            controls = {}
        j = {
            'ball_phys': self.write_physobj(state.ball),
            'cars': [
                self.write_car(car, controls.get(agent_id))
                for agent_id, car in state.cars.items()
            ],
            'boost_pad_states': (state.boost_pad_timers <= 0).tolist()
        }

        self.sock.sendto(json.dumps(j).encode('utf-8'), (self.udp_ip, self.udp_port))

    def close(self):
        pass
