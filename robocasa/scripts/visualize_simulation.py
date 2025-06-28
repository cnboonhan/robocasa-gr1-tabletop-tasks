import zmq
import sys
import time
from io import BytesIO
import torch
import rerun as rr

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()
    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://127.0.0.1:5556")
rr.init("robocasa_visualizer", spawn=True)

while True:
    response = TorchSerializer.from_bytes(socket.recv())
    for key in response["observations"]:
        if key == 'video.ego_view':
            rr.log(f"observations.{key}", rr.Image(response["observations"][key]))
    #     else:
    #         rr.log(f"observations.{key}", rr.Tensor(response["observations"][key]))
    # for key in response["actions"]:
    #     rr.log(f"actions.{key}", rr.Tensor(response["actions"][key]))
    socket.send(b"1")