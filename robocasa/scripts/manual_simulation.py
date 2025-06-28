import gymnasium as gym
import zmq
import torch
from io import BytesIO
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
from gymnasium.vector import SyncVectorEnv

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

env_id = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
env = SyncVectorEnv([lambda: gym.make(env_id, enable_render=True) for _ in range(1)])

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://127.0.0.1:5555")
socket.setsockopt( zmq.RCVTIMEO, 500 )
vis_socket = context.socket(zmq.REQ)
vis_socket.connect(f"tcp://127.0.0.1:5556")
vis_socket.setsockopt( zmq.RCVTIMEO, 500 )

obs, _ = env.reset()

for i in range(50):
    for i in range(16):
        obs["video.ego_view"] = obs.pop("video.ego_view_bg_crop_pad_res256_freq20")
        request = {"endpoint": "get_action", "data": obs}
        socket.send(TorchSerializer.to_bytes(request))
        response = TorchSerializer.from_bytes(socket.recv())
        obs, rewards, terminations, truncations, env_infos = env.step({key: [array[i]] for key, array in response.items()})
        vis = {"observations": request["data"]}
        vis_socket.send(TorchSerializer.to_bytes(vis))
        vis_socket.recv()