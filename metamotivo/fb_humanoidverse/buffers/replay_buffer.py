from metamotivo.buffers.buffers import DictBuffer
import dataclasses

@dataclasses.dataclass
class ReplayBuffer(DictBuffer):

    def __post_init__(self):
        super().__post_init__()
