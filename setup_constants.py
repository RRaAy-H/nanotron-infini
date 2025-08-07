from dataclasses import dataclass, field
from typing import Optional

@dataclass
class InfiniAttentionConfig:
    segment_length: int = 64
    turn_on_memory: bool = True
    balance_init_type: str = "zeros"
    balance_act_type: str = "orig_sigmoid"

@dataclass
class Config:
    infini_attention: InfiniAttentionConfig = field(default_factory=InfiniAttentionConfig)

# Import and configure constants
from nanotron import constants

# Set up the configuration
constants.CONFIG = Config()

print("Infini attention constants configured successfully!")
print(f"Segment length: {constants.CONFIG.infini_attention.segment_length}")
print(f"Turn on memory: {constants.CONFIG.infini_attention.turn_on_memory}")
print(f"Balance init type: {constants.CONFIG.infini_attention.balance_init_type}")
print(f"Balance act type: {constants.CONFIG.infini_attention.balance_act_type}")
