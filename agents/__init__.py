"""
Economic simulation agent definitions.

This package contains the agent classes that form the core of the economic simulation.
Each agent type represents a different economic actor with its own decision-making
process and behavioral characteristics.
"""

from .producer import ProducerAgent
from .consumer import ConsumerAgent
from .resource import ResourceAgent

# Define what's available when using `from agents import *`
__all__ = [
    'ProducerAgent',
    'ConsumerAgent',
    'ResourceAgent',
]