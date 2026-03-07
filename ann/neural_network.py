"""
Compatibility wrapper so that autograders importing
`ann.neural_network.NeuralNetwork` can find the class.

The actual implementation lives in `src.ann.neural_network`.
"""

from src.ann.neural_network import NeuralNetwork

__all__ = ["NeuralNetwork"]

