"""
Neuron models module: LIF and Hodgkin-Huxley neuron implementations.
"""

from .lif_neuron import LIFNeuron, simulate_lif_neuron
from .neuron_population import NeuronPopulation, simulate_population

__all__ = [
    'LIFNeuron',
    'simulate_lif_neuron',
    'NeuronPopulation',
    'simulate_population'
]
