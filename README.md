# SeQuEnCe - Quantum Network Simulator

SeQuEnCe is a high-performance quantum network simulator that implements discrete event simulation with optimized event queue processing and parallel execution capabilities.

## Features

- Discrete Event Simulation (DES) for quantum networks
- Optimized event queue with batching capabilities
- Parallel processing support
- Comprehensive benchmarking tools
- Performance profiling and visualization

## Installation

```bash
pip install -e .
```

## Usage

### Basic Simulation

```python
from SeQUeNCe.simulation_managers.des_manager import DESManager
from SeQUeNCe.topology.topology import Topology

# Create and configure your topology
topology = Topology()

# Initialize the simulation manager
manager = DESManager(topology)

# Run the simulation
manager.run()
```

### Benchmarking

```python
from utils.benchmark import run_benchmark

# Run benchmark with 1000 events
run_benchmark(1000)
```

## Performance Optimization

The simulator includes several optimizations:
- Event queue batching for improved throughput
- Parallel processing capabilities
- Memory-efficient event handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 