import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from typing import Dict, List
import pandas as pd
from pathlib import Path
import sys
import os

# Add the SeQUeNCe/parallel/src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SeQUeNCe", "parallel", "src"))

from sequence.kernel.event import Event
from sequence.kernel.process import Process
from p_timeline import ParallelTimeline

class BenchmarkNode:
    def __init__(self, name: str, timeline: ParallelTimeline, event_rate: float):
        self.name = name
        self.timeline = timeline
        self.event_rate = event_rate
        timeline.entities[name] = self
        
    def generate_event(self) -> Event:
        """Generate a new event with random destination."""
        # Randomly choose destination process
        dest_id = np.random.randint(0, MPI.COMM_WORLD.Get_size())
        dest_name = f"node_{dest_id}"
        
        # Create process for the event
        if dest_id == MPI.COMM_WORLD.Get_rank():
            process = Process(self, "handle_event", [])
        else:
            process = Process(dest_name, "handle_event", [])
            
        # Schedule event with exponential inter-arrival time
        event_time = self.timeline.time + int(np.random.exponential(1/self.event_rate))
        return Event(event_time, process)
        
    def init(self):
        """Initialize node with some events."""
        for _ in range(100):  # Start with 100 events
            event = self.generate_event()
            self.timeline.schedule(event)
            
    def handle_event(self):
        """Handle an event and generate new ones."""
        # Generate 1-3 new events
        num_new = np.random.randint(1, 4)
        for _ in range(num_new):
            event = self.generate_event()
            self.timeline.schedule(event)

def run_benchmark(config: Dict) -> Dict:
    """Run the benchmark with given configuration."""
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    
    # Create timeline
    timeline = ParallelTimeline(
        lookahead=config['lookahead'],
        stop_time=config['stop_time']
    )
    
    # Create nodes
    nodes = []
    for i in range(size):
        if i == rank:
            node = BenchmarkNode(f"node_{i}", timeline, config['event_rate'])
            nodes.append(node)
        else:
            timeline.add_foreign_entity(f"node_{i}", i)
    
    # Initialize nodes
    for node in nodes:
        node.init()
    
    # Run simulation
    start_time = time.time()
    timeline.run()
    end_time = time.time()
    
    # Collect statistics
    stats = timeline.get_stats()
    stats['wall_time'] = end_time - start_time
    
    return stats

def plot_results(results: List[Dict], output_dir: str):
    """Plot benchmark results."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot wall time vs number of processes
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_processes'], df['wall_time'], 'o-', label='Wall Time')
    plt.xlabel('Number of Processes')
    plt.ylabel('Wall Time (s)')
    plt.title('Wall Time vs Number of Processes')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/wall_time.png')
    plt.close()
    
    # Plot communication time vs computing time
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_processes'], df['communication_time'], 'o-', label='Communication Time')
    plt.plot(df['num_processes'], df['computing_time'], 'o-', label='Computing Time')
    plt.xlabel('Number of Processes')
    plt.ylabel('Time (s)')
    plt.title('Communication vs Computing Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/time_breakdown.png')
    plt.close()
    
    # Plot load balance
    plt.figure(figsize=(10, 6))
    plt.plot(df['num_processes'], df['avg_load'], 'o-', label='Average Load')
    plt.xlabel('Number of Processes')
    plt.ylabel('Average Load')
    plt.title('Load Balance')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/load_balance.png')
    plt.close()
    
    # Save raw data
    df.to_csv(f'{output_dir}/benchmark_results.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Benchmark parallel timeline implementation')
    parser.add_argument('--lookahead', type=int, default=1000, help='Lookahead time')
    parser.add_argument('--stop-time', type=int, default=10000, help='Simulation stop time')
    parser.add_argument('--event-rate', type=float, default=0.1, help='Event generation rate')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory')
    args = parser.parse_args()
    
    # Run benchmark
    config = {
        'lookahead': args.lookahead,
        'stop_time': args.stop_time,
        'event_rate': args.event_rate
    }
    
    results = run_benchmark(config)
    
    # Gather results from all processes
    all_results = MPI.COMM_WORLD.gather(results, root=0)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Process results
        processed_results = []
        for i, result in enumerate(all_results):
            result['num_processes'] = MPI.COMM_WORLD.Get_size()
            result['rank'] = i
            processed_results.append(result)
        
        # Plot results
        plot_results(processed_results, args.output_dir)
        
        # Save raw results
        with open(f'{args.output_dir}/raw_results.json', 'w') as f:
            json.dump(processed_results, f, indent=2)

if __name__ == '__main__':
    main() 