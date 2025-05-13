import time
import cProfile
import pstats
import io
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os
import sys

# Add the SeQUeNCe directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SeQUeNCe"))

from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline

class DummyProcess(Process):
    def __init__(self, owner, name: str):
        super().__init__(owner, name, [])
        self.counter = 0

    def run(self):
        self.counter += 1

def generate_events(num_events: int, time_range: int) -> List[Event]:
    """Generate a list of random events for testing."""
    events = []
    for _ in range(num_events):
        time = np.random.randint(0, time_range)
        process = DummyProcess(None, "dummy")
        event = Event(time, process)
        events.append(event)
    return events

def save_profile_stats(profiler: cProfile.Profile, filename: str):
    """Save detailed profiling statistics to a file."""
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    with open(filename, 'w') as f:
        stats.print_stats()
        stats.print_callers()

def run_benchmark(num_events: int, time_range: int, batch_size: int = 1000) -> Dict:
    """Run benchmark comparing original and optimized implementations."""
    # Generate test events
    events = generate_events(num_events, time_range)
    
    # Test original implementation
    original_timeline = Timeline()
    original_timeline.events.batch_size = 1  # Disable batching for original
    
    # Profile original implementation
    original_profiler = cProfile.Profile()
    original_profiler.enable()
    
    start_time = time.time()
    for event in events:
        original_timeline.schedule(event)
    original_schedule_time = time.time() - start_time
    
    start_time = time.time()
    while not original_timeline.events.isempty():
        original_timeline.events.pop()
    original_pop_time = time.time() - start_time
    
    original_profiler.disable()
    
    # Test optimized implementation
    optimized_timeline = Timeline()
    optimized_timeline.events.batch_size = batch_size
    
    # Profile optimized implementation
    optimized_profiler = cProfile.Profile()
    optimized_profiler.enable()
    
    start_time = time.time()
    for event in events:
        optimized_timeline.schedule(event)
    optimized_schedule_time = time.time() - start_time
    
    start_time = time.time()
    while not optimized_timeline.events.isempty():
        optimized_timeline.events.pop()
    optimized_pop_time = time.time() - start_time
    
    optimized_profiler.disable()
    
    # Save profiling results
    save_profile_stats(original_profiler, f'profile_original_{num_events}.txt')
    save_profile_stats(optimized_profiler, f'profile_optimized_{num_events}.txt')
    
    # Get event list statistics
    event_stats = optimized_timeline.events.get_stats()
    
    return {
        "num_events": num_events,
        "time_range": time_range,
        "batch_size": batch_size,
        "original_schedule_time": original_schedule_time,
        "original_pop_time": original_pop_time,
        "optimized_schedule_time": optimized_schedule_time,
        "optimized_pop_time": optimized_pop_time,
        "schedule_speedup": original_schedule_time / optimized_schedule_time,
        "pop_speedup": original_pop_time / optimized_pop_time,
        "event_stats": event_stats
    }

def plot_performance_comparison(results_list: List[Dict]):
    """Create visualizations of the benchmark results."""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # Extract data
    num_events = [r['num_events'] for r in results_list]
    schedule_speedups = [r['schedule_speedup'] for r in results_list]
    pop_speedups = [r['pop_speedup'] for r in results_list]
    batch_sizes = [r['batch_size'] for r in results_list]
    avg_batch_sizes = [r['event_stats']['avg_batch_size'] for r in results_list]
    merge_times = [r['event_stats']['batch_merge_time'] for r in results_list]
    
    # Plot 1: Schedule Time Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(num_events, [r['original_schedule_time'] for r in results_list], 'o-', label='Original')
    ax1.plot(num_events, [r['optimized_schedule_time'] for r in results_list], 's-', label='Optimized')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Events')
    ax1.set_ylabel('Schedule Time (s)')
    ax1.set_title('Schedule Operation Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Pop Time Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(num_events, [r['original_pop_time'] for r in results_list], 'o-', label='Original')
    ax2.plot(num_events, [r['optimized_pop_time'] for r in results_list], 's-', label='Optimized')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Events')
    ax2.set_ylabel('Pop Time (s)')
    ax2.set_title('Pop Operation Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Speedup Comparison
    ax3 = fig.add_subplot(gs[1, :])
    width = 0.35
    x = np.arange(len(num_events))
    ax3.bar(x - width/2, schedule_speedups, width, label='Schedule Speedup')
    ax3.bar(x + width/2, pop_speedups, width, label='Pop Speedup')
    ax3.set_xlabel('Test Configuration')
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Performance Speedup')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{n:,} events' for n in num_events])
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Batch Analysis
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(num_events, batch_sizes, 'o-', label='Target Batch Size')
    ax4.plot(num_events, avg_batch_sizes, 's-', label='Actual Avg Batch Size')
    ax4.set_xscale('log')
    ax4.set_xlabel('Number of Events')
    ax4.set_ylabel('Batch Size')
    ax4.set_title('Batch Size Analysis')
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5: Merge Time Analysis
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(num_events, merge_times, 'o-')
    ax5.set_xscale('log')
    ax5.set_xlabel('Number of Events')
    ax5.set_ylabel('Time (s)')
    ax5.set_title('Batch Merge Time')
    ax5.grid(True)
    
    plt.tight_layout()
    plt.savefig('event_queue_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Test different configurations
    configurations = [
        {"num_events": 1000, "time_range": 1000, "batch_size": 100},
        {"num_events": 5000, "time_range": 5000, "batch_size": 500},
        {"num_events": 10000, "time_range": 10000, "batch_size": 1000}
    ]
    
    print("Running Event Queue Benchmark")
    print("=" * 80)
    
    results_list = []
    for config in configurations:
        print(f"\nConfiguration: {config}")
        print("-" * 40)
        
        results = run_benchmark(**config)
        results_list.append(results)
        
        print(f"Number of events: {results['num_events']}")
        print(f"Time range: {results['time_range']}")
        print(f"Batch size: {results['batch_size']}")
        print("\nTiming Results:")
        print(f"Original Schedule Time: {results['original_schedule_time']:.4f}s")
        print(f"Optimized Schedule Time: {results['optimized_schedule_time']:.4f}s")
        print(f"Schedule Speedup: {results['schedule_speedup']:.2f}x")
        print(f"Original Pop Time: {results['original_pop_time']:.4f}s")
        print(f"Optimized Pop Time: {results['optimized_pop_time']:.4f}s")
        print(f"Pop Speedup: {results['pop_speedup']:.2f}x")
        
        print("\nEvent Queue Statistics:")
        stats = results['event_stats']
        print(f"Total Events: {stats['total_events']}")
        print(f"Batch Count: {stats['batch_count']}")
        print(f"Average Batch Size: {stats['avg_batch_size']:.2f}")
        print(f"Batch Merge Time: {stats['batch_merge_time']:.4f}s")
        print(f"Current Queue Size: {stats['current_queue_size']}")
        print(f"Current Batch Size: {stats['current_batch_size']}")
    
    # Generate visualizations
    plot_performance_comparison(results_list)
    print("\nVisualization saved as 'event_queue_performance.png'")
    print("Detailed profiling results saved in profile_*.txt files")

if __name__ == "__main__":
    main() 