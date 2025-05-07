# Run this script to determine your multicore performance.
# It will automatically select an ideal core count. If it causes too much lag, subtract 2 from its result and use that

import sys
import time
import multiprocessing as mp
from multiprocessing import Process, Manager

from Simulation.simulation_environment import SimulationEnvironment


def run_simulation(process_id, steps, result_queue, render=False):
    """Run a simulation environment for a specified number of processes."""
    # Create a dedicated environment for this process
    sim_env = SimulationEnvironment(render=render)
    step_count = 0

    start_time = time.time()

    # Run the simulation for the specified number of steps
    while step_count < steps:
        # Default controls (no user input in multiprocess mode)
        throttle = 0.1  # Use a constant throttle for testing
        steer = 0.2  # No steering for straight-line testing

        # Execute simulation step
        done, _, _, _ = sim_env.step([throttle, steer])
        step_count += 1

        # Reset if episode is done
        if done:
            sim_env.reset_environment()

    end_time = time.time()

    # Calculate metrics
    elapsed_time = end_time - start_time
    fps = steps / elapsed_time if elapsed_time > 0 else 0

    # Put results in the queue for the main process to collect
    result_queue.put({
        'process_id': process_id,
        'steps': steps,
        'elapsed_time': elapsed_time,
        'fps': fps,
    })


def main():
    # Configuration
    steps_per_process = 20000
    render = False  # No rendering for performance testing

    # Run tests for 1 to 20 processes
    all_results = []

    for num_processes in range(1, 21):
        print(f"\nRunning test with {num_processes} process(es)...")

        # Create a queue for results
        manager = Manager()
        result_queue = manager.Queue()

        # Create and start processes
        processes = []
        for i in range(num_processes):
            p = Process(
                target=run_simulation,
                args=(i, steps_per_process, result_queue, render)
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        combined_fps = sum(r['fps'] for r in results)

        all_results.append({
            'num_processes': num_processes,
            'combined_fps': combined_fps
        })

        print(f"Processes: {num_processes}")
        print(f"Combined FPS: {combined_fps:.2f}")

    print("\n" + "=" * 40)
    print("PERFORMANCE COMPARISON")
    print("=" * 40)
    print(f"{'Processes':^10} | {'Combined FPS':^14}")
    print("-" * 40)

    for r in all_results:
        print(f"{r['num_processes']:^10} | {r['combined_fps']:^14.2f}")

    # Print the optimal number of processes based on highest combined FPS
    best_result = max(all_results, key=lambda x: x['combined_fps'])
    print("\nBest Performance:")
    print(f"Recommended Number of Processes: {best_result['num_processes']}")
    print(f"Combined FPS: {best_result['combined_fps']:.2f}")


if __name__ == "__main__":
    # Set multiprocessing start method
    if sys.platform != 'win32':
        mp.set_start_method('fork')
    else:
        mp.set_start_method('spawn')
    main()
