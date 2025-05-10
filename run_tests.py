import os
import multiprocessing as mp
from pyinstrument import Profiler
import time
from datetime import datetime

# Disable Pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Import the benchmark function from the correct location
from scripts.train_test import run_benchmark

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"profile_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Store benchmark results
    results = []

    # Run benchmarks with different numbers of environments (even numbers from 2 to 14)
    for num_envs in range(4, 50, 4):
        print(f"\n{'=' * 50}")
        print(f"Running benchmark with {num_envs} environments")
        print(f"{'=' * 50}\n")

        # Output HTML filename for this run
        html_file = os.path.join(output_dir, f"benchmark_profile_{num_envs}_envs.html")

        # Run the profiler
        p = Profiler()
        with p:
            # Run benchmark with the specified number of environments
            metrics = run_benchmark(num_envs=num_envs)

        # Save HTML report directly to the output directory
        p.write_html(html_file)  # Use write_html instead of output_html + manual write

        print(f"Profile for {num_envs} envs saved to: {html_file}")

        # Store results
        results.append(metrics)

    # Save a summary file in the same directory
    summary_file = os.path.join(output_dir, "benchmark_summary.txt")
    with open(summary_file, "w") as f:
        f.write("BENCHMARK SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Environments':<12}{'FPS':<10}{'MSPT (ms)':<12}{'MSPE (ms)':<12}{'Steps':<10}{'Total Env Steps':<16}{'Env Time %':<12}{'Inference %':<12}{'Train %':<12}\n")
        f.write("-" * 100 + "\n")

        for result in results:
            # Calculate percentages
            total_compute_time = result['env_time'] + result['inference_time'] + result['train_time']
            env_pct = (result['env_time'] / total_compute_time * 100) if total_compute_time > 0 else 0
            inference_pct = (result['inference_time'] / total_compute_time * 100) if total_compute_time > 0 else 0
            train_pct = (result['train_time'] / total_compute_time * 100) if total_compute_time > 0 else 0

            f.write(f"{result['num_envs']:<12}{result['fps']:<10.2f}{result['mspt']:<12.2f}{result.get('mspe', 0):<12.2f}{result['steps']:<10}{result.get('total_env_steps', 0):<16}{env_pct:<12.1f}{inference_pct:<12.1f}{train_pct:<12.1f}\n")

    # Print summary to console
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Environments':<12}{'FPS':<10}{'MSPT (ms)':<12}{'MSPE (ms)':<12}{'Steps':<10}{'Total Env Steps':<16}{'Env Time %':<12}{'Inference %':<12}{'Train %':<12}")
    print("-" * 100)

    for result in results:
        # Calculate percentages
        total_compute_time = result['env_time'] + result['inference_time'] + result['train_time']
        env_pct = (result['env_time'] / total_compute_time * 100) if total_compute_time > 0 else 0
        inference_pct = (result['inference_time'] / total_compute_time * 100) if total_compute_time > 0 else 0
        train_pct = (result['train_time'] / total_compute_time * 100) if total_compute_time > 0 else 0

        print(f"{result['num_envs']:<12}{result['fps']:<10.2f}{result['mspt']:<12.2f}{result.get('mspe', 0):<12.2f}{result['steps']:<10}{result.get('total_env_steps', 0):<16}{env_pct:<12.1f}{inference_pct:<12.1f}{train_pct:<12.1f}")

    print(f"\nAll results saved to: {output_dir}")
