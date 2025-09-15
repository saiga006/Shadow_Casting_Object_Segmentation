#!/usr/bin/env python3
"""
GPU Memory Monitor

A simple script to monitor GPU memory usage during training.
Can be run alongside training to track memory consumption.
"""

import time
import subprocess
import sys
import argparse

def get_gpu_memory_nvidia():
    """Get GPU memory info using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True
        )
        
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used_mb': int(parts[2]),
                    'memory_total_mb': int(parts[3]),
                    'utilization_percent': int(parts[4])
                })
        
        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []

def get_gpu_memory_tensorflow():
    """Get GPU memory info using TensorFlow."""
    try:
        import tensorflow as tf
        gpu_info = []
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        for i, gpu in enumerate(gpus):
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                gpu_info.append({
                    'index': i,
                    'name': gpu.name,
                    'memory_used_mb': memory_info['current'] // (1024*1024),
                    'memory_total_mb': 'N/A',
                    'utilization_percent': 'N/A'
                })
            except:
                gpu_info.append({
                    'index': i,
                    'name': gpu.name,
                    'memory_used_mb': 'N/A',
                    'memory_total_mb': 'N/A',
                    'utilization_percent': 'N/A'
                })
        
        return gpu_info
    except ImportError:
        return []

def monitor_gpu(interval=5, use_tensorflow=False, csv_output=None):
    """Monitor GPU memory usage continuously."""
    if csv_output:
        print(f"GPU Memory Monitor - Logging to {csv_output}")
        print("=" * 50)
        print("Press Ctrl+C to stop monitoring")
        print()
        
        # Write CSV header
        with open(csv_output, 'w') as f:
            f.write("timestamp,gpu_id,memory_used_mb,memory_total_mb,utilization_percent\n")
    else:
        print("GPU Memory Monitor")
        print("=" * 50)
        print("Press Ctrl+C to stop monitoring")
        print()
    
    try:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            if use_tensorflow:
                gpu_info = get_gpu_memory_tensorflow()
                method = "TensorFlow"
            else:
                gpu_info = get_gpu_memory_nvidia()
                method = "nvidia-smi"
            
            if gpu_info:
                if not csv_output:
                    print(f"[{timestamp}] GPU Memory Status ({method}):")
                
                for info in gpu_info:
                    if csv_output:
                        # Write to CSV file
                        with open(csv_output, 'a') as f:
                            f.write(f"{timestamp},{info['index']},{info['memory_used_mb']},{info['memory_total_mb']},{info['utilization_percent']}\n")
                    else:
                        print(f"  GPU {info['index']} ({info['name']}): "
                              f"Used: {info['memory_used_mb']}MB, "
                              f"Total: {info['memory_total_mb']}MB, "
                              f"Util: {info['utilization_percent']}%")
            else:
                if csv_output:
                    with open(csv_output, 'a') as f:
                        f.write(f"{timestamp},0,N/A,N/A,N/A\n")
                else:
                    print(f"[{timestamp}] No GPU information available")
            
            if not csv_output:
                print()
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU memory usage')
    parser.add_argument('--interval', '-i', type=int, default=5,
                        help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--tensorflow', '-tf', action='store_true',
                        help='Use TensorFlow for memory monitoring instead of nvidia-smi')
    parser.add_argument('--csv', '-c', type=str,
                        help='Output to CSV file instead of console')
    
    args = parser.parse_args()
    
    monitor_gpu(interval=args.interval, use_tensorflow=args.tensorflow, csv_output=args.csv)

if __name__ == '__main__':
    main()
