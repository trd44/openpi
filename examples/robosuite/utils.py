# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
import numpy as np
import math
import subprocess
import psutil
import time
import logging

# ------------------------------------------------------------------------------
# System monitoring functions
# ------------------------------------------------------------------------------
def get_gpu_power_usage():
    """Get GPU power usage in watts."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            power_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(power_values) if power_values else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_gpu_utilization():
    """Get GPU utilization percentage."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            util_values = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
            return sum(util_values) / len(util_values) if util_values else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_gpu_memory_usage():
    """Get GPU memory usage percentage."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            total_used = 0
            total_available = 0
            for line in lines:
                if ',' in line:
                    used, total = line.split(',')
                    total_used += float(used.strip())
                    total_available += float(total.strip())
            return (total_used / total_available * 100) if total_available > 0 else 0.0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    return 0.0

def get_cpu_power_usage():
    """Get CPU power usage estimation based on frequency and utilization."""
    try:
        # Get CPU frequency and utilization
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        if cpu_freq and cpu_freq.current > 0:
            # Rough estimation: higher frequency and utilization = higher power
            # This is a simplified model - actual power depends on many factors
            base_power = 15.0  # Base power in watts
            freq_factor = (cpu_freq.current / cpu_freq.max) if cpu_freq.max > 0 else 1.0
            util_factor = cpu_percent / 100.0
            
            estimated_power = base_power * freq_factor * (0.5 + 0.5 * util_factor)
            return estimated_power
    except Exception:
        pass
    return 0.0

def get_system_metrics():
    """Get comprehensive system metrics with timestamps."""
    try:
        # Get current timestamp
        current_time = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # GPU metrics
        gpu_power = get_gpu_power_usage()
        gpu_util = get_gpu_utilization()
        gpu_memory = get_gpu_memory_usage()
        
        # CPU power estimation
        cpu_power = get_cpu_power_usage()
        
        return {
            'timestamp': current_time,
            'timestamp_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
            'cpu_percent': cpu_percent,
            'cpu_freq_mhz': cpu_freq.current if cpu_freq else 0,
            'cpu_count': cpu_count,
            'cpu_power_watts': cpu_power,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'disk_percent': disk_percent,
            'disk_used_gb': disk_used_gb,
            'disk_total_gb': disk_total_gb,
            'gpu_power_watts': gpu_power,
            'gpu_utilization_percent': gpu_util,
            'gpu_memory_percent': gpu_memory,
        }
    except Exception as e:
        logging.warning(f"Error getting system metrics: {e}")
        return {}

def calculate_energy_consumption(power_data_points):
    """
    Calculate total energy consumption from power data points with timestamps.
    
    Args:
        power_data_points: List of dicts with 'timestamp' and power values
        
    Returns:
        Dict with total energy consumption in watt-hours and joules
    """
    if len(power_data_points) < 2:
        return {'total_energy_wh': 0, 'total_energy_joules': 0, 'duration_seconds': 0}
    
    # Sort by timestamp
    sorted_points = sorted(power_data_points, key=lambda x: x['timestamp'])
    
    total_gpu_energy = 0
    total_cpu_energy = 0
    total_duration = 0
    
    for i in range(1, len(sorted_points)):
        prev = sorted_points[i-1]
        curr = sorted_points[i]
        
        # Time interval in hours
        time_interval_hours = (curr['timestamp'] - prev['timestamp']) / 3600.0
        time_interval_seconds = curr['timestamp'] - prev['timestamp']
        
        # Average power during this interval
        avg_gpu_power = (prev.get('gpu_power_watts', 0) + curr.get('gpu_power_watts', 0)) / 2
        avg_cpu_power = (prev.get('cpu_power_watts', 0) + curr.get('cpu_power_watts', 0)) / 2
        
        # Energy = Power × Time
        gpu_energy = avg_gpu_power * time_interval_hours
        cpu_energy = avg_cpu_power * time_interval_hours
        
        total_gpu_energy += gpu_energy
        total_cpu_energy += cpu_energy
        total_duration += time_interval_seconds
    
    total_energy_wh = total_gpu_energy + total_cpu_energy
    total_energy_joules = total_energy_wh * 3600  # Convert watt-hours to joules
    
    return {
        'total_energy_wh': total_energy_wh,
        'total_energy_joules': total_energy_joules,
        'gpu_energy_wh': total_gpu_energy,
        'cpu_energy_wh': total_cpu_energy,
        'duration_seconds': total_duration,
        'duration_hours': total_duration / 3600
    }

def generate_video_filename(args, episode: int) -> str:
    """Generate video filename based on current arguments and episode number
    """
    env = args.env
    date = time.strftime('%Y%m%d')
    timestamp = time.strftime('%H%M%S')
    return f"{env}/{date}/{env}_{timestamp}_seed{args.seed}_ep{episode+1}.mp4"
