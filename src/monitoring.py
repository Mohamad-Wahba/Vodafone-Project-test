"""
Application Monitoring Module

This module provides functionality to monitor resource usage of the current application.

Functions:
    get_app_metrics(): Returns a dictionary containing current application metrics.

Usage:
    from monitoring import get_app_metrics
    metrics = get_app_metrics()
    print(f"App CPU Usage: {metrics['cpu_usage']}%")

Authors: Mohamed Bassiony, Mohamad Wahba, Beshoy Ashraf Samir, Mohamad Sharqawi
Date: September 24, 2024
"""

import psutil
import os
import time

last_cpu_time = 0
last_cpu_measure_time = time.time()

def get_app_metrics():
    """
    Retrieve current application metrics.

    Returns:
        dict: A dictionary containing the following keys:
            - 'cpu_usage': Current CPU usage of the app as a percentage
            - 'memory_usage': Current memory usage of the app in MB
            - 'uptime': Application uptime in seconds
    """
    global last_cpu_time, last_cpu_measure_time

    current_process = psutil.Process(os.getpid())
    
    # Calculate CPU usage
    current_time = time.time()
    current_cpu_time = sum(current_process.cpu_times()[:2])
    cpu_usage = (current_cpu_time - last_cpu_time) / (current_time - last_cpu_measure_time) * 100
    last_cpu_time = current_cpu_time
    last_cpu_measure_time = current_time

    # Get memory usage
    memory_usage = current_process.memory_info().rss / (1024 * 1024)  # Convert to MB

    # Get uptime
    uptime = time.time() - current_process.create_time()

    return {
        "cpu_usage": round(cpu_usage, 2),
        "memory_usage": round(memory_usage, 2),
        "uptime": round(uptime, 2)
    }
