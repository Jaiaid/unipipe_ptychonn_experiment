import pandas as pd
import matplotlib.pyplot as plt
import sys

# Global variables for range selection
TIMESTAMP_START = 1732836094252950000  # Start range for timestamp
TIMESTAMP_END = 1732836094270548800    # End range for timestamp

def plot_from_log(file_path):
    # Read the file into a pandas DataFrame
    try:
        data = pd.read_csv(file_path, sep=',')
        print(data.columns)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    # Rename columns for easier access (optional, based on your log format)
    data.columns = [
        "device_id", "timestamp", "txpci", "rxpci", "gpu_util", 
        "mem_util", "mem_used", "mem_free", "cpu_usage", 
        "cpu_vmem", "cpu_vmem_avail", "cpu_vmem_total", "proc_mem"
    ]

    # Filter data based on the global timestamp range
    filtered_data = data[(data['timestamp'] >= TIMESTAMP_START) & (data['timestamp'] <= TIMESTAMP_END)]
    print(filtered_data)

    # Check if there's data to plot
    if filtered_data.empty:
        print("No data available in the specified timestamp range.")
        return

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['timestamp'], filtered_data['gpu_util'], label='GPU Utilization (%)', color='b', marker='x')
    plt.plot(filtered_data['timestamp'], filtered_data['gpu_util'], label='GPU Utilization (%)', color='b', marker='x')
    plt.plot(filtered_data['timestamp'], filtered_data['gpu_util'], label='GPU Utilization (%)', color='b', marker='x')
    plt.xlabel('Timestamp')
    plt.ylabel('GPU Utilization')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot or show it
    plt.savefig("plot.png")  # Save the plot as an image file
    plt.show()               # Display the plot

if __name__ == "__main__":
    # Check if the script is run with a file argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <log_file_path>")
    else:
        log_file_path = sys.argv[1]
        plot_from_log(log_file_path)
