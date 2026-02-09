# this file is to simply visualize the time series sample(s).
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument("--file_path", type=str, default="syn_data/har/00000.npy", help="which file to read")
parser.add_argument("--number", type=int, default=-1, help="which sample to visualize. -1 for all samples")
parser.add_argument("--channel", type=int, default=0, help="which channel to visualize")

args = parser.parse_args()

channel = 0

data = np.load(args.file_path)
if args.number >= 0:
    number = args.number
    sample = data[number][channel]
    y = sample
    fig = plt.figure()
    plt.plot(y)
else:
    mean_values = np.mean(data[:, channel, :], axis=0)  
    std_values = np.std(data[:, channel, :], axis=0)   
    timestamps = np.arange(data.shape[-1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timestamps, mean_values, label='Mean', color='blue', linewidth=2)
    ax.fill_between(timestamps, 
                    mean_values - std_values, 
                    mean_values + std_values, 
                    color='blue', alpha=0.3, label='Standard Deviation')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Mean and Standard Deviation of Channel {channel}')
    ax.legend()

# save fig
fig.savefig('picture.png')