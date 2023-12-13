import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog

# Load data using pandas
file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
pings = pd.read_csv(file_path, sep="\t", header=None, names=["time"])

# Convert data to PyTorch tensor
time_tensor = torch.tensor(pings["time"].values, dtype=torch.float32)


min_time = torch.min(time_tensor).item()
mean_time = torch.mean(time_tensor).item()
std_dev_time = torch.std(time_tensor).item()
max_time = torch.max(time_tensor).item()


plt.figure(figsize=(14, 8))
sns.lineplot(data=pings, x=pings.index, y="time")
plt.xlabel("Index")
plt.ylabel("Time (ms)")
plt.title("Ping Time Series")
plt.savefig("pings_seaborn.png")
plt.show()

root = Tk()
root.title("Ping Data Analysis")


stats_label = Label(root, text=f"Min: {min_time}\nMean: {mean_time}\nStd Dev: {std_dev_time}\nMax: {max_time}")
stats_label.pack()


canvas = FigureCanvasTkAgg(plt.figure(), master=root)
canvas.get_tk_widget().pack()

root.mainloop()
