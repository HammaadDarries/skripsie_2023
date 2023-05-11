import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Read position data
with open('position_history.csv', 'r') as position_file:
    position_reader = csv.reader(position_file)
    next(position_reader)  # skip header
    position_data = [tuple(map(float, row)) for row in position_reader]

# Read speed data
with open('speed_history.csv', 'r') as speed_file:
    speed_reader = csv.reader(speed_file)
    next(speed_reader)  # skip header
    speed_data = [float(row[0]) for row in speed_reader]

# Normalize speed data to [0, 1]
speed_data = np.array(speed_data)
normalized_speed = (speed_data - np.min(speed_data)) / (np.max(speed_data) - np.min(speed_data))

# Create a colormap for speed values
cmap = cm.get_cmap('RdYlGn_r')

# Plot trajectory
fig, ax = plt.subplots()

for i in range(1, len(position_data)):
    x1, y1 = position_data[i - 1]
    x2, y2 = position_data[i]
    color = cmap(normalized_speed[i])
    
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)

ax.set_title("Car Trajectory Coloured by Speed")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# Add colorbar
norm = mcolors.Normalize(vmin=np.min(speed_data), vmax=np.max(speed_data))
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Speed')

plt.show()