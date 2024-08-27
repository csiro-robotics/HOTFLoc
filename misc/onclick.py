import matplotlib.pyplot as plt
import numpy as np

# Lists to store the coordinates
x_coords = []
y_coords = []

# Define a function to handle mouse clicks
def onclick(event):
    if event.inaxes:  # Check if the click is within the axes
        x, y = event.xdata, event.ydata
        # Normalize coordinates between 0 and 1
        x_norm = (x - event.inaxes.get_xlim()[0]) / (event.inaxes.get_xlim()[1] - event.inaxes.get_xlim()[0])
        y_norm = (y - event.inaxes.get_ylim()[0]) / (event.inaxes.get_ylim()[1] - event.inaxes.get_ylim()[0])
        print(f"Click at normalized coordinates: x={x_norm:.2f}, y={y_norm:.2f}")

        # Plot a dot at the clicked location
        ax.plot(x, y, 'ro')  # 'ro' means red color and circle marker
        
        # Store the coordinates
        x_coords.append(x)
        y_coords.append(y)
        
        fig.canvas.draw()    # Update the plot with the new dot

def on_close(event):
    # Round coordinates to two decimal places for printing
    x_coords_rounded = [f"{coord:.2f}" for coord in x_coords]
    y_coords_rounded = [f"{coord:.2f}" for coord in y_coords]
    
    # Print the list of all point coordinates in the desired format
    print("Figure closed. Coordinates of all points:")
    print("[[{}], [{}]]".format(", ".join(x_coords_rounded), ", ".join(y_coords_rounded)))

# Create a figure and axis
fig, ax = plt.subplots()

# Set the limits for the x and y axes
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid()
major_ticks = np.linspace(0, 1, 9)
ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)

# Connect the click event to the onclick function
cid = fig.canvas.mpl_connect('button_press_event', onclick)
# Connect the close event to the on_close function
cid_close = fig.canvas.mpl_connect('close_event', on_close)

# Display the plot
plt.title('Click anywhere in the figure')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
