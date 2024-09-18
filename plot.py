import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button

EARTH_RADIUS = 6.378 * 10**6

def animate_trajectories(all_trajectories, step_ms, d_s):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create sphere (Earth)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = EARTH_RADIUS * np.cos(u) * np.sin(v)
    y = EARTH_RADIUS * np.sin(u) * np.sin(v)
    z = EARTH_RADIUS * np.cos(v)
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Prepare missile lines
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # List of colors
    missile_lines = [ax.plot([], [], [], f'{colors[i % len(colors)]}-')[0] for i in range(len(all_trajectories))]

    # Set initial view
    ax.set_xlim([-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5])
    ax.set_ylim([-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5])
    ax.set_zlim([-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Missile Trajectories')

    # Animation control
    paused = False
    current_frame = [0]

    def update(frame):
        if not paused:
            current_frame[0] = frame
            for i, trajectory in enumerate(all_trajectories):
                if frame < len(trajectory):
                    x_data, y_data, z_data = zip(*trajectory[:frame+1])
                    missile_lines[i].set_data(x_data, y_data)
                    missile_lines[i].set_3d_properties(z_data)
        return missile_lines

    def on_click(event):
        nonlocal paused
        paused = not paused

    def on_press(event):
        if event.key == 'right' and current_frame[0] < len(all_trajectories[0]) - 1:
            current_frame[0] += 1
        elif event.key == 'left' and current_frame[0] > 0:
            current_frame[0] -= 1
        update(current_frame[0])
        fig.canvas.draw()

    # Add pause button
    pause_ax = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    pause_button = Button(pause_ax, 'Pause/Play')
    pause_button.on_clicked(on_click)

    fig.canvas.mpl_connect('key_press_event', on_press)

    anim = animation.FuncAnimation(fig, update, frames=len(all_trajectories[0]), 
                                   interval=step_ms/5000, blit=False, repeat=False)

    plt.show()
