import numpy as np
import matplotlib.pyplot as plt

grid_size = 8
sleeping_babies_positions = [(1,1), (3,3), (4,0), (5,4), (5,5), (6,6)] # 6 sleeping babies for 8x8 grid
crying_babies_positions = [(2,2), (2,3), (4,4), (5,6), (6,7)] # 5 crying babies for 8x8 grid
agent_position =(0,0)
milk_position = (7,7)      


grid = np.zeros((grid_size, grid_size), dtype=str)

# Fill the grid with numbers
# for i in range(grid_size):
#     for j in range(grid_size):
#         grid[i, j] = '0'

# Mark the agent position
grid[grid_size-1 - agent_position[1], agent_position[0]] = 'A'

# Mark the milk position
grid[grid_size-1 - milk_position[1], milk_position[0]] = 'M'

# Mark the crying babies positions
for pos in crying_babies_positions:
    grid[grid_size-1 - pos[1], pos[0]] = 'C'

# Mark the sleeping babies positions
for pos in sleeping_babies_positions:
    grid[grid_size-1 - pos[1], pos[0]] = 'S'

# Mark the agent trajectory
agent_steps = [(1,0),(2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(3,5),(4,5),(4,4),(4,5),(4,6),(5,6),(5,7),(6,7)]
for pos in agent_steps:
    grid[grid_size-1 - pos[1], pos[0]] = 'A'

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))


# Display the grid with color coding for C and S
for i in range(grid_size):
    for j in range(grid_size):
        if grid[i, j] == 'C':
            ax.text(j, i, grid[i, j], va='center', ha='center', color='red', fontsize=20)
        elif grid[i, j] == 'S':
            ax.text(j, i, grid[i, j], va='center', ha='center', color='green', fontsize=20)
        elif grid[i, j] == 'A':
            ax.text(j, i, grid[i, j], va='center', ha='center', color='blue', fontsize=20)
        elif grid[i, j] == 'M':
            ax.text(j, i, grid[i, j], va='center', ha='center', color='blue', fontsize=20)
        elif grid[i, j] == 'A':
            ax.text(j, i, grid[i, j], va='center', ha='center', color='black', fontsize=14)
        # else:
        #     ax.text(j, i, grid[i, j], va='center', ha='center', fontsize=14)
            
ax.set_xticks(np.arange(-0.5, grid_size, 1))
ax.set_yticks(np.arange(-0.5, grid_size, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

plt.gca().invert_yaxis()
plt.show()
