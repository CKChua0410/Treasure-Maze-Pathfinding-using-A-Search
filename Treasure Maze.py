# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:00:48 2025

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import heapq

Type = {
    (0,0): "empty",(0,1): "empty",(0,2): "empty",(0,3): "empty",(0,4): "Reward 1",
    (0,5): "empty",(0,6): "empty",(0,7): "empty",(0,8): "empty",(0,9): "empty",

    (1,0): "empty",(1,1): "Trap 2",(1,2): "empty",(1,3): "Trap 4",(1,4): "Treasure",
    (1,5): "empty",(1,6): "Trap 3",(1,7): "empty",(1,8): "Obstacle",(1,9): "empty",

    (2,0): "empty",(2,1): "empty",(2,2): "Obstacle",(2,3): "empty",(2,4): "Obstacle",
    (2,5): "empty",(2,6): "empty",(2,7): "Reward 2",(2,8): "Trap 1",(2,9): "empty",

    (3,0): "Obstacle",(3,1): "Reward 1",(3,2): "empty",(3,3): "Obstacle",(3,4): "empty",
    (3,5): "Trap 3",(3,6): "Obstacle",(3,7): "Treasure",(3,8): "empty",(3,9): "Treasure",

    (4,0): "empty",(4,1): "empty",(4,2): "Trap 2",(4,3): "Treasure",(4,4): "Obstacle",
    (4,5): "empty",(4,6): "Obstacle",(4,7): "Obstacle",(4,8): "empty",(4,9): "empty",

    (5,0): "empty",(5,1): "empty",(5,2): "empty",(5,3): "empty",(5,4): "empty",
    (5,5): "Reward 2",(5,6): "empty",(5,7): "empty",(5,8): "empty",(5,9): "empty"
}

hexColors = {
    "empty": "white",
    "Obstacle": "gray",
    "Reward 1": "mediumturquoise",
    "Reward 2": "mediumturquoise",
    "Treasure": "orange",
    "Trap 1": "violet",
    "Trap 2": "violet",
    "Trap 3": "violet",
    "Trap 4": "violet"
}

hexSymbols = {
    "Treasure": "",
    "Trap 1": "⊖",
    "Trap 2": "⊕",
    "Trap 3": "⊗",
    "Trap 4": "⊘",
    "Reward 1": "⊞",
    "Reward 2": "⊠",
}

# Player and Game State Setup
pState = {
    "position": (0, 0),
    "energyGrowth": 1.0,        # Multiplies after each move, affected by Trap 1 / Reward 1
    "stepMultiplier": 1.0,      # Trap 2 & Reward 2 modify this
    "collectedTreasures": set(),
        
    "R1Counter": 0,  # counter for Reward 1 (energyGrowth)
    "R2Counter": 0   # counter for Reward 2 (stepMultiplier)
}

gState = {
    "treasures": {pos for pos, cell in Type.items() if cell == "Treasure"}
}

visited = [pState["position"]]


totalStepsTaken = 0
totalEnergyUsed = 0.0


# Trap and Reward Activation
def trapOrReward(position, pState, gState, direction=None):
    cellType = Type.get(position, "empty")

    if cellType == "Trap 1":
        pState['energyGrowth'] = 2
        print("Trap 1 activated: Gravity increased! Each next step will cost DOUBLE the previous step.")
        
    elif cellType == "Trap 2":
        pState['stepMultiplier'] = 2
        print("Trap 2 activated: Speed decreased! You will need DOUBLE the steps to move.")

    elif cellType == "Trap 3":
        if direction:
            newPos = move2Cells(position, direction)
            print(f"Trap 3 activated: Forced movement to {newPos}")
            return newPos
        print("Trap 3 activated but direction unknown. No forced move applied.")

    elif cellType == "Trap 4":
        print("Trap 4 activated: All UNCOLLECTED treasures removed!")
        gState['treasures'].clear()

    elif cellType == "Reward 1":
        pState['energyGrowth'] = 0.5
        pState['R1Counter'] = 2
        print("Reward 1 activated: Gravity decreased! Energy growth is now halved.")

    elif cellType == "Reward 2":
        pState['stepMultiplier'] = 0.5
        pState['R2Counter'] = 2
        print("Reward 2 activated: Speed increased! Moving requires HALF the steps now.")

    elif cellType == "Treasure":
        if position not in pState['collectedTreasures']:
            pState['collectedTreasures'].add(position)
            gState['treasures'].discard(position)
            print(f"Treasure collected at {position}. Remaining treasures: {len(gState['treasures'])}")

    return position

def move2Cells(position, direction):
    r, q = position
    dr, dq = direction
    newPos = (r - 2*dr, q - 2*dq)
    return newPos if newPos in Type and Type[newPos] != "Obstacle" else position

# Visualization
def displayMap():
    fig, ax = plt.subplots(figsize=(12, 8))
    hexSize = 1
    dx = np.sqrt(3) * hexSize
    dy = np.sqrt(3) * hexSize

    for (r, q), typ in Type.items():
        x = q * dx
        y = -r * dy
        if q % 2 == 0:
            y -= dy / 2

        hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=hexSize * 0.95,
                                         orientation=np.pi / 6,
                                         facecolor=hexColors.get(typ, "white"),
                                         edgecolor='black')
        ax.add_patch(hexagon)

        symbol = hexSymbols.get(typ, "")
        if symbol:
            ax.text(x, y, symbol, ha='center', va='center', fontsize=16)
    
    # Show player position
    pr, pq = pState['position']
    px = pq * dx
    py = -pr * dy
    if pq % 2 == 0:
        py -= dy / 2
    ax.plot(px, py, marker='o', color='red', markersize=15, label='Player')
            
    for (r, q) in visited:
        x = q * dx
        y = -r * dy
        if q % 2 == 0:
            y -= dy / 2
        hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=hexSize * 0.95,
                                         orientation=np.pi / 6,
                                         facecolor='beige', edgecolor='black')
        ax.add_patch(hexagon)


    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    # Display stats
    yStats = 1.0 * dy
    ax.text(0, yStats, 
            f"Total Steps: {totalStepsTaken}", 
            fontsize=25, ha='left', va='bottom')
    ax.text(4 * dx, yStats, 
            f"Total Energy: {totalEnergyUsed:.2f}", 
            fontsize=25, ha='left', va='bottom')
    ax.text(8 * dx, yStats, 
            f"\nRemaining Treasures: {len(gState['treasures'])}", 
            fontsize=25, ha='left', va='bottom')

    plt.show()



# Example Move
def playerMoves(newPosition, direction=None):
    if newPosition not in visited:
        visited.append(newPosition)

    pState['position'] = newPosition
    updatedPosition = trapOrReward(newPosition, pState, gState, direction)
    
    if updatedPosition != newPosition and updatedPosition not in visited:
        visited.append(updatedPosition)

    pState['position'] = updatedPosition

    print(f"Player now at {pState['position']}")
    print(f"Energy growth multiplier: {pState['energyGrowth']} | Step multiplier: {pState['stepMultiplier']}\n")

    displayMap()


def stepsTaken():
    global totalStepsTaken, totalEnergyUsed

    stepCost = pState['stepMultiplier']
    energyCost = pState['energyGrowth']
    totalStepsTaken += stepCost
    totalEnergyUsed += energyCost

    print(f"Energy used for this step: {energyCost:.2f}")

    # === Reward 1 Duration Handling ===
    if pState['R1Counter'] > 0:
        pState['R1Counter'] -= 1
        if pState['R1Counter'] == 0:
            pState['energyGrowth'] = 1.0
            print("Reward 1 effect expired: energy growth reset to 1.0")
            

    # === Reward 2 Duration Handling ===
    if pState['R2Counter'] > 0:
        pState['R2Counter'] -= 1
        if pState['R2Counter'] == 0:
            pState['stepMultiplier'] = 1.0
            print("Reward 2 effect expired: step multiplier reset to 1.0")


    

def traps(position):
    cellType = Type.get(position, "empty")
    return cellType.startswith("Trap")

def getNeighbors(pos):
    r, q = pos
    if q % 2 == 0:  # even columns
        directions = [(0, -1), (-1, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:  # odd columns
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, 0), (0, 1)]
    neighbors = []
    for dr, dq in directions:
        nr, nq = r + dr, q + dq
        if (nr, nq) in Type:
            # Skip obstacles and traps
            if Type[(nr, nq)] != "Obstacle" and not traps((nr, nq)):
                neighbors.append((nr, nq))
    return neighbors

def moveCosts(fromPos, toPos):
    """Calculate movement cost between positions"""
    cellType = Type.get(toPos, "empty")
    
    # Different costs for different terrain types
    costMap = {
        "empty": 1.0,
        "Obstacle": float('inf'),  # Make obstacles truly impassable
        "Trap 1": 100.0,
        "Trap 2": 100.0,
        "Trap 3": 100.0,
        "Trap 4": 100.0,
        "Reward 1": 0.8,   # Slightly favorable
        "Reward 2": 0.8,
        "Treasure": 1.0
    }
    
    costBase = costMap.get(cellType, 1.0)
    return costBase * pState['stepMultiplier']

def heuristic(a, b):
    # Convert offset coords to cube coordinates
    def offset_to_cube(r, q):
        x = q
        z = r - (q - (q&1)) // 2
        y = -x - z
        return (x, y, z)

    x1, y1, z1 = offset_to_cube(*a)
    x2, y2, z2 = offset_to_cube(*b)
    return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))

def AStar(start, goal):
    openSet = []
    heapq.heappush(openSet, (0, start))
    cameFrom = {}
    gScore = {start: 0}
    fScore = {start: heuristic(start, goal)}

    while openSet:
        _, current = heapq.heappop(openSet)

        if current == goal:
            path = []
            while current in cameFrom:
                path.append(current)
                current = cameFrom[current]
            path.append(start)
            return path[::-1]

        for neighbor in getNeighbors(current):
            # Use movement cost function that penalizes traps
            cost = moveCosts(current, neighbor)
            tentative_gScore = gScore[current] + cost

            if neighbor not in gScore or tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + heuristic(neighbor, goal)
                heapq.heappush(openSet, (fScore[neighbor], neighbor))
    
    return None  # No path found

def collectAllTreasures():
    while gState['treasures']:
        start = pState['position']
        Ts = list(gState['treasures'])

        # Find the nearest treasure by heuristic
        Ts.sort(key=lambda t: heuristic(start, t))
        goal = Ts[0]

        print(f"Navigating to treasure at {goal}...")

        path = AStar(start, goal)
        if path:
            for i in range(1, len(path)):
                prev = path[i - 1]
                curr = path[i]
                direction = (curr[0] - prev[0], curr[1] - prev[1])
                stepsTaken()
                playerMoves(curr, direction)


        else:
            print(f"No path to treasure at {goal}. Skipping.")
            gState['treasures'].discard(goal)  # Remove unreachable
            
        


# =============================
# Example Usage (Test these lines manually):
# =============================

# displayMap()
# playerMoves((2,8))       # Trap 1 → increase energy growth
# stepsTaken()
# playerMoves((1,1))       # Trap 2 → double step cost
# playerMoves((1,6), (1,0))  # Trap 3 → force move in direction (1,0)
# playerMoves((1,3))       # Trap 4 → remove uncollected treasures
# playerMoves((1,4))       # Treasure → collect

displayMap()
collectAllTreasures()