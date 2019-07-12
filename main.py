import networkx as nx
import numpy as np
import os

from collections import namedtuple

from itertools import chain

from matplotlib import colors
from matplotlib import pyplot as plt

from numpy.random import rand, randint, shuffle

from scipy.spatial import Delaunay, Voronoi

from perlin_noise import PerlinNoise

VERSION = "v0.1.0"

flip_funcs = dict(central=lambda m: np.flip(m),
                  diagonal=lambda m: np.transpose(m),
                  horizontal=lambda m: np.flip(m, axis=1))

Neighborhood = namedtuple("Neighborhood", "north west south east H V")

def neighborhood(i, j):
    return np.array([[i,   j+1],
                     [i-1, j+1],
                     [i-1, j],
                     [i-1, j-1],
                     [i,   j-1],
                     [i+1, j-1],
                     [i+1, j],
                     [i+1, j+1]])

def to_str(array, sep=","):
    return sep.join([str(a) for a in array])

class MapGenerationFailure(Exception):
    pass

class Map:
    starting_mills = None

    def __init__(self, size=45, padded_size=79,
                 symmetry="central",
                 min_starting_dist=15,
                 seed=None):

        if seed is None:
            seed = randint(2**31)
        
        self.seed = seed
        np.random.seed(seed)

        self.to_quadrant_funcs = dict(central=self.to_upper_triangle,
                                      diagonal=self.to_upper_triangle,
                                      horizontal=self.to_upper_half)
        
        self.reflect_point_funcs = dict(central=self.reflect_central)

        self.to_quadrant = self.to_quadrant_funcs[symmetry]
        self.reflect_point = self.reflect_point_funcs[symmetry]

        self.symmetry = symmetry
        self.symmetric_array = flip_funcs[symmetry]
                                 
        self.size = size
        self.padded_size = padded_size

        pad_before = (padded_size - size)//2
        pad_after = padded_size - size - pad_before

        instart = pad_before
        instop = pad_before + size

        self.instart = instart
        self.instop = instop
        self.inslice = slice(instart, instop)
        self.padded_tiles = np.zeros((self.padded_size, self.padded_size), dtype=int)

        self.min_starting_dist = min_starting_dist

        self.ownerships = -2*np.ones((self.padded_size, self.padded_size), dtype=int)

        pnoise = PerlinNoise(scale=20, octaves=2)  # Take info from constructor

        elevations = pnoise.noise_grid(range(size), range(size)) + 0.5
        elevations += self.symmetric_array(elevations)

        elevations = np.array(np.round(3/2*elevations), dtype=int)
        
        self.padded_elevations = np.zeros((self.padded_size, self.padded_size), dtype=int)
        self.padded_elevations[self.inslice, self.inslice] = elevations
        
        for k in range(pad_before):
            self.padded_elevations[k, self.inslice] = elevations[0, :]
            self.padded_elevations[self.inslice, k] = elevations[:, 0]

        for k in range(pad_after):
            self.padded_elevations[-(k+1), self.inslice] = elevations[-1, :]
            self.padded_elevations[self.inslice, -(k+1)] = elevations[:, -1]

        self.padded_elevations[:instart, :instart] = elevations[0, 0]*np.ones((pad_before, pad_before))
        self.padded_elevations[:instart, instop:] = elevations[0, -1]*np.ones((pad_before, pad_after))
        self.padded_elevations[instop:, :instart] = elevations[-1, 0]*np.ones((pad_after, pad_before))
        self.padded_elevations[instop:, instop:] = elevations[-1, -1]*np.ones((pad_after, pad_after))

        water = self.padded_elevations == 0

        self.terraform()

        self.waypoint_mask = np.ones_like(self.padded_tiles, dtype=bool)
        self.waypoint_mask[self.inslice, self.inslice] = False

        self.add_blockers()
        self.add_waypoints()

        self.mills = []
        self.add_mills()
        self.remove_head_to_head()

        water = np.logical_and(self.padded_elevations < 3, water)
        water = np.logical_and(np.logical_not(self.waypoint_mask), water)
        self.padded_tiles[water] = tiles["water"].index
        self.padded_elevations[water] = 2
        self.padded_elevations -= 2

        cliffs = self.padded_elevations % 2 == 1
        self.padded_tiles[cliffs] = tiles["cliff"].index

        self.add_roads()
        self.close_map()


    @property
    def elevations(self):
        return self.padded_elevations[self.inslice, self.inslice]

    @property
    def local_waypoints(self):
        return np.array(self.waypoints) - self.instart

    @property
    def tiles(self):
        return self.padded_tiles[self.inslice, self.inslice]
    
    def add_blockers(self):
        # TODO make noise scale and f constructor parameters/random
        pnoise = PerlinNoise(scale=30, octaves=1)
        f = 0.8  # Blockers amount

        density = pnoise.noise_grid(range(self.padded_size), range(self.padded_size)) + 0.5
        density = f*(density + self.symmetric_array(density))/2
        rs = rand(self.padded_size, self.padded_size)
        rs = (rs + self.symmetric_array(rs))/2

        # plt.matshow(density)
        # plt.colorbar()

        self.padded_tiles[np.logical_and(rs < density, self.padded_tiles == 0)] = tiles["blocker"].index
    
    def add_mill(self, x, y, layout, start=False):
        self.mills.append((x, y))
        self.mills.append(self.reflect_point(x, y))

        flips = randint(2, size=3)

        if flips[0] == 1:
            layout = np.flip(layout, axis=0)
        
        if flips[1] == 1:
            layout = np.flip(layout, axis=1)

        if flips[2] == 1:
            layout = np.transpose(layout)

        Lx, Ly = layout.shape
        el = np.max(self.padded_elevations[x-3:x-3+2*Lx, y-3:y-3+2*Ly])

        if el % 2 == 1:
            el += 1

        for k1 in range(Lx):
            i = x + 2*k1 - 3
            for k2 in range(Ly):
                j = y + 2*k2 - 3
                tile = layout[k1, k2]

                for di in range(2):
                    ii = i + di
                    for dj in range(2):
                        jj = j + dj

                        self.padded_tiles[ii, jj] = tile.index
                        self.terraform_tile(ii, jj, el)

                        rii, rjj = self.reflect_point(ii, jj)
                        self.padded_tiles[rii, rjj] = tile.index
                        self.terraform_tile(rii, rjj, el)

                        if start:
                            self.ownerships[ii, jj] = 0
                            self.ownerships[rii, rjj] = 1

    
    def add_mills(self):
        starting_mill_placed = False
        for i, j in self.waypoints:
            if (self.distance_to_map_edge(i, j) >= 5
                and self.distance_to_reflection(i, j) >= self.min_starting_dist):

                starting_mill_placed = True
                self.starting_mills = [(i, j), self.reflect_point(i, j)]
                self.add_mill(i, j, starting_mill_layout, start=True)

                break

        if not starting_mill_placed:
            print("Unable to find starting mill")
            raise MapGenerationFailure("No position was good for starting mill.")

        wps = np.array(self.waypoints)
        shuffle(wps)
        n_mills = 0

        for i, j in wps:
            if (self.distance_to_map_edge(i, j) >= 5
                and self.distance_to_reflection(i, j) >= 10
                and self.distance_to_closest_mill(i, j) >= 10):

                n_mills += 2
                self.add_mill(i, j, mill_layout)
        
        if n_mills < 4:
            raise MapGenerationFailure(f"Only able to place {n_mills} mills.")

    def add_roads(self):
        w, rw = self.starting_mills

        road_closed = False
        connected = [w]
        rconnected = [rw]

        road = []

        while not road_closed:
            neigs = list(self.waypoint_network.neighbors(w))
            wend = neigs[randint(len(neigs))]
            rwend = self.reflect_point(*wend)

            road += self.raytrace(*w, *wend, width=4)
            road += self.raytrace(*rw, *rwend, width=4)

            if wend in rconnected:
                road_closed = True
            
            connected.append(wend)
            rconnected.append(rwend)

            w, rw = wend, rwend

        for mill in self.mills:
            if mill not in connected and mill not in rconnected:
                rmill = self.reflect_point(*mill)

                wend = connected[randint(len(connected))]
                rwend = self.reflect_point(*wend)

                road += self.raytrace(*mill, *wend, width=3)
                road += self.raytrace(*rmill, *rwend, width=3)

                connected.append(mill)
                rconnected.append(rmill)

        for i, j in road:
            current = self.padded_tiles[i, j]

            if current == tiles["blocker"].index or current == tiles["grass"].index:
                self.padded_tiles[i, j] = tiles["road"].index
            elif current == tiles["cliff"].index:
                self.padded_tiles[i, j] = tiles["ramp"].index
    
    def add_waypoints(self):
        n = randint(32, 48)
        positions = randint(0, self.padded_size, size=(n, 2))
        waypoints = list(positions)
        
        waypoints.extend([self.reflect_point(*pos) for pos in positions])
        
        waypoints = np.array(waypoints)

        vor = Voronoi(waypoints)

        f = 0.2  # Rate of regularization  # TODO pass it from constructor

        for k in range(len(waypoints)):
            region = vor.regions[vor.point_region[k]]
            point = waypoints[k]

            if len(region) > 0 and np.min(region) >= 0:
                vertices = np.array([vor.vertices[j] for j in region])
                waypoints[k] = f*np.mean(vertices, axis=0) + (1 - f)*point
        
        waypoints = np.round(waypoints)
        # Keep only the original waypoints and re-reflect them ensure symmetry
        waypoints = list(waypoints[:n])

        self.waypoints = []

        for i, j in waypoints:
            if self.instart < i < self.instop and self.instart < j < self.instop:
                self.waypoints.append([i, j])
                self.waypoints.append(self.reflect_point(i, j))

        self.waypoints = np.array(sorted(self.waypoints, key=lambda p:-self.distance_to_reflection(*p)), dtype=int)

        delaunay = Delaunay(self.waypoints)
        self.waypoint_network = nx.Graph()

        for simplex in delaunay.simplices:
            v1, v2, v3 = [tuple(v) for v in np.array(delaunay.points[simplex], dtype=int)]
            self.waypoint_network.add_edge(v1, v2)
            self.waypoint_network.add_edge(v2, v3)
            self.waypoint_network.add_edge(v3, v1)

    def close_map(self):
        el = self.padded_elevations[self.instart-1,:]
        ramps = el % 2 == 1
        self.padded_tiles[self.instart-2,:] = tiles["blocker"].index
        self.padded_tiles[self.instart-2,:][ramps] = tiles["cliff"].index
        
        el = self.padded_elevations[self.instop+1,:]
        ramps = el % 2 == 1
        self.padded_tiles[self.instop+1,:] = tiles["blocker"].index
        self.padded_tiles[self.instop+1,:][ramps] = tiles["cliff"].index
        
        el = self.padded_elevations[:,self.instart-1]
        ramps = el % 2 == 1
        self.padded_tiles[:,self.instart-2] = tiles["blocker"].index
        self.padded_tiles[:,self.instart-2][ramps] = tiles["cliff"].index
        
        el = self.padded_elevations[:,self.instop+1]
        ramps = el % 2 == 1
        self.padded_tiles[:,self.instop+1] = tiles["blocker"].index
        self.padded_tiles[:,self.instop+1][ramps] = tiles["cliff"].index

    def distance(self, i1, j1, i2, j2):
        return np.max(np.abs([i1 - i2, j1 - j2]))
    
    def distance_to_closest_mill(self, i, j):
        if len(self.mills) == 0:
            return 1000
        
        ds = [self.distance(i, j, mi, mj) for mi, mj in self.mills]

        return np.min(ds)

    def distance_to_map_edge(self, i, j):
        if not (self.instart < i < self.instop and self.instart < j < self.instop):
            return -1  # Out of the map
        
        return np.min([i - self.instart,
                       self.instop - i,
                       j - self.instart,
                       self.instop - j])

    def distance_to_reflection(self, i, j):
        ri, rj = self.reflect_point(i, j)
        return self.distance(i, j, ri, rj)

    def is_in_quadrant(self, i, j):
        qi, qj = self.to_quadrant(i, j)
        return qi == i and qj == j

    def raytrace(self, x1, y1, x2, y2, width=0):
        dx = x2 - x1
        dy = y2 - y1

        perp = np.array([-dy, dx])/np.sqrt(dx**2 + dy**2)
        v0 = np.array([x1, y1]) - width*perp/2
        v1 = np.array([x2, y2]) - width*perp/2

        ts = np.linspace(0, width, int(np.ceil(width)) + 1)
        tiles = set()

        for t in ts:
            tiles.update(self.raytrace_thin(v0 + t*perp, v1 + t*perp))
        
        return list(tiles)

    def raytrace_thin(self, v0, v1):
        # The equation of the ray is v = v0 + t*d
        d = v1 - v0
        inc = np.sign(d)  # Gives the direction in which the ray progress

        tile = np.array(np.round(v0), dtype=int)
        endtile = np.array(np.round(v1), dtype=int)

        if d[0] == 0:
            return [(tile[0], tile[1] + k*inc[1]) for k in range(np.abs(tile[0] - endtile[0]))]
        
        if d[1] == 0:
            return [(tile[0] + k*inc[0], tile[1]) for k in range(np.abs(tile[1] - endtile[1]))]

        v = v0

        tiles = [tuple(tile)]

        # Continue as long as we are not in the last tile
        while np.max(np.abs(v1 - v)) > 0.5:
            # L = (Lx, Ly) where Lx is the x coordinate of the next vertical
            # line and Ly the y coordinate of the next horizontal line
            L = tile + 0.5*inc

            # Solve the equation v + d*t == L for t, simultaneously for the next
            # horizontal line and vertical line
            t = (L - v)/d

            if t[0] < t[1]:  # The vertical line is intersected first
                tile[0] += inc[0]
                v += t[0]*d
            else:  # The horizontal line is intersected first
                tile[1] += inc[1]
                v += t[1]*d
            
            tiles.append(tuple(tile))
        
        return tiles


    def reflect_central(self, i, j):
        return self.instop + self.instart - i - 1, self.instop + self.instart - j - 1
    
    def remove_head_to_head(self):
        change_detected = True
        while change_detected:
            change_detected = self.remove_horizontal_head_to_head()

        self.padded_elevations = np.transpose(self.padded_elevations)

        change_detected = True
        while change_detected:
            change_detected = self.remove_horizontal_head_to_head()

        self.padded_elevations = np.transpose(self.padded_elevations)
    
    def remove_horizontal_head_to_head(self):
        change_detected = False

        for i in range(1, self.padded_size-2):
            for j in range(1, self.padded_size-2):
                current = self.padded_elevations[i, j]
                if current % 2 == 1:
                    west = self.padded_elevations[i, j-1]

                    if west - current == 1:
                        east = self.padded_elevations[i, j+1]
                        east2 = self.padded_elevations[i, j+2]

                        if east - current == 1:
                            change_detected = True
                            self.padded_elevations[i, j] += 1
                        elif east - current == 0 and east2 - current == 1:
                            change_detected = True
                            self.padded_elevations[i, j] += 1
                            self.padded_elevations[i, j+1] += 1

        return change_detected

    def save_map(self):
        processed = np.zeros_like(self.padded_tiles, dtype=bool)
        elevations = []
        terrains = []
        entities = []
        entity_id = 1
        start_tiles = []

        for j in range(self.padded_size):
            for i in range(self.padded_size):
                t = ts[self.padded_tiles[i, j]]
                el = self.padded_elevations[i, j]
                elevations.append(el)
                
                if t == tiles["ramp"]:
                    terrains.append(terrainkeys["grass_ramp"])
                    continue

                if t == tiles["cliff"]:
                    terrains.append(terrainkeys["grass_cliff"])
                    continue
                
                if t == tiles["water"]:
                    terrains.append(terrainkeys["water"])
                    continue
                
                if not processed[i, j]:
                    processed[i, j] = True
                    if t == tiles["blocker"]:
                        blocker = templates["autumn_rocks"].format(
                            x=i,
                            y=j,
                            h=el,
                            id=entity_id
                        )
                        entity_id += 1
                        entities.append(blocker)
                    
                    if t == tiles["farm"] or t == tiles["pig"]:
                        farm = templates["farmland"].format(
                            id=entity_id,
                            i=i,
                            j=j,
                            x=i + 0.5,
                            y=j + 0.5,
                            h=el
                        )
                        entity_id += 1
                        entities.append(farm)

                        processed[i:i+2, j:j+2] = True
                    
                    if t == tiles["pig"]:
                        pig = templates["structure_farm"].format(
                            id=entity_id,
                            i=i,
                            j=j,
                            x=i + 0.5,
                            y=j + 0.5,
                            h=el,
                            owner_id=self.ownerships[i,j]
                        )
                        entity_id += 1
                        entities.append(pig)
                    
                    if t == tiles["mill"] or t == tiles["start"]:
                        mill = templates["windmill"].format(
                            id=entity_id,
                            i=i,
                            j=j,
                            x=i + 0.5,
                            y=j + 0.5,
                            h=el
                        )
                        entity_id += 1
                        entities.append(mill)
                        
                        processed[i:i+2, j:j+2] = True
                    
                    if t == tiles["start"]:
                        gristmill = templates["structure_gristmill"].format(
                            id=entity_id,
                            i=i,
                            j=j,
                            x=i + 0.5,
                            y=j + 0.5,
                            h=el,
                            owner_id=self.ownerships[i, j]
                        )
                        entities.append(gristmill)
                        entity_id += 1

                        start_tiles.append((i, j))

                terrains.append(terrainkeys["grass"])

        start0, start1 = start_tiles

        mapname = f"kestrl_{VERSION}_{self.seed}"
        mapxml = templates["map"].format(
            map_name=mapname,
            dimx=self.padded_size,
            dimy=self.padded_size,
            instartx=self.instart - 1,
            instopx=self.instop + 1,
            instarty=self.instart - 1,
            instopy=self.instop + 1,
            padded_insizex=self.instop - self.instart + 2,
            padded_insizey=self.instop - self.instart + 2,
            starting_mill_0=to_str(start0),
            starting_mill_1=to_str(start1),
            last_entity_id=entity_id - 1,
            terrains=to_str(terrains),
            elevations=to_str(elevations),
            entities=to_str(entities, sep="\n"))
        
        with open(f"maps/{mapname}.xml", "w", encoding="utf-8") as file:
            file.write(mapxml)

    def set_axis(self, ax):
        ax.set_xticks(np.arange(self.padded_size) + 0.5, minor='true')
        ax.set_yticks(np.arange(self.padded_size) + 0.5, minor='true')
        ax.grid(color="black", which='minor', alpha=0.2)

    def terraform(self):
        el = self.padded_elevations

        self.padded_elevations = np.zeros_like(self.padded_elevations, dtype=int)

        for i in range(self.padded_size):
            for j in range(self.padded_size):
                target_el = 2*el[i, j]
                if target_el > self.padded_tiles[i, j]:
                    self.terraform_tile(i, j, target_el, allow_down=False)

    def terraform_tile(self, i, j, el, allow_down=False):
        if i == 0 or j == 0 or i == self.padded_size - 1 or j == self.padded_size - 1:
            return

        diff = el - self.padded_elevations[i, j]

        if diff < 0 and not allow_down:
            return

        self.padded_elevations[i, j] = el
        
        for m, n in neighborhood(i, j):
            other = self.padded_elevations[m, n]

            if el - other >= 2:
                self.terraform_tile(m, n, el - 1, allow_down=allow_down)
            elif other - el >= 2:
                self.terraform_tile(m, n, el + 1, allow_down=allow_down)

    def to_upper_half(self, x, y):
        if x > self.size/2:
            x = self.size - x
        
        return np.array([x, y])
    
    def to_upper_triangle(self, x, y):
        return np.asarray(sorted([x, y]))



class Tile:
    index = 0

    def __init__(self, name, size=1, color="black"):
        self.index = Tile.index
        Tile.index += 1

        self.name = name
        self.size = size
        self.color = color


ts = [Tile("grass", size=1, color="burlywood"),
      Tile("free", size=2, color="burlywood"),
      Tile("blocker", size=1, color="darkgreen"),
      Tile("cliff", size=1, color="darkslategray"),
      Tile("ramp", size=1, color="peru"),
      Tile("water", size=1, color="dodgerblue"),
      Tile("road", size=1, color="burlywood"),
      Tile("pig", size=2, color="gold"),
      Tile("farm", size=2, color="wheat"),
      Tile("mill", size=2, color="white"),
      Tile("start", size=2, color="darkred")]

tiles = {t.name:t for t in ts}

# 3 warren space are always guaranteed
mill_layout = [["farm", "farm", "farm", "free"],
               ["farm", "mill", "farm", "free"],
               ["farm", "farm", "farm", "free"]]

starting_mill_layout = [["free", "free",  "free", "free"],
                        ["farm", "farm",   "pig", "free"],
                        ["farm", "start",  "pig", "free"],
                        ["farm",  "pig",   "pig", "free"]]


mill_layout = np.asarray([[tiles[t] for t in row] for row in mill_layout])
starting_mill_layout = np.asarray([[tiles[t] for t in row] for row in starting_mill_layout])

templates = {}

for filename in os.listdir("templates"):
    name = filename[:-4]

    with open(f"templates/{filename}", "r") as file:
        templates[name] = file.read()

terrainkeys = {"grass":"1",
               "water":"2",
               "grass_cliff":"3",
               "grass_ramp":"4"}

N = 0

while N < 20:
    try:
        game_map = Map(size=45)
        game_map.save_map()
        N += 1
    except MapGenerationFailure:
        pass

# print(list(chain.from_iterable(game_map.padded_elevations)))

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

xx, yy = np.transpose(game_map.waypoints)
tri = Delaunay(game_map.waypoints)
xlim = [game_map.instart - 0.5, game_map.instop - 0.5]
ylim = xlim
extent = xlim + ylim

for ax in axes:
    game_map.set_axis(ax)

ax = axes[0]

color_bounds = np.arange(len(tiles) + 1) - 0.5
cmap = colors.ListedColormap([tile.color for tile in tiles.values()])
norm = colors.BoundaryNorm(color_bounds, cmap.N, clip=True)
ax.imshow(np.transpose(game_map.padded_tiles), cmap=cmap, norm=norm, extent=extent, origin="lower")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# ax.plot(xx, yy, "ks")
# ax.triplot(xx, yy, tri.simplices, "k-", alpha=0.3)

ax = axes[1]
ax.matshow(np.transpose(game_map.padded_elevations), extent=extent, origin="lower")

plt.show()