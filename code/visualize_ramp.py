# import of modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# plotting of junctions, boundaries and electrodes, to be modified manually
# 
# each line is defined by a list of two lists, first is 'x' coordinate, second is 'y' coordinate
# note that 'y' coordinate is inverted

plot_junctions = True

contour = [[0.0, 170.0, 170.0, 0.0, 0.0], [0.0, 0.0, -25.0, -25.0, 0.0]]
incollector = [[0.0, 170.0], [-20.0, -20.0]]
collectorbase = [[0.0, 140.0, 140.0], [-6.0, -6.0, 0.0]]
emitterbase = [[15.0, 15.0, 130.0, 130.0], [0.0, -5.0, -5.0, 0.0]]

eemitter = [[60.0, 100.0], [0.0, 0.0]]
ecollector = [[0.0, 170.0], [-25.0, -25.0]]
ebase1 = [[0.0, 10.0], [0.0, 0.0]]
ebase2 = [[132.0, 138.0], [0.0, 0.0]]

lines = [contour, incollector, collectorbase, emitterbase]
elecs = [eemitter, ecollector, ebase1, ebase2]

def plot_le(ax, plot_junctions, lines, elecs):
    if not plot_junctions:
        return
    for line in lines:
        ax.plot(line[0], line[1], color='k')
    for elec in elecs:
        ax.plot(elec[0], elec[1], color='w', linewidth=3.0)
    return

# data extraction and initial processing functions

def extract(filename):
    with open(filename, 'r') as file:
        points = []
        values = []
        for line in file:
            if line[0] == 'c' and line[1] == ' ':
                points.append(line[1:].split())
            elif line[0] == 'n' and line[1] == ' ':
                values.append(line[1:].split())
    for point in points:
        point[0] = int(point[0])
        for i in range(1, len(point)):
            point[i] = float(point[i])
    for value in values:
        value[0] = int(value[0])
        for i in range(1, len(value)):
            value[i] = float(value[i])
    dell = []
    for i in range(len(points[:-1])):
        if points[i+1][0] - points[i][0] != 1:
            nval = []
            for j in range(len(points[i+1])):
                if points[i+1][j] != points[i][j]:
                    nval.append((points[i+1][j] + points[i][j])/2)
                else:
                    nval.append(points[i][j])
            points[i] = nval
            dell.append(i+1)
    for deli in reversed(dell):
        del points[deli]
    dell = []
    for i in range(len(values[:-1])):
        if values[i+1][0] - values[i][0] == 0:
            nval = []
            for j in range(len(values[i+1])):
                if values[i+1][j] != values[i][j]:
                    nval.append((values[i+1][j] + values[i][j])/2)
                else:
                    nval.append(values[i][j])
            values[i] = nval
            dell.append(i+1)
    for deli in reversed(dell):
        del values[deli]
    if len(points) != len(values):
        raise Exception("Number of points and number of values do not agree after initial processing.")
    return points, values

# data processing function, index of required data and data function is to be modified manually
# 
# inecies are found in .str file, line starting with "s ", keys in lines starting with "Q "
# "indx" syntax: "int" for single values, "list" with two integers for 2D vectors

indx = [-9, -33, -34, -3, -6, -28, -35] # total current density (A/cm^2); hole density (1/cm^3); electron density (1/cm^3), displacement current density (A/cm^2); conductive current density (A/cm^2); temperature (K); potential (V)
d2vec = lambda x, i, j: (x[i]**2 + x[j]**2)**0.5

def full_data(points, values, indx):
    data = []
    for i in range(len(points)):
        lst = [points[i][1], points[i][2]]
        for ix in indx:
            if type(ix) is int:
                lst.append(values[i][ix])
            elif type(ix) is list:
                if len(ix) == 2:
                    lst.append(d2vec(values[i], ix[0], ix[1]))
        data.append(lst)
    data = np.array(data)
    return data

# single sample plotting
# 
# update cmap_pivots to change colormap properties

# cmap_pivots = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]   # uniform
cmap_pivots = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # logarithmic
cmap_dict = {'red':   [[cmap_pivots[0], 1.0, 1.0], [cmap_pivots[1], 1.0, 1.0], [cmap_pivots[2], 0.0, 0.0], [cmap_pivots[4], 0.0, 0.0], [cmap_pivots[5], 1.0, 1.0]],
             'green': [[cmap_pivots[0], 0.0, 0.0], [cmap_pivots[1], 1.0, 1.0], [cmap_pivots[3], 1.0, 1.0], [cmap_pivots[4], 0.0, 0.0], [cmap_pivots[5], 0.0, 0.0]],
             'blue':  [[cmap_pivots[0], 0.0, 0.0], [cmap_pivots[2], 0.0, 0.0], [cmap_pivots[3], 1.0, 1.0], [cmap_pivots[5], 1.0, 1.0]]}
fine_rygcbm_cmap = LinearSegmentedColormap('fine_rygcbm_cmap', cmap_dict, N=65536)

def scmap(fig, ax, sim, n):
    filename = "ramp_" + sim + "_r/ramp_" + sim + str(n) + ".str"
    data = full_data(*extract(filename), indx)
    x = data[:, 0]
    y = -data[:, 1]
    z = data[:, 2]
    xg = np.linspace(min(x), max(x), 1000)
    yg = np.linspace(min(y), max(y), 1000)
    xg, yg = np.meshgrid(xg, yg)
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    zg = interp(xg, yg)
    pcm = ax.pcolormesh(xg, yg, zg, shading='auto', cmap=fine_rygcbm_cmap, vmin=0.0, vmax=8000.0)
    fig.colorbar(pcm, ax=ax)
    plot_le(ax, plot_junctions, lines, elecs)
    return pcm
   
#fig, ax = plt.subplots()
#scmap(fig, ax, "medium", 30)

# animated plotting

sim = "slow"
if sim == "slow":
    tstep = 1.0
    framen = 80
elif sim == "medium":
    tstep = 0.06
    framen = 72
elif sim == "fast":
    tstep = 0.03
    framen = 73
# tstep = 1.0   # 1.0 for slow; 0.06 for medium; 0.03 for fast
ttl = lambda n: f"Hole density [1/m^3], t = {round(tstep * n, 2)}ns"   # update title
dindex = 3   # update index of plotted data
dmax = -1000000000
dmin = 1000000000

fig, ax = plt.subplots()
plt.title("ramp_" + sim)
filename = "ramp_" + sim + "_r/ramp_" + sim + "0.str"
data = full_data(*extract(filename), indx)
x = data[:, 0]
y = -data[:, 1]
z = data[:, dindex] #- min(data[:, dindex])
xg = np.linspace(min(x), max(x), 1000)
yg = np.linspace(min(y), max(y), 1000)
xg, yg = np.meshgrid(xg, yg)
interp = LinearNDInterpolator(list(zip(x, y)), z)
zg = interp(xg, yg)
pcm = ax.pcolormesh(xg, yg, zg, shading='auto', cmap=fine_rygcbm_cmap, vmin=0.0, vmax=120000)
fig.colorbar(pcm, ax=ax)
plot_le(ax, plot_junctions, lines, elecs)

def init():
    filename = "ramp_" + sim + "_r/ramp_" + sim + "0.str"
    data = full_data(*extract(filename), indx)
    x = data[:, 0]
    y = -data[:, 1]
    z = data[:, dindex] #- min(data[:, dindex])
    xg = np.linspace(min(x), max(x), 1000)
    yg = np.linspace(min(y), max(y), 1000)
    xg, yg = np.meshgrid(xg, yg)
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    zg = interp(xg, yg)
    pcm.set_array(zg)
    plt.title(ttl(1))
    return pcm

def update(frame):
    global dmax, dmin
    filename = "ramp_" + sim + "_r/ramp_" + sim + str(frame) + ".str"
    data = full_data(*extract(filename), indx)
    x = data[:, 0]
    y = -data[:, 1]
    z = data[:, dindex] #- min(data[:, dindex])
    zmax = max(z)
    zmin = min(z)
    dmax = max([zmax, dmax])
    dmin = min([zmin, dmin])
    xg = np.linspace(min(x), max(x), 1000)
    yg = np.linspace(min(y), max(y), 1000)
    xg, yg = np.meshgrid(xg, yg)
    interp = LinearNDInterpolator(list(zip(x, y)), z)
    zg = interp(xg, yg)
    pcm.set_array(zg)
    plt.title(ttl(frame+1))
    return pcm

ani = FuncAnimation(fig, update, frames=np.arange(0, framen), init_func=init, repeat=False, interval=600)
ani.save(filename=("ramp_" + sim + "_holes.gif"), writer="pillow")

print(f"minimun value: {dmin}")
print(f"maximum value: {dmax}")
