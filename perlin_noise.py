import numpy as np

from numpy import random

# Based on noise.py by Casey Duncan
# Copyright (c) 2008, Casey Duncan (casey dot duncan at gmail dot com)

"""Perlin noise -- numpy implementation"""

# 3D Gradient vectors
_GRAD3 = np.asarray((
    (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
    (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
    (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1),
    (1,1,0),(0,-1,1),(-1,1,0),(0,-1,-1),
))

# Simplex skew constants
_F2 = 0.5 * (np.sqrt(3.0) - 1.0)
_G2 = (3.0 - np.sqrt(3.0)) / 6.0
_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0

class SimplexNoise:
    """
    Perlin simplex noise generator

    Adapted from Stefan Gustavson's Java implementation described here:

    http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf

    To summarize:

    "In 2001, Ken Perlin presented 'simplex noise', a replacement for his classic
    noise algorithm.  Classic 'Perlin noise' won him an academy award and has
    become an ubiquitous procedural primitive for computer graphics over the
    years, but in hindsight it has quite a few limitations.  Ken Perlin himself
    designed simplex noise specifically to overcome those limitations, and he
    spent a lot of good thinking on it. Therefore, it is a better idea than his
    original algorithm. A few of the more prominent advantages are:

    * Simplex noise has a lower computational complexity and requires fewer
      multiplications.
    * Simplex noise scales to higher dimensions (4D, 5D and up) with much less
      computational cost, the complexity is O(N) for N dimensions instead of
      the O(2^N) of classic Noise.
    * Simplex noise has no noticeable directional artifacts.  Simplex noise has
      a well-defined and continuous gradient everywhere that can be computed
      quite cheaply.
    * Simplex noise is easy to implement in hardware."
    """

    def __init__(self, period=256, scale=1.0, amplitude=1.0):
        """
        Initialize the simplex noise generator.

        # Parameters
            * `period`: Determine the interval after which the noise repeats,
                which is useful for creating tiled textures. This value should
                be an integer power of 2 (no check is performed). Default to `256`.
            * `scale`: Scale of the noise. Default to `1`.
            * `amplitude`: Amplitude of the noise. The noise returned will be
                between `-amplitude` and `amplitude`.

        Note that the speed of the noise algorithm is independent of
        the period size, though larger periods mean a larger table, which
        consume more memory.
        """
        self.period = period
        self.scale = scale
        self.amplitude = amplitude
        permutation = random.permutation(period)
        # Repeat permutation array so we don't need to wrap
        self.permutation = np.hstack((permutation, permutation))

    def noise2(self, x, y):
        """2D Perlin simplex noise.

        Return a numpy `array` of floating point value from `-self.amplitude` to
        `self.amplitude` for arrays of `x` and `y` coordinates. The same value
        is always returned for a given coordinate.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")

        x = x/self.scale
        y = y/self.scale

        # Skew input space to determine which simplex (triangle) we are in
        s = (x + y) * _F2
        i = np.floor(x + s)
        j = np.floor(y + s)
        t = (i + j) * _G2

        # "Unskewed" distances from cell origin
        x0 = x - (i - t)
        y0 = y - (j - t)

        # Upper triangle, YX order: (0,0)->(0,1)->(1,1)
        i1 = np.zeros_like(x0, dtype=int)
        j1 = np.ones_like(x0, dtype=int)

        # Lower triangle, XY order: (0,0)->(1,0)->(1,1)
        mask = x0 > y0
        i1[mask] = 1
        j1[mask] = 0

        # Offsets for middle corner in (x,y) unskewed coords
        x1 = x0 - i1 + _G2
        y1 = y0 - j1 + _G2

        # Offsets for last corner in (x,y) unskewed coords
        x2 = x0 + _G2 * 2.0 - 1.0
        y2 = y0 + _G2 * 2.0 - 1.0

        # Determine hashed gradient indices of the three simplex corners
        perm = self.permutation
        ii = i.astype(int) % self.period
        jj = j.astype(int) % self.period
        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1]] % 12
        gi2 = perm[ii + 1 + perm[jj + 1]] % 12

        # Calculate the contribution from the three corners
        noise = np.zeros_like(x)
        self.corner_contribution(x0, y0, _GRAD3[gi0], noise)
        self.corner_contribution(x1, y1, _GRAD3[gi1], noise)
        self.corner_contribution(x2, y2, _GRAD3[gi2], noise)

        noise *= 70.0 # scale noise to [-1, 1]
        return self.amplitude * noise

    def corner_contribution(self, x, y, g, noise):
        tt = 0.5 - x**2 - y**2
        mask = tt > 0
        noise[mask] += tt[mask]**4 * (g[mask, 0]*x[mask] + g[mask, 1]*y[mask])


class PerlinNoise:
    """
    Perlin noise generator with fractal brownian motion (FBM).

    FBM consists in adding several simplex noises of different scale and
    amplitudes to get a less smooth result.
    """
    def __init__(self, period=256, scale=1.0, amplitude=1.0, octaves=5, persistence=0.5, lacunarity=2.0):
        """
        Initialize the PerlinNoise generator.

        # Parameters
            * `octaves`: Number of simplex noises to add together. Default to `5`.
            * `persistence`: Factor multiplying the amplitude of successive
                octaves. Default to `0.5`.
            * `lacunarity`: Factor dividing the scale of successive octaves.
                Default to `2.0`.

        Other parameters determine the `period`, `scale` and `amplitude` of the
        first octave. For more details see `SimplexNoise`.
        """

        subnoises = []
        amp = amplitude
        max = 0
        for k in range(octaves):
            subnoises.append(SimplexNoise(amplitude=amp, scale=scale, period=period))
            max += amp
            amp *= persistence
            scale = scale/lacunarity

        self.normalization = amplitude/max
        self.subnoises = subnoises
        self.amplitude = amplitude
        self.shadows = {}

    def noise2(self, x, y):
        res = 0
        for noise in self.subnoises:
            res += noise.noise2(x, y)
        return res * self.normalization

    def noise_grid(self, xx, yy):
        return np.asarray([self.noise2(np.ones_like(xx)*y, xx) for y in yy])

    def is_in_shadow(self, x, y, lightdir, h=None, step=2):
        try:
            return self.shadows[(x, y)]
        except KeyError:
            h = self.noise2(np.asarray([x]), np.asarray([y]))[0]

            lightdir = np.asarray(lightdir)
            lightdir /= norm(lightdir)
            Lx, Ly, Lz = lightdir
            pos = np.array((x, y, h))

            # Light will pass over everything that is further away, since the maximum height is `self.max`
            D = -(self.amplitude - h)/Lz

            points = np.multiply.outer(np.arange(1, D, step), -lightdir) + pos
            point_hs = self.noise2(*points[:, 0:2].T)
            light_hs = points[:, 2]

            shadow = (light_hs < point_hs).any()
            self.shadows[(x, y)] = shadow

            return shadow
