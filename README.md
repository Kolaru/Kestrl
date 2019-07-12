# Kestrl

The Kestrl generator (**K**olaru **E**xceptional **S**ymmetric **T**errain **R**oad and **L**and generator) is a map generator for the game Tooth and Tail.

## The algorithm

The algorithm goes as follow

1. Generate Perlin noise and add it to itself to make it symmetric.
2. Use it to generate the elevation of the terrain and water.
3. Terraform the terrain (i.e. create valid ramp transition between elevations).
4. Generate random symmetric waypoints in the map.
5. Improve their distribution using partial Voronoi regularization (i.e. move each point closer to the centroid of its Voronoi cell).
6. Find two good waypoints to put the starting mills on.
7. Select random waypoints and try to add mills to them until all waypoints have been tried.
8. Flatten the mills regions (including free warren spaces around).
9. Generate Perlin noise (again symmetrized), representing the blocker density at each tile (0% to 100%).
10. Add blockers according to this density.
11. Generate the Delaunay triangularization of the waypoints.
12. Create one random symmetric path from one starting mill to the other (going through the same waypoint multiple time is allowed).
13. Clear blockers on that path.
14. Connect all mills not already connected to it to that path.
15. Close the borders of the map.