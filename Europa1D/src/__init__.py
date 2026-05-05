"""
Europa Ice Shell Thermal Model - Source Package

Core modules:
    constants           - Physical constants and material parameters
    Physics             - Ice physics engine (viscosity, tidal heating, Stefan condition)
    Convection          - Stagnant-lid convection parameterization
    Boundary_Conditions - Surface boundary conditions (Dirichlet, Stefan-Boltzmann)
    Solver              - 1D transient heat conduction solver (Crank-Nicolson)
    Monte_Carlo         - Monte Carlo uncertainty framework
    regional_samplers   - Regional parameter samplers (equator/pole)
"""
