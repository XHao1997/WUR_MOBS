import numpy as np


# %% Simulation function
def fnc_y(model, x0, d, tsim):
    # Reset initial conditions
    grass, water = model
    x0_grs, x0_wtr = x0
    d_grs, d_wtr = d
    grass.x0 = x0_grs.copy()
    water.x0 = x0_wtr.copy()

    # Initial disturbance
    d_grs['WAI'] = np.array([[0, 1, 2, 3, 4], [1., ] * 5]).T

    # Iterator
    # (stop at second-to-last element, and store index in Fortran order)
    it = np.nditer(tsim[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Integration span
        tspan = (tsim[idx], tsim[idx + 1])
        # Controlled inputs
        u_grs = {'f_Gr': 0, 'f_Hr': 0}  # [kgDM m-2 d-1]
        u_wtr = {'f_Irg': 0}  # [mm d-1]
        # Run grass model
        y_grs = grass.run(tspan, d_grs, u_grs)
        # Retrieve grass model outputs for water model
        d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
        # Run water model
        y_wtr = water.run(tspan, d_wtr, u_wtr)
        # Retrieve water model outputs for grass model
        d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])

    # Return result of interest (WgDM [kgDM m-2])
    return grass.y['Wg'] / 0.4
