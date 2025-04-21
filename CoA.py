import numpy as np

def COA(FOBJ, lu, nfevalMAX, learning_rate, n_packs=20, n_coy=5):
    # Optimization problem variables
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]
    # Algorithm parameters
    if n_coy < 3:
        raise Exception("At least 3 coyotes per pack must be used")
    # Probability of leaving a pack
    p_leave = 0.005 * (n_coy ** 2)
    Ps = 1 / D
    # Packs initialization (Eq. 2)
    pop_total = n_packs * n_coy
    costs = np.zeros((1, pop_total))
    coyotes = np.tile(VarMin, [pop_total, 1]) + np.random.rand(pop_total, D) * np.tile(VarMax, [pop_total, 1]) - \
              np.tile(VarMin, [pop_total, 1])
    ages = np.zeros((1, pop_total))
    packs = np.random.permutation(pop_total).reshape(n_packs, n_coy)
    # Evaluate coyotes adaptation (Eq. 3)
    for c in range(pop_total):
        costs[0, c] = FOBJ(coyotes[c, :])
    nfeval = pop_total
    # Output variables
    globalMin = np.min(costs[0, :])
    ibest = np.argmin(costs[0, :])
    globalParams = coyotes[ibest, :]
    # Main loop
    year = 1
    while nfeval < nfevalMAX:  # Stopping criteria
        # Update the years counter
        year += 1
        # Execute the operations inside each pack
        for p in range(n_packs):
            # Get the coyotes that belong to each pack
            coyotes_aux = coyotes[packs[p, :], :]
            costs_aux = costs[0, packs[p, :]]
            ages_aux = ages[0, packs[p, :]]
            # Detect alphas according to the costs (Eq. 5)
            ind = np.argsort(costs_aux)
            costs_aux = costs_aux[ind]
            coyotes_aux = coyotes_aux[ind, :]
            ages_aux = ages_aux[ind]
            c_alpha = coyotes_aux[0, :]
            # Compute the social tendency of the pack (Eq. 6)
            tendency = np.median(coyotes_aux, 0)
            # Update coyotes' social condition
            new_coyotes = np.zeros((n_coy, D))
            for c in range(n_coy):
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(n_coy)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(n_coy)
                # Try to update the social condition according
                # to the alpha and the pack tendency (Eq. 12)
                new_coyotes[c, :] = coyotes_aux[c, :] + learning_rate * (
                        np.random.rand() * (c_alpha - coyotes_aux[rc1, :]) +
                        np.random.rand() * (tendency - coyotes_aux[rc2, :]))
                # Apply boundary constraints (Eq. 7)
                new_coyotes[c, :] = Limita(new_coyotes[c, :], VarMin, VarMax)
                # Evaluate the new social condition
                new_cost = FOBJ(new_coyotes[c, :])
                nfeval += 1
                # Compare the new condition with the previous one (Eq. 13)
                if new_cost < costs_aux[c]:
                    coyotes_aux[c, :] = new_coyotes[c, :]
                    costs_aux[c] = new_cost
                    ages_aux[c] = 0
                else:
                    ages_aux[c] += 1
            # Apply aging (Eq. 14)
            ages_aux += 1
            # Update the packs and costs
            packs[p, :] = packs[p, np.argsort(costs_aux)]
            costs[0, packs[p, :]] = costs_aux[np.argsort(costs_aux)]
            coyotes[packs[p, :], :] = coyotes_aux[np.argsort(costs_aux), :]
            ages[0, packs[p, :]] = ages_aux[np.argsort(costs_aux)]
        # Update the best solution
        ibest = np.argmin(costs)
        globalMin = np.min(costs)
        globalParams = coyotes[ibest, :]
    return globalMin, globalParams

# Define the objective function
def objective_function(x):
    return x[0] ** 2 + x[1] ** 2  # Sample objective function (sum of squares)

# Define the boundary constraints function
def Limita(x, VarMin, VarMax):
    return np.maximum(np.minimum(x, VarMax), VarMin)

# Example usage

# Define the search space bounds
lower_bound = np.array([-10, -10])
upper_bound = np.array([10, 10])
bounds = np.array([lower_bound, upper_bound])

# Set the learning rate
learning_rate = 0.001

# Perform the Coyote Optimization Algorithm
best_cost, best_params = COA(objective_function, bounds, nfevalMAX=10000, learning_rate=learning_rate)

# Print the results
print("Learning Rate:", learning_rate)