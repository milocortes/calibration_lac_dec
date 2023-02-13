from scipy.stats.qmc import LatinHypercube,scale
import math


# Definimos la clase Particle
class Particle:
    def __init__(self,x,v):
        self.x = x
        self.v = v
        self.x_best = x
        
def PSO(f, pop_size, maxiter, n_var, lb, ub, α, β, w):
    '''
    ------------------------------------------
                        PSO
    Particle Swarm Optimization
    -------------------------------------------
    ## Implemented as a minimization algorithm
    # Inputs:
        * f             - function to be minimized
        * pop_size      - number of individuals in the population
        * max_iter     - maximum number of optimization iterations
        * n_var
        * lb
        * ub
        * α             - Social scaling parameter
        * β             - Cognitive scaling parameter
        * w             - velocity inertia
        
    # Output
        * x_best        - best solution found
        * fitness_values - history of best score
    '''   
    # LatinHypercube sampling
    engine = LatinHypercube(d=n_var)
    sample = engine.random(n=pop_size)

    l_bounds = np.array(lb)
    u_bounds = np.array(ub)

    sample_scaled = scale(sample,l_bounds, u_bounds)
    sample_scaled = scale(sample,l_bounds, u_bounds)

    # define particle population
    pob = [Particle(x,np.array([0]*n_var)) for x in sample_scaled]


    
    x_best = pob[0].x_best
    y_best = f(x_best)

    
    # minimum value for the velocity inertia
    w_min = 0.4
    # maximum value for the velocity inertia
    w_max = 0.9

    # Velocidad máxima
    vMax = np.multiply(u_bounds-l_bounds,0.2)
    # Velocidad mínima
    vMin = -vMax

    
    for P in pob:
        y = f(P.x)
        if y < y_best:
            x_best = P.x_best
            y_best = y

    fitness_values = []

    print("*********************")
    print(f"Mejor valor {y_best}")
    print("*********************")
    for k in range(maxiter):
        
        print("-----------------------------")
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("        Iteración {}".format(k))
        print("-%%%%%%%%%%%%%%%%%%%%%%%%%%%-")
        print("-----------------------------")
        
        for P in pob:
            # Actualiza velocidad de la partícula
            ϵ1,ϵ2 = np.random.uniform(), np.random.uniform()
            P.v = w*P.v + α*ϵ1*(P.x_best - P.x) + β*ϵ2*(x_best - P.x)

            # Ajusta velocidad de la partícula
            index_vMax = np.where(P.v > vMax)
            index_vMin = np.where(P.v < vMin)

            if np.array(index_vMax).size > 0:
                P.v[index_vMax] = vMax[index_vMax]
            if np.array(index_vMin).size > 0:
                P.v[index_vMin] = vMin[index_vMin]

            # Actualiza posición de la partícula
            P.x += P.v

            # Ajusta posición de la particula
            index_pMax = np.where(P.x > u_bounds)
            index_pMin = np.where(P.x < l_bounds)

            if np.array(index_pMax).size > 0:
                P.x[index_pMax] = u_bounds[index_pMax]
            if np.array(index_pMin).size > 0:
                P.x[index_pMin] = l_bounds[index_pMin]

            # Evaluamos la función
            y = f(P.x)

            if y < y_best:
                x_best = np.copy(P.x_best)
                y_best = y
            if y < f(P.x_best):
                P.x_best = np.copy(P.x)
            

            # Actualizamos w

            w = w_max - k * ((w_max-w_min)/maxiter)

        print(y_best)
        fitness_values.append(y_best)

    return fitness_values ,x_best