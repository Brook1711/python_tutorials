import numpy as np
import math
def calculate_fading(stdShadow, dist):
    h_pl_db = 148.1 + 37.6 * math.log10(dist)
    h_sh_db = np.random.randn(1) * stdShadow

    ray_array = np.random.randn(2)
    h_small =math.pow((math.pow(ray_array[0],2)+ math.pow(ray_array[1],2)),0.5)

    h_large = math.pow(10 , ( ( - h_pl_db + h_sh_db) /10.0))

    h_combined = h_small * h_large
    return h_combined