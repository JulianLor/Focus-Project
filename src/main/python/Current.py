import numpy as np

class Current:
    # class level constants
    I_MAX   = 64     # maximal current

    # class attribute types
    system: str     # how many coils: 2, 3, 4
    alpha:  float   # sigma_1/sigma_2: ratio of ellipse's axis: short over long
    angle:  list    # angle between coils
    Hz:     float   # frequency of rotation
    period: float   # duration for the rot. axis to return to original axis

    def __init__(self, system, alpha, angle, Hz, period):
        self.system = system
        self.alpha = alpha
        self.angle = angle
        self.Hz = Hz
        self.period = period

    
