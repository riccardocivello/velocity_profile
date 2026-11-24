# constants.py
from scipy import constants as sc

N_AVOGADRO = sc.Avogadro
K_B = sc.Boltzmann                 # J/K
A2M = 1e-10
FS2S = 1e-15                       # femtoseconds to seconds
NS2S = 1e-9                        # nanoseconds to seconds
G2KG = 1e-3
A2NM = 0.1
G_MOL2KG = 1e-3 / N_AVOGADRO
A_FS2M_S = A2M / FS2S
KCAL_MOL_A2N = 4184 / N_AVOGADRO / A2M   # force_factor voncerting from Kcal/mol/Angstrom to Newton
MOM_FACTOR = A_FS2M_S*G_MOL2KG
INTERFACE_Z_A = 28.2