#######################################
#### FUNÇÕES PARA UTILIZAÇÃO REO 1 ####
#######################################

import numpy as np
def f_media (x):
    it=0
    somador=0
    for el in x:
        somador += el
        it = it+1
        media= somador / it
    return media

#############################################################
def f_var_amostral (x):
    global var_amostral
    somador = 0
    it = 0
    sdquad = 0
    for el in x:
        it = it + 1
        somador += el
        sdquad += el**2
    var_amostral= (sdquad-((somador)**2/ it)) / (it - 1)
    return var_amostral
