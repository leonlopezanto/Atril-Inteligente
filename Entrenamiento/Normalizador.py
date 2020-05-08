 # -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:20:37 2019

@author: Antonio
"""


import numpy 

class Normalizador(object):
    """
    Clase Normalizadora
    Acumula los datos estadísticos del dataset.
    """
    def __init__(self):
        """Inicializa los datos que va a acumular"""
        self.N = 0
        self.mean = 0
        self.S1 = 0
        self.S2 = 0
        self.std = 0
        
        
    def accumulate(self,stat):
        """Acumula las estadísticas que recibe 
        Parametros:
            -stat : dict
        """   
        self.N += stat['N']
        self.mean += stat['mean']
        self.S1 += stat['S1']
        self.S2 += stat['S2']
        
        
    def finalize(self):
        """Calcula la media y la desviación típica acumuladas """
        
        self.mean = self.S1 / self.N
        self.std = numpy.sqrt((self.N * self.S2 - (self.S1 * self.S1))/ (self.N * (self.N-1)))
        
        #Si tuviesemos demasiado material, podríamos obtener std = Nan, 0.0
        self.std = numpy.nan_to_num(self.std)
        
        self.mean = self.mean.reshape(1, -1)
        self.std = self.std.reshape(1,-1)
    
    
    def saveValues(self, nameModel=''):
        values = []
        
        values.append(self.mean)
        values.append(self.std)

        if nameModel == '':
            pathM = "./mediaNorm.txt"
            pathS = "./stdNorm.txt"
        else:
            pathM = "./Modelos/" +'Model_' + nameModel + "/mediaNorm.txt"
            pathS = "./Modelos/" +'Model_' + nameModel + "/stdNorm.txt"
            
        with open(pathM,"bw") as dataFile:
            numpy.savetxt(dataFile, self.mean, fmt='%.20e')

        with open(pathS,"bw") as dataFile:
            numpy.savetxt(dataFile, self.mean, fmt='%.20e')
    
    
    def loadValues(self, path='./'):
        pathMEAN = path +'mediaNorm.txt'
        pathSTD = path+'stdNorm.txt'
        
        with open(pathMEAN,"br") as datafiles:
            self.mean = numpy.loadtxt(datafiles)
#            self.mean = numpy.mean(self.mean)
        with open(pathSTD, "br") as datafiles:
            self.std = numpy.loadtxt(datafiles)
#            self.std = numpy.std(self.std)
        
        
                
    
    def normalize(self, data_matrix):
        """Normaliza la matriz de datos con los datos internos de la clase.
        Parametros:
            -data_matrix : matriz que se va a normalizar.
        Return: data_matrix después de haber sido normalizada.
        """
        return numpy.nan_to_num((data_matrix - self.mean)/self.std)