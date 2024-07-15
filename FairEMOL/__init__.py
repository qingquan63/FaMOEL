# -*- coding: utf-8 -*-
import sys
import platform

__author__ = "Geatpy Team"
__version__ = "2.6.0"

# import the core



plat = platform.system().lower()

if plat == 'windows':
    lib_path = __file__[:-11] + 'core/Windows/lib64/v3.7/'
else:
    lib_path = __file__[:-11] + 'core/Linux/lib64/v3.7/'

if lib_path not in sys.path:
    sys.path.append(lib_path)



lib_path = __file__[:-11]
if lib_path not in sys.path:
    sys.path.append(lib_path)
from Algorithm import Algorithm
from Algorithm import MoeaAlgorithm
from Algorithm import SoeaAlgorithm
from Population import Population
from Problem import Problem
from FairProblem import FairProblem

from EvolutionaryCodes.nets import Population_NN, IndividualNet, weights_init
from EvolutionaryCodes.Evaluate import Cal_objectives
from EvolutionaryCodes.load_data import load_data
from EvolutionaryCodes.nets import sigmoid, mutate
from EvolutionaryCodes.GroupInfo import GroupInfo, GroupsInfo
from EvolutionaryCodes.Mutation_NN import Mutation_NN
from EvolutionaryCodes.Mutation_NN import Crossover_NN
from EvolutionaryCodes.data.objects.ProcessedData import ProcessedData

from templates.Fair_MOEA_template import Fair_MOEA_template

from templates.MOEAs.TwoArch2 import update_DA, update_CA
