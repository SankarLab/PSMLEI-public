"""
Importing libraries
"""
import config_event_generation
import os, sys
import numpy as np
import random
from numpy import genfromtxt
import pandas as pd
from scipy import io
import matplotlib.pyplot as plt
import networkx as net
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import numpy.ma as ma
from tabulate import tabulate
import time
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
# from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib # Set the backend to 'Agg'
# matplotlib.use('Agg')
import pickle
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix

from sklearn.metrics import pairwise_distances
import plotly.offline as pyo
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib.animation as animation
from typing import Union
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from scipy.special import comb
from matplotlib.legend_handler import HandlerBase, HandlerTuple
import textwrap
from pandas.plotting import scatter_matrix
from sklearn.metrics import pairwise_distances_argmin_min

from scipy.io import loadmat
from collections import Counter
from heapq import nlargest
from sklearn import tree
from matplotlib.font_manager import FontProperties
import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score, \
    precision_score
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from imblearn.under_sampling import (
    TomekLinks, NeighbourhoodCleaningRule, AllKNN, EditedNearestNeighbours,
    CondensedNearestNeighbour, ClusterCentroids, NearMiss, RepeatedEditedNearestNeighbours
)

from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from math import factorial
from sklearn.utils import shuffle
## Semi- supervised repo -- import libraries
from methods import scikitTSVM
from frameworks.CPLELearning import CPLELearningModel
from methods.scikitWQDA import WQDA
from itertools import product
import seaborn as sns
from warnings import simplefilter
import os
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.linalg import hankel
from scipy import signal
from scipy.io import loadmat
import pandas as pd
import pytz, datetime
import math
from scipy.io import savemat
from scipy import io
warnings.filterwarnings("ignore", category=RuntimeWarning)
simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# Ignore the PendingDeprecationWarning
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
import matplotlib
matplotlib.use('Agg')
min_max_scaler = preprocessing.MinMaxScaler()


def create_environment(psse_path):
    PSSE_PATH_1 = psse_path
    sys.path.append(PSSE_PATH_1)
    os.environ['PATH'] += ';' + PSSE_PATH_1


class SystemComponents:
    def __init__(self,
                 raw_path, psspy, psse_vers):
        """
        This class retrieves the list of following components:
        busnumbers: number of buses in the system
        busbaseMVA: Base MVA of each bus in the system
        busbaseKV: base KV of each bus in the system
        line_mva: sorted list of lines based on their rated MVA
        line_from_bus: from bus of the lines
        line_to_bus: to bus of the lines
        trans_from_bus: from bus of the two-winding transformers in the system
        trans_to_bus: to bus of the two-winding transformers in the system
        load_id: load ids at each bus
        load_bus_num: sorted listed of load buses based on their actual MVA
        load_p: list of loads active power
        load_q: list of loads reactive power
        gen_name: name of the generators in the system
        gen_id: id of the generators at each bus in the system
        gen_bus_num: sorted list of the generator buses in the system based on their MVA generation capacity
        gen_p: list of generators active power
        gen_q: list of generators reactive power
        n_lines: number of lines in the system
        n_gens: number of generators in the system
        n_loads: number of load in the system
        :param raw_path: directory of the .raw file
        :param psspy: initializing the psse python api
        :param psse_vers: psse version of the .raw file
        """
        psspy.readrawversion(0, psse_vers, raw_path)
        _, load_id = psspy.aloadchar(sid=-1, string="ID")  # names and ids
        _, load_bus_num = psspy.aloadint(sid=-1, string="NUMBER")  # names and ids
        _, load_pq = psspy.aloadcplx(sid=-1, string="MVAACT")

        _, busnumbers = psspy.abusint(sid=-1, string="NUMBER")
        _, busbaseMVA = psspy.abusreal(sid=-1, string="BASE")
        _, busbaseKV = psspy.abusreal(sid=-1, string="KV")

        # From the PSSE raw file
        _, self.line_from_bus = psspy.abrnint(sid=-1, string='FROMNUMBER')
        _, self.line_to_bus = psspy.abrnint(sid=-1, string='TONUMBER')
        _, line_mva = psspy.abrnreal(sid=-1, string="RATE1")

        _, self.trans_from_bus = psspy.atrnint(sid=-1, string="FROMNUMBER")
        _, self.trans_to_bus = psspy.atrnint(sid=-1, string="TONUMBER")

        _, gen_name = psspy.amachchar(sid=-1, string="NAME")
        _, gen_id = psspy.amachchar(sid=-1, string="ID")
        _, n_gens = psspy.amachcount(sid=-1, flag=1)
        _, gen_bus_num = psspy.agenbusint(sid=-1, flag=4, string="NUMBER")
        _, gen_p = psspy.agenbusreal(sid=-1, flag=4, string="PGEN")
        _, gen_q = psspy.agenbusreal(sid=-1, flag=4, string="QGEN")
        _, self.gen_pq = psspy.amachcplx(sid=-1, flag=1, string="PQGEN")
        # agenbusreal(sid, flag, string)

        self.busnumbers = busnumbers[0]
        self.busbaseMVA = busbaseMVA[0]
        self.busbaseKV = busbaseKV[0]
        lines_sorted_lists = sorted(zip(line_mva[0], self.line_from_bus[0], self.line_to_bus[0]), key=lambda x: x[0], reverse=True)
        # Unpack the sorted lists
        self.line_mva_sorted, self.line_from_bus_sorted, self.line_to_bus_sorted = zip(*lines_sorted_lists)

        self.trans_from_bus = self.trans_from_bus
        self.trans_to_bus = self.trans_to_bus

        self.load_id = load_id[0]
        # Sort loads and their corresponding bus numbers based on their mva in descending order
        load_pq_abs = [abs(c) for c in load_pq[0]]
        load_sorted_lists = sorted(enumerate(load_pq_abs), key=lambda x: x[1], reverse=True)
        # Unpack the sorted lists and indices
        sorted_load_indices, self.load_pq_abs_sort = zip(*load_sorted_lists)
        self.load_bus_num_sorted = [load_bus_num[0][i] for i in sorted_load_indices]
        self.load_id_sorted = [self.load_id[i] for i in sorted_load_indices]
        self.load_p_sorted = [load_pq[0][i].real for i in sorted_load_indices]
        self.load_q_sorted = [load_pq[0][i].imag for i in sorted_load_indices]


        self.gen_name = gen_name[0]
        self.gen_bus_num = gen_bus_num[0]
        self.gen_id = gen_id[0]
        self.gen_p = gen_p[0]
        self.gen_q = gen_q[0]
        # Sort generators and their corresponding bus numbers based on their mva in descending order
        # self.gen_pq_abs = [abs(c) for c in self.gen_pq[0]]
        gen_sorted_lists = sorted(enumerate(self.gen_p), key=lambda x: x[1], reverse=True)
        # Unpack the sorted lists and indices
        self.sorted_gen_indices, self.gen_p_sorted = zip(*gen_sorted_lists)
        self.gen_bus_num_sorted = [self.gen_bus_num[i] for i in self.sorted_gen_indices]
        # self.gen_p_sorted = [self.gen_p[0][i] for i in self.sorted_gen_indices]
        self.gen_q_sorted = [self.gen_q[i] for i in self.sorted_gen_indices]

        self.n_lines = len(self.line_from_bus_sorted)
        self.n_gens = n_gens
        self.n_loads = len(self.load_bus_num_sorted)


class loading_scenarios:
    def __init__(self,
                 start_load_level: float = 0.99,
                 end_load_level: float = 1.01,
                 n_scenarios: float = 1,
                 min_rand: float = -0.02,
                 max_rand: float = 0.02):
        """
        Generate various system loading condition scenarios.
        For any individual load_i -> consumption S_i (MVA)= P_i (MW) + j Q_i (MVAR)
        Total system loading -> S_tot = sum(S_i)
        :param start_load_level: Lower bound in system loading condition scenarios ( min S_tot )/( nominal S_tot )
        :param end_load_level: Upper bound in system loading condition scenarios ( max S_tot )/( nominal S_tot)
        :param n_scenarios: Number of different loading condition scenarios. Load consumption for each load_i for any
                 given loading condition scenario j = 1, ..., n_scenarios is obtained as follows:
                 S_i[j] = (nominal S_i) + (nominal S_i) * [(end_load_level - start_load_level) / n_scenario] * j
        :param min_rand, max_rand:
                given 'min_rand' and 'max_rand', random load change ratios are generated within this range to change
                each individual load_i at loading scenario j, denoted as S_i[j].
                :
                S_i[j] = S_i[j] + rand(min_rand, max_rand) * S_i[j]
        """
        self.start_load_level = start_load_level
        self.end_load_level = end_load_level
        self.n_scenarios = n_scenarios
        self.min_rand = min_rand
        self.max_rand = max_rand


class simulation_setting():
    def __init__(self,
                 system_files_path: str,
                 project_path: str,
                 raw_file_name: str,
                 dyr_file_name: str,
                 psse_files_vers: str,
                 pmu_bus_select_method: str,
                 pmu_bus_num: None,
                 psse_set_minor: int = 4,
                 psseinit: int = 5000,
                 flat_run_time: float = 1.00,
                 remove_fault_time: float = 1.05,
                 simulation_time: float = 20.00,
                 generate_loading_scenarios: bool = True,
                 start_load_level: float = 0.99,
                 end_load_level: float = 1.01,
                 n_scenarios: float = 1,
                 min_rand: float = -0.02,
                 max_rand: float = 0.02,
                 event_types_list: list = []):
        """
        "random_in_range" --> Generates a list of bus numbers with pmus that spans from bus #1 to bus #n_bus with n_pmus
                      installed on every n_bus/n_pmus buses in the system.
        "random_in_buses" --> Randomly selects n_pmus buses out of n_bus buses in the system
        "specify_buses" --> User defined list of buses with pmus. If this method used, a pmu_bus_num list required as an input
        :return:
        """

        """
        Specifying the path for .raw and .dyr network data
        """
        # path -- PSSE .raw file
        self.raw_path = system_files_path + """\\""" + raw_file_name
        # path -- PSSE .dyr file
        self.dyr_path = system_files_path + """\\""" + dyr_file_name
        # PSSE version
        self.psse_vers = psse_files_vers
        self.pmu_bus_select_method = pmu_bus_select_method
        self.pmu_bus_num = pmu_bus_num
        """
        Initialize event types indicator
        """
        self.generate_generation_loss = False
        self.generate_load_loss = False
        self.generate_line_trip = False
        self.generate_line_fault = False
        self.generate_bus_fault = False
        """
        Simulation setting for specified event types
        """
        generated_events_path = project_path + """\\""" + "results" + """\\""" + "generated_events"
        manage_directory(path=generated_events_path, description='generated events path warning')

        generated_dataset_path = project_path + """\\""" + "results" + """\\""" + "generated_dataset"
        manage_directory(path=generated_dataset_path, description='generated dataset path warning')




        any_event_type_specified = False
        if "generation_loss" in event_types_list:
            self.generate_generation_loss = True
            # path -- to store the generation loss simulation results
            self.gen_res_path = generated_events_path #+ r"""\generation_loss"""
            # manage_directory(path=self.gen_res_path, description='Generation Loss events warning')
            any_event_type_specified = True

        if "load_loss" in event_types_list:
            self.generate_load_loss = True
            # path -- to store the load loss simulation results
            self.loadloss_res_path = generated_events_path #+ r"""\load_loss"""
            # manage_directory(path=self.loadloss_res_path, description='Load Loss events warning')
            any_event_type_specified = True

        if "line_trip" in event_types_list:
            self.generate_line_trip = True
            # path -- to store the line trip simulation results
            self.linetrip_res_path = generated_events_path #+ r"""\line_trip"""
            # manage_directory(path=self.linetrip_res_path, description='Line Trip events warning')
            any_event_type_specified = True

        if "line_fault" in event_types_list:
            self.generate_line_fault = True
            # path -- to store the line fault simulation results
            self.line_res_path = generated_events_path #+ r"""\line_fault"""
            # manage_directory(path=self.line_res_path, description='Line Fault events warning')
            any_event_type_specified = True

        if "bus_fault" in event_types_list:
            self.generate_bus_fault = True
            # path -- to store the bus fault simulation results
            self.bus_res_path = generated_events_path #+ r"""\bus_fault"""
            # manage_directory(path=self.bus_res_path, description='Bus Fault events warning' )
            any_event_type_specified = True

        if not any_event_type_specified:
            raise ValueError("Specified event types are not supported. \n "
                             "Use the following convention to specify the types of events:\n"
                             "\"generation_loss\", \"load_loss\", \"line_trip\", \"line_fault\", \"bus_fault\"")

        self.psse_set_minor = psse_set_minor
        self.psseinit = psseinit
        self.flat_run_time = flat_run_time
        self.remove_fault_time = remove_fault_time
        self.simulation_time = simulation_time

        self.generate_loading_scenarios = generate_loading_scenarios
        if self.generate_loading_scenarios:
            self.load_scen = loading_scenarios(start_load_level=start_load_level, end_load_level=end_load_level,
                                               n_scenarios=n_scenarios, min_rand=min_rand, max_rand=max_rand)
        self.n_scenarios = n_scenarios


def plot_network(system_components):
    sc = system_components
    busnumbers_temp = map(str, sc.busnumbers)
    buses = list(busnumbers_temp)
    lines = [(str(x), str(y)) for (x,y) in zip(sc.line_from_bus, sc.line_to_bus)]
    trans = [(str(x), str(y)) for (x,y) in zip(sc.trans_from_bus, sc.trans_to_bus)]
    edges = lines + trans
    G = net.Graph()
    G.add_nodes_from(buses)
    G.add_edges_from(edges)
    net.draw(G, with_labels=False)


def get_pmu_bus_nums(sim_setting: object,
                     start_range: int,
                     end_range: int,
                     n_pmus: int):
    """
    :param sim_setting: user defined simulation setting (pmu_bus_select_method, pmu_bus_num)
    :param busnumbers: set(subset) of bus numbers in the network
    :param start_range: first selected bus in the busnumbers list
    :param end_range: last selected bus in the busnumbers list
    :param n_pmus: number of buses with pmu within the (start_range, end_range)
    :return: pmu_bus_num = list of bus numbers with pmus
    """
    if sim_setting.pmu_bus_select_method == "random_in_range":
        step = int(abs((end_range - start_range) / n_pmus))
        pmu_bus_num_temp = list(range(start_range, end_range + 1, step))
        pmu_bus_num = pmu_bus_num_temp[0:n_pmus]
    elif sim_setting.pmu_bus_select_method == "random_in_buses":
        pmu_bus_num = random.sample(sim_setting.busnumbers, n_pmus)
    elif sim_setting.pmu_bus_select_method == "specify_buses":
        if sim_setting.pmu_bus_num == None:
            raise ValueError("Specify list of pmu buses. ")
        else:
            pmu_bus_num = sim_setting.pmu_bus_num
    else:
        raise ValueError("PMU selection method is not defined. ")
    return pmu_bus_num


def get_load_change_prcnt(system_components: object,
                          sim_setting: object):

    """
    This function generates random loading operation condition based on the following parameters
    :param system_components: system components
    :param sim_setting: simulation setting to obtain loading_scenarios parameters
    :return:
    """
    sc = system_components
    lc = sim_setting.load_scen
    load_chng_prcnt = np.zeros((lc.n_scenarios+1, sc.n_loads))
    jj = 1
    for jj in range(1, lc.n_scenarios+1):
        for i in range(sc.n_loads):
                load_chng_prcnt[0, i] = sc.load_bus_num_sorted[i]
                load_chng_prcnt[jj, i] = random.uniform(lc.min_rand, lc.max_rand)
    n_load_levels = np.size(load_chng_prcnt, 0)
    load_change_step = (lc.end_load_level-lc.start_load_level) / n_load_levels
    load_change = np.zeros((lc.n_scenarios + 1, sc.n_loads))
    jj = 1
    for jj in range(1, lc.n_scenarios+1):
        for i in range(sc.n_loads):
                load_change[0, i] = sc.load_bus_num_sorted[i]
                load_change[jj, i] = lc.start_load_level + jj * load_change_step


    return load_change, load_chng_prcnt


def apply_load_change(psspy, system_components, load_change, load_chng_prcnt):
    sc = system_components
    for i in range(len(sc.load_bus_num_sorted)):
        load_tot = psspy.loddt2(sc.load_bus_num_sorted[i], sc.load_id_sorted[i], 'TOTAL', 'ACT')
        psspy.load_chng_5(sc.load_bus_num_sorted[i], sc.load_id_sorted[i], REALAR1=load_tot[1].real*load_change[i]+load_tot[1].real*load_chng_prcnt[i],
                          REALAR2=load_tot[1].imag*load_change[i]+load_tot[1].imag*load_chng_prcnt[i])




def psse_powerflow_fact_tysl(psspy):
    # psspy.fdns([0, 0, 0, 1, 1, 0, 99, 0])
    psspy.fnsl([0, 0, 0, 1, 1, 1, 99, 0])
    psspy.cong(0)
    psspy.conl(0, 1, 1, [0, 0], [0.0, 0.0, 0.0, 0.0])
    psspy.conl(0, 1, 2, [0, 0], [0.0, 0.0, 0.0, 0.0])
    psspy.conl(0, 1, 3, [0, 0], [0.0, 0.0, 0.0, 0.0])
    psspy.fact()
    psspy.tysl(0)



def psse_channel_setup(psspy, pmu_bus_num):
    psspy.bsys(1, 0, [0.0, 0.0], 0, [], 95, pmu_bus_num, 0, [], 0, [])
    # psspy.chsb(1, 0, [-1, -1, -1, 1, 13, 0]) # VOLT, bus pu voltages (complex)
    # psspy.chsb(1, 0, [-1, -1, -1, 1, 1, 0]) # ANGLE, machine relative rotor angle (degrees).
    psspy.chsb(1, 0, [-1, -1, -1, 1, 12, 0]) # BSFREQ, bus pu frequency deviations
    psspy.chsb(1, 0, [-1, -1, -1, 1, 14, 0]) # voltage and angle
    # psspy.chsb(1, 0, [-1, -1, -1, 1, 16, 0]) # flow (P and Q).


def apply_generation_loss(psspy, gen_res_path, gen_number, flat_run_time, simulation_time, scenario_number):
    psspy.strt_2([0, 1], gen_res_path + """\scen-""" + str(scenario_number) + """-gen""" + str(gen_number) + """.out""")
    psspy.run(0, flat_run_time,1200, 4, 4)
    psspy.dist_machine_trip(gen_number, r"""1""")
    psspy.run(0, simulation_time, 1200, 4, 4)


def save_generation_loss(dyntools, pssplot, gen_res_path, gen_number, scenario_number):
    pssplot.newplotbook()
    pssplot.setselectedpage(0)
    achnf = dyntools.CHNF(
        gen_res_path + """\scen-""" + str(scenario_number) + """-gen""" + str(gen_number) + """.out""", outvrsn=0)
    xlsx_flie = gen_res_path + """\scen-""" + str(scenario_number) + """-gen""" + str(gen_number) + """.xlsx"""
    achnf.xlsout(xlsfile=xlsx_flie, show=False)


def save_generation_loss_as_mat(gen_res_path, n_pmus, event_num, gen_number, scenario_number):
    xlsx_file = gen_res_path + """\scen-""" + str(scenario_number) + """-gen""" + str(gen_number) + """.xlsx"""
    mat_file = gen_res_path + """\scen-""" + str(scenario_number) + """-gen""" + str(gen_number) + """.mat"""
    temp_data = pd.read_excel(xlsx_file, skiprows=4)
    temp_data_array = temp_data.to_numpy()
    temp_data_dict = {"result": temp_data_array}
    io.savemat(mat_file, temp_data_dict)
    f = temp_data_array[:, 1:1 + n_pmus]
    vm = temp_data_array[:, 1+n_pmus:-1:2]
    va = temp_data_array[:, 2 + n_pmus::2]
    temp_f_dict = {"f": f}
    temp_vm_dict = {"vm": vm}
    temp_va_dict = {"va": va}
    mat_file_f = gen_res_path   + """\e_f""" + str(event_num) + """.mat"""
    mat_file_vm = gen_res_path   + """\e_vm""" + str(event_num) + """.mat"""
    mat_file_va = gen_res_path   + """\e_va""" + str(event_num) + """.mat"""
    io.savemat(mat_file_f, temp_f_dict)
    io.savemat(mat_file_vm, temp_vm_dict)
    io.savemat(mat_file_va, temp_va_dict)


def apply_line_fault(psspy, line_res_path, branch_from_bus, branch_to_bus, flat_run_time, remove_fault_time, simulation_time,
                     scenario_number):
    psspy.strt_2([0, 1], line_res_path + """\scen-""" + str(scenario_number) + """-line""" + str(
        branch_from_bus) + """-""" + str(branch_to_bus) + """.out""")
    psspy.run(0, flat_run_time, 1200, 4, 4)
    psspy.dist_branch_fault(branch_from_bus, branch_to_bus, r"""1""", 1, 0.0, [0.0, -0.2E+10])
    psspy.run(0, remove_fault_time, 1200, 4, 4)
    psspy.dist_clear_fault(1)
    psspy.run(0, simulation_time, 1200, 4, 4)


def save_line_fault(dyntools, pssplot, line_res_path, branch_from_bus, branch_to_bus, scenario_number):
    pssplot.newplotbook()
    pssplot.setselectedpage(0)
    achnf = dyntools.CHNF(
        line_res_path + """\scen-""" + str(scenario_number) + """-line""" + str(branch_from_bus) + """-""" + str(
            branch_to_bus) + """.out""", outvrsn=0)
    xlsx_flie = line_res_path + """\scen-""" + str(scenario_number) + """-line""" + str(
        branch_from_bus) + """-""" + str(branch_to_bus) + """.xlsx"""
    achnf.xlsout(xlsfile=xlsx_flie, show=False)


def save_line_fault_as_mat(line_res_path, n_pmus, event_num, branch_from_bus, branch_to_bus, scenario_number):
    xlsx_file = line_res_path + """\scen-""" + str(scenario_number) + """-line""" + str(
        branch_from_bus) + """-""" + str(branch_to_bus) + """.xlsx"""
    mat_file = line_res_path + """\scen-""" + str(scenario_number) + """-line""" + str(branch_from_bus) + """-""" + str(
        branch_to_bus) + """.mat"""
    temp_data = pd.read_excel(xlsx_file, skiprows=4)
    temp_data_array = temp_data.to_numpy()
    temp_data_dict = {"result": temp_data_array}
    io.savemat(mat_file, temp_data_dict)
    f = temp_data_array[:, 1:1 + n_pmus]
    vm = temp_data_array[:, 1+n_pmus:-1:2]
    va = temp_data_array[:, 2 + n_pmus::2]
    temp_f_dict = {"f": f}
    temp_vm_dict = {"vm": vm}
    temp_va_dict = {"va": va}
    mat_file_f = line_res_path  + """\e_f""" + str(event_num) + """.mat"""
    mat_file_vm = line_res_path  + """\e_vm""" + str(event_num) + """.mat"""
    mat_file_va = line_res_path  + """\e_va""" + str(event_num) + """.mat"""
    io.savemat(mat_file_f, temp_f_dict)
    io.savemat(mat_file_vm, temp_vm_dict)
    io.savemat(mat_file_va, temp_va_dict)


def apply_line_trip(psspy, linetrip_res_path, branch_from_bus, branch_to_bus, flat_run_time, simulation_time,
                     scenario_number):
    psspy.strt_2([0, 1], linetrip_res_path + """\scen-""" + str(scenario_number) + """-linetrip""" + str(
        branch_from_bus) + """-""" + str(branch_to_bus) + """.out""")
    psspy.run(0, flat_run_time, 1200, 4, 4)
    psspy.dist_branch_trip(branch_from_bus, branch_to_bus, r"""1""")
    psspy.run(0, simulation_time, 1200, 4, 4)

def save_line_trip(dyntools, pssplot, linetrip_res_path, branch_from_bus, branch_to_bus, scenario_number):
    pssplot.newplotbook()
    pssplot.setselectedpage(0)
    achnf = dyntools.CHNF(
        linetrip_res_path + """\scen-""" + str(scenario_number) + """-linetrip""" + str(branch_from_bus) + """-""" + str(
            branch_to_bus) + """.out""", outvrsn=0)
    xlsx_flie = linetrip_res_path + """\scen-""" + str(scenario_number) + """-linetrip""" + str(
        branch_from_bus) + """-""" + str(branch_to_bus) + """.xlsx"""
    achnf.xlsout(xlsfile=xlsx_flie, show=False)

def save_line_trip_as_mat(linetrip_res_path, n_pmus, event_num,  branch_from_bus, branch_to_bus, scenario_number):
    xlsx_file = linetrip_res_path + """\scen-""" + str(scenario_number) + """-linetrip""" + str(
        branch_from_bus) + """-""" + str(branch_to_bus) + """.xlsx"""
    mat_file = linetrip_res_path + """\scen-""" + str(scenario_number) + """-linetrip""" + str(branch_from_bus) + """-""" + str(
        branch_to_bus) + """.mat"""
    temp_data = pd.read_excel(xlsx_file, skiprows=4)
    temp_data_array = temp_data.to_numpy()
    temp_data_dict = {"result": temp_data_array}
    io.savemat(mat_file, temp_data_dict)
    f = temp_data_array[:, 1:1 + n_pmus]
    vm = temp_data_array[:, 1+n_pmus:-1:2]
    va = temp_data_array[:, 2 + n_pmus::2]
    temp_f_dict = {"f": f}
    temp_vm_dict = {"vm": vm}
    temp_va_dict = {"va": va}
    mat_file_f = linetrip_res_path + """\e_f""" + str(event_num) + """.mat"""
    mat_file_vm = linetrip_res_path + """\e_vm""" + str(event_num) + """.mat"""
    mat_file_va = linetrip_res_path + """\e_va""" + str(event_num) + """.mat"""
    io.savemat(mat_file_f, temp_f_dict)
    io.savemat(mat_file_vm, temp_vm_dict)
    io.savemat(mat_file_va, temp_va_dict)

def apply_bus_fault(psspy, bus_res_path, busnumber, flat_run_time, remove_fault_time, simulation_time,
                     scenario_number):
    psspy.strt_2([0, 1], bus_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(
        busnumber) +  """.out""")
    psspy.run(0, flat_run_time, 1200, 4, 4)
    psspy.dist_bus_fault(busnumber,  1)
    psspy.run(0, remove_fault_time, 1200, 4, 4)
    psspy.dist_clear_fault(busnumber)
    psspy.run(0, simulation_time, 1200, 4, 4)

def save_bus_fault(dyntools, pssplot, bus_res_path, busnumber, scenario_number):
    pssplot.newplotbook()
    pssplot.setselectedpage(0)
    achnf = dyntools.CHNF(
        bus_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(busnumber) +  """.out""", outvrsn=0)
    xlsx_flie = bus_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(busnumber) + """.xlsx"""
    achnf.xlsout(xlsfile=xlsx_flie, show=False)

def save_bus_fault_as_mat(bus_res_path,n_pmus, event_num, busnumber, scenario_number):
    xlsx_file = bus_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(busnumber) + """.xlsx"""
    mat_file = bus_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(busnumber)  + """.mat"""
    temp_data = pd.read_excel(xlsx_file, skiprows=4)
    temp_data_array = temp_data.to_numpy()
    temp_data_dict = {"result": temp_data_array}
    io.savemat(mat_file, temp_data_dict)
    f = temp_data_array[:, 1:1 + n_pmus]
    vm = temp_data_array[:, 1+n_pmus:-1:2]
    va = temp_data_array[:, 2 + n_pmus::2]
    temp_f_dict = {"f": f}
    temp_vm_dict = {"vm": vm}
    temp_va_dict = {"va": va}
    mat_file_f = bus_res_path + """\e_f""" + str(event_num) + """.mat"""
    mat_file_vm = bus_res_path + """\e_vm""" + str(event_num) + """.mat"""
    mat_file_va = bus_res_path + """\e_va""" + str(event_num) + """.mat"""
    io.savemat(mat_file_f, temp_f_dict)
    io.savemat(mat_file_vm, temp_vm_dict)
    io.savemat(mat_file_va, temp_va_dict)


# def dist_load_loss(psspy, system_components, load_loss_list):
#     sc = system_components
#     for i in range(len(load_loss_list)):
#         load_tot = psspy.loddt2(sc.load_bus_num[i], sc.load_id[i], 'TOTAL', 'ACT')
#         psspy.load_chng_5(sc.load_bus_num[i], sc.load_id[i], REALAR1=load_tot[1].real * 0.1,
#                           REALAR2=load_tot[1].imag*0.1)

def apply_load_loss(psspy, loadloss_res_path, load_bus_num, load_id, flat_run_time, simulation_time,
                     scenario_number):
    psspy.strt_2([0, 1], loadloss_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(
        load_bus_num) +  """.out""")
    psspy.run(0, flat_run_time, 1200, 4, 4)
    load_tot = psspy.loddt2(load_bus_num, load_id, 'TOTAL', 'ACT')
    psspy.load_chng_5(load_bus_num, load_id, REALAR1=load_tot[1].real * 0.1,
                      REALAR2=load_tot[1].imag * 0.1)
    psspy.run(0, simulation_time, 1200, 4, 4)

def save_load_loss(dyntools, pssplot, loadloss_res_path,  load_bus_num, scenario_number):
    pssplot.newplotbook()
    pssplot.setselectedpage(0)
    print(loadloss_res_path)
    achnf = dyntools.CHNF(
        loadloss_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(load_bus_num) +  """.out""", outvrsn=0)
    xlsx_flie = loadloss_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(load_bus_num) + """.xlsx"""
    print(xlsx_flie)
    # pssplot.channelfileexcelexport(xlsx_flie)
    achnf.xlsout(xlsfile=xlsx_flie, show=False)




def save_load_loss_as_mat(loadloss_res_path, n_pmus, event_num, load_bus_num, scenario_number):
    xlsx_file = loadloss_res_path + """\scen-""" + str(scenario_number) + """-bus""" + str(load_bus_num) + """.xlsx"""
    temp_data = pd.read_excel(xlsx_file, skiprows=4)
    temp_data_array = temp_data.to_numpy()
    f = temp_data_array[:, 1:1 + n_pmus]
    vm = temp_data_array[:, 1+n_pmus:-1:2]
    va = temp_data_array[:, 2 + n_pmus::2]
    temp_f_dict = {"f": f}
    temp_vm_dict = {"vm": vm}
    temp_va_dict = {"va": va}
    mat_file_f =loadloss_res_path + """\e_f""" + str(event_num) + """.mat"""
    mat_file_vm =loadloss_res_path + """\e_vm""" + str(event_num) + """.mat"""
    mat_file_va =loadloss_res_path + """\e_va""" + str(event_num) + """.mat"""
    io.savemat(mat_file_f, temp_f_dict)
    io.savemat(mat_file_vm, temp_vm_dict)
    io.savemat(mat_file_va, temp_va_dict)
    # return temp_data_array


def save_event_labels_as_mat(event_label, project_path):
    # Convert event_label to a NumPy array with 'int' data type
    y_label = np.array(event_label, dtype=int)
    # Define the file path where you want to save the .mat file
    generated_events_path = project_path + """\\""" + "results" + """\\""" + "generated_dataset" +  """\y_data.mat"""
    # Save the NumPy array as a .mat file
    # Print a message indicating that the labels have been saved
    print(f"Saving event labels as {generated_events_path}")
    io.savemat(generated_events_path, {'y_data': y_label})

def manage_directory(path, description):
    # Check if the directory exists
    if os.path.exists(path):
        # Directory exists, inform the user about overwriting
        print("==================================")
        print(description)
        print("==================================")
        response = input(
            "The directory already exists. Existing files will be overwritten. Continue? (y/n): ").strip().lower()

        if response == 'n':
            print("Aborting operation.")

        else:
            try:
                if os.path.isdir(path):
                    os.rmdir(path)
                    print(f"Successfully deleted the folder at {path}")
                else:
                    print(f"{path} is not a directory.")
            except Exception as e:
                print(f"An error occurred while deleting the folder: {str(e)}")

            # Recreate the directory
            os.makedirs(path)

    else:
        # Directory doesn't exist or we want to create it
        os.makedirs(path)



class FeatureExtraction():
    def __init__(self,
                 project_path: str,
                 generated_events_path: str,
                 first_event=1,
                 last_event=5,
                 first_sample=30,
                 last_sample=330,
                 n_pmus=95,
                 n_pmus_prime=20,
                 L=None,
                 Rank=6,
                 Ts=1 / 30,
                 p=6,
                 p_prime=3,
                 decimal_tol=5):
        self.project_path = project_path
        self.generated_events_path = generated_events_path
        self.first_event = first_event
        self.last_event = last_event
        self.first_sample = first_sample
        self.last_sample = last_sample
        self.n_samples = self.last_sample - self.first_sample + 1
        self.mpm_window = np.arange(self.first_sample, self.last_sample + 1)
        self.n_pmus = n_pmus
        self.n_pmus_prime = n_pmus_prime
        self.L = L
        if self.L is None:
            self.L = int(self.n_samples / 2)
        self.Rank = Rank
        self.Ts = Ts
        self.p = p
        self.p_prime = p_prime
        self.decimal_tol = decimal_tol

    def import_raw_data(self):
        """ imports .mat files named; Vm{x}, Va{x}, F{x}, where x is the event number
        Required arguments:
        None
        """
        # import raw .mat data
        os.chdir(self.generated_events_path)
        vm, va, f = {}, {}, {}
        for e in range(self.first_event, self.last_event + 1):
            vm[e] = sio.loadmat(self.generated_events_path + '\e_vm{}.mat'.format(e))["vm"].T
            va_temp = sio.loadmat(self.generated_events_path + '\e_va{}.mat'.format(e))["va"].T
            va[e] = va_temp - va_temp[0]
            f[e] = sio.loadmat(self.generated_events_path + '\e_f{}.mat'.format(e))["f"].T

            print(vm[e])
        # Assuming you have already populated the dictionaries vm, va, and f
        vm_size = len(vm)
        va_size = len(va)
        f_size = len(f)

        print("Size of vm dictionary:", vm_size)
        print("Size of va dictionary:", va_size)
        print("Size of f dictionary:", f_size)

        # Show the number of rows and columns in each vm[e] array
        for e in vm:
            e_array = vm[e]
            n_rows, n_cols = e_array.shape
            print(f"vm[{e}] array has {n_rows} rows and {n_cols} columns")

        return vm, va, f

    def detrend_raw_data(self, vm, va, f):
        """ detrend input time series signal
        Required arguments:
        vm -- voltage magnitude signal
        va -- voltage angle signal
        f -- frequency signal
        """
        de_vm, de_va, de_f = [], [], []
        for e in range(self.first_event, self.last_event + 1):
            print("----------------")
            print(e)
            print("----------------")
            de_vm.append(signal.detrend(vm[e], axis=1))
            de_va.append(signal.detrend(va[e], axis=1))
            de_f.append(signal.detrend(f[e], axis=1))
        return de_vm, de_va, de_f

    def define_data_window(self, vm, va, f):
        """ define data window for feature extraction
        Required arguments:
        vm -- voltage magnitude signal
        va -- voltage angle signal
        f -- frequency signal
        """
        x_vm, x_va, x_f = [], [], []
        for e in range(self.first_event -1, self.last_event):
            print(e)
            print(vm[e][:, self.first_sample - 1:self.last_sample])
            x_vm.append(vm[e][:, self.first_sample - 1:self.last_sample])
            x_va.append(va[e][:, self.first_sample - 1:self.last_sample])
            x_f.append(f[e][:, self.first_sample - 1:self.last_sample])
        return x_vm, x_va, x_f

    def energy_sort(self, x):
        e_energy, e_idx = [], []
        for e in range(self.first_event-1, self.last_event):
            energy = np.square(x[e]).sum(axis=1)
            idx = np.argsort(-energy)
            e_energy.append(energy)
            e_idx.append(idx)
        return e_energy, e_idx


    def Hankel(self, timeseries):
        """ construct the hankle matrix from a timeseries signal
        Required arguments:
        timeseries -- input timeseries signal
        L -- pencil parameter
        """
        K = len(timeseries) - self.L + 1
        hankelized = hankel(timeseries, np.zeros(K)).T
        hankelized = hankelized[:, :self.L]
        return hankelized

    def MPM(self, xm, ch_name, idx):
        """ perform matrix pencil modal analysis
        Required arguments:
        xm -- list of timeseries signals for events and PMUs
        ch -- PMU channel in xm
        """
        x_vector = np.zeros(
            (self.last_event - (self.first_event - 1), 1 + 2 * self.p_prime + 2 * self.p_prime * self.n_pmus_prime))
        e_ia = []
        # e_freq, e_alfa, e_Amp, e_theta = [], [], [], []
        for e in range(self.first_event - 1, self.last_event):
            print(ch_name)
            print('event {0}'.format(e))
            Y = np.empty((0, self.L))  # Empty matrix to build the Hankel matrix based on different PMU measurements
            X = []
            for m in range(self.n_pmus):
                Y_temp = self.Hankel(xm[e][m, :])
                Y = np.append(Y, Y_temp, axis=0)
                X_temp = xm[e][m, :].T
                X = np.append(X, X_temp, axis=0)

            u, s, vh = np.linalg.svd(Y, full_matrices=False)
            sr = np.diag(s[:self.Rank])
            ur = u[:, :self.Rank]
            vhr = vh[:self.Rank, :]
            Yr = np.matrix(u[:, :self.Rank]) * np.diag(s[:self.Rank]) * np.matrix(vh[:self.Rank, :])
            UrSr = np.matrix(u[:, :self.Rank]) * np.diag(s[:self.Rank])
            Xm_T = np.transpose(xm[e])

            # Using only right singular vectors of Y_r
            vr = vhr.T
            V_r_1 = vr[0:self.L - 1, 0:self.Rank]
            V_r_2 = vr[1:, 0:self.Rank]
            H1 = np.matrix(np.transpose(V_r_1)) * np.matrix(V_r_1)
            H2 = np.matrix(np.transpose(V_r_2)) * np.matrix(V_r_1)
            # H1 = Yr[0:-1,:]
            # H2 = Yr[1:,:]
            l_temp = np.linalg.eig(np.linalg.pinv(H1) * H2)
            l = l_temp[0]
            alfa = np.log(np.abs(l)) / self.Ts
            freq = np.arctan2(np.imag(l), np.real(l)) / (2 * np.pi * self.Ts)
            Lambda = np.log(l) / self.Ts
            # e_freq.append(freq)
            # e_alfa.append(alfa)

            Z = np.zeros((self.n_samples, len(l)), dtype='complex_')
            N_vector = np.arange(0, self.n_samples)
            for i in range(len(l)):
                Z[:, i] = np.transpose(np.power(l[i], N_vector))

            h = np.zeros((len(l), self.n_pmus), dtype='complex_')
            Amp = np.zeros((len(l), self.n_pmus))
            theta = np.zeros((len(l), self.n_pmus))

            for m in range(self.n_pmus):
                h[:, m] = np.linalg.lstsq(Z, xm[e][m, :].T)[0]
                Amp[:, m] = abs(h[:, m])
                theta[:, m] = np.angle(h[:, m])

            x_vector[e, 0] = e
            amp_unique = -np.sort(-abs(Amp).mean(1))
            ia0, ia_temp = np.zeros(2 * self.p_prime).astype(int), np.zeros(2 * self.p_prime).astype(int)
            for i in range(2 * self.p_prime):
                ia0[i] = np.argwhere(abs(Amp).mean(1) == amp_unique[i])[0]
                amp_unique_temp = np.around(amp_unique / amp_unique[i], decimals=self.decimal_tol)
                temp = np.argwhere(amp_unique_temp == 1)
                ia_temp[i] = temp[0]
            ia = (ia0[np.unique(ia_temp)])[0:3]

            amp_zeros, theta_zeros = np.zeros((self.p_prime, self.n_pmus_prime)), np.zeros(
                (self.p_prime, self.n_pmus_prime))
            for i in range(self.p_prime):
                amp_sort = Amp[ia[i], idx[e]]
                amp_zeros[i, 0:self.n_pmus_prime] = amp_sort[0:self.n_pmus_prime]
            x_vector[e, 1:self.p_prime + 1] = abs(freq[ia])
            x_vector[e, self.p_prime + 1:2 * self.p_prime + 1] = alfa[ia]
            amp_sort_idx = idx[e]
            theta_sort = np.zeros((len(ia), len(amp_sort_idx)))

            for m in range(len(ia)):
                for n in range(len(amp_sort_idx)):
                    theta_sort[m, n] = theta[ia[m], amp_sort_idx[n]]
            theta_zeros[:, 0:self.n_pmus_prime] = theta_sort[:, 0:self.n_pmus_prime]
            x_vector[e, 2 * self.p_prime + 1:2 * self.p_prime + 3 * self.n_pmus_prime + 1] = amp_zeros.flatten()
            x_vector[e,
            2 * self.p_prime + 3 * self.n_pmus_prime + 1:2 * self.p_prime + 6 * self.n_pmus_prime + 1] = theta_zeros.flatten()
            e_ia.append(ia)

        return x_vector, e_ia

    def return_event_features_matrix(self, x_vm_features, x_va_features, x_f_features):
        x_data = np.append(
            np.append(x_vm_features[:, 1:(2 * self.p_prime + 1) + (6 * self.n_pmus_prime)], x_va_features[:, 1:(2 * self.p_prime + 1) + (6 * self.n_pmus_prime)], axis=1),
            x_f_features[:, 1:(2 * self.p_prime + 1) + (6 * self.n_pmus_prime)], axis=1)
        return x_data

    def save_event_features_matrix_as_mat(self, x_data):
        # Define the file path where you want to save the .mat file
        generated_events_path = self.project_path + r"\results\generated_dataset\x_data.mat"
        # Print a message indicating that the labels have been saved
        print(f"Saving events features matrix as {generated_events_path}")
        io.savemat(generated_events_path, {'x_data': x_data})


def generate_circle_dataset(n_samples, n_classes, noise_prcnt, noise_std, plot_dataset=False):
    # Number of samples per class
    n_samples_per_class = n_samples // n_classes  # Divide total samples evenly among classes
    radius_min, radius_max = 0.0, 1.0  # Update the radius range to [0, 1]
    # Calculate the step size for equally distributing the radii
    radius_step = (radius_max - radius_min) / (n_classes - 1)

    # Generate radius values for each class
    radii = np.arange(radius_min, radius_max + 2 * radius_step, radius_step)
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    radii = scaler.fit_transform(radii.reshape(-1, 1))
    # Generate random angles for each class
    angles = np.linspace(0, 2 * np.pi, n_samples_per_class)

    # Calculate x and y coordinates for each class with noise
    x_data_temp = np.zeros((n_classes * n_samples_per_class,))
    y_data_temp = np.zeros((n_classes * n_samples_per_class,))
    for i in range(n_classes):
        noise_scale = (noise_prcnt / 100) * radius_step  # Scale noise based on the radius
        x_noise = np.random.normal(0, noise_scale, n_samples_per_class)
        y_noise = np.random.normal(0, noise_scale, n_samples_per_class)
        x_data_temp[i * n_samples_per_class:(i + 1) * n_samples_per_class] = radii[i] * np.cos(angles) + x_noise
        y_data_temp[i * n_samples_per_class:(i + 1) * n_samples_per_class] = radii[i] * np.sin(angles) + y_noise

    # Add additional noise to coordinates
    additional_noise_x = np.random.normal(0, noise_std, n_classes * n_samples_per_class)
    additional_noise_y = np.random.normal(0, noise_std, n_classes * n_samples_per_class)
    x_data_temp += additional_noise_x
    y_data_temp += additional_noise_y

    # Create the feature vectors and labels
    x_data = np.column_stack((x_data_temp, y_data_temp))
    y_data = np.repeat(np.arange(n_classes), n_samples_per_class)

    # Plot the dataset
    if plot_dataset:
        for i in range(n_classes):
            plt.plot(x_data[y_data == i, 0], x_data[y_data == i, 1], marker="o", linestyle="", label=f"Class {i + 1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Classes around Perimeters of Circles")
        plt.legend()
        plt.show()

    return x_data, y_data



class Dataset:
    def __init__(self,
                 dataset_name,
                 x_data_path,
                 y_data_path,
                 class_labels,
                 selected_class_labels,
                 shuffle_data=True,
                 undersampling_method=None,
                 random_state=42):
        """
        Initializes a Dataset object.

        Parameters:
        - dataset_name (str): Name of the dataset.
        - x_data_path (str): Path to the feature data file (e.g., x_data.mat).
        - y_data_path (str): Path to the label data file (e.g., y_data.mat).
        - class_labels (list): List of all possible class labels in the dataset.
        - selected_class_labels (list): List of selected class labels to include in the dataset.
        - shuffle_data (bool): Whether to shuffle the data.
        - undersampling_method (object): Undersampling method to apply to the data.
        - random_state (int): Random seed for data shuffling.

        Attributes:
        - dataset_name (str): Name of the dataset.
        - x_data_path (str): Path to the feature data file.
        - y_data_path (str): Path to the label data file.
        - class_labels (list): List of all possible class labels in the dataset.
        - selected_class_labels (list): List of selected class labels included in the dataset.
        - shuffle_data (bool): Whether to shuffle the data.
        - undersampling_method (object): Undersampling method to apply to the data.
        - random_state (int): Random seed for data shuffling.
        - n_classes (int): Number of classes in the dataset.
        - x_data_orig (numpy.ndarray): Original feature data.
        - y_data_orig (numpy.ndarray): Original label data.
        - data_orig_object (DataAnalysis): Data analysis object for original data.
        - x_data (numpy.ndarray): Processed feature data.
        - y_data (numpy.ndarray): Processed label data.
        - n_samples (int): Number of samples in the dataset.
        - data_object (DataAnalysis): Data analysis object for processed data.
        - class_labels_list (str): String representation of selected class labels.

        Returns:
        None
        """
        self.dataset_name = dataset_name
        self.x_data_path = x_data_path
        self.y_data_path = y_data_path
        self.class_labels = class_labels
        self.selected_class_labels = selected_class_labels
        self.shuffle_data = shuffle_data
        self.undersampling_method = undersampling_method
        self.random_state = random_state
        self.n_classes = len(self.class_labels)

    def _apply_resampling(self, x_data, y_data):
        """
        Applies undersampling method and shuffling to the data.

        Parameters:
        - x_data (numpy.ndarray): Feature data.
        - y_data (numpy.ndarray): Label data.

        Returns:
        - x_data (numpy.ndarray): Resampled and shuffled feature data.
        - y_data (numpy.ndarray): Resampled and shuffled label data.
        """
        if self.undersampling_method is not None:
            select_samples = self.undersampling_method
            x_data, y_data = select_samples.fit_resample(x_data, y_data)

        if self.shuffle_data:
            x_data, y_data = shuffle(x_data, y_data, random_state=self.random_state)

        return x_data, y_data

    def _load_dataset(self):
        """
        Loads the dataset from files, applies resampling, and performs data analysis.

        Returns:
        - x_data (numpy.ndarray): Resampled and shuffled feature data.
        - y_data (numpy.ndarray): Resampled and shuffled label data.
        - x_data_orig (numpy.ndarray): Original feature data.
        - y_data_orig (numpy.ndarray): Original label data.
        - data_object (DataAnalysis): Data analysis object for processed data.
        - data_orig_object (DataAnalysis): Data analysis object for original data.
        - class_labels_list (str): String representation of selected class labels.
        """
        self.class_labels_list = " - ".join(map(str, self.selected_class_labels))
        tempx_data = loadmat(self.x_data_path)
        tempy_data = loadmat(self.y_data_path)
        x_data_temp = tempx_data["x_data"]
        y_data_temp = tempy_data["y_data"].reshape(-1).astype(int)

        x_data_subset = []
        y_data_subset = []

        for class_name in self.selected_class_labels:
            x_data_subset.append(x_data_temp[y_data_temp == class_name])
            y_data_subset.append(y_data_temp[y_data_temp == class_name])

        self.x_data_orig = np.vstack(x_data_subset)
        self.y_data_orig = np.concatenate(y_data_subset, axis=0)
        self.data_orig_object = DataAnalysis(self.x_data_orig, self.y_data_orig)

        self.x_data, self.y_data = self._apply_resampling(self.x_data_orig, self.y_data_orig)
        self.n_samples = len(self.y_data)
        self.data_object = DataAnalysis(self.x_data, self.y_data)

        return self.x_data, self.y_data, self.x_data_orig, self.y_data_orig, self.data_object, self.data_orig_object, \
               self.class_labels_list




class SemiSupSettings:
    """
    A class for storing settings related to semi-supervised learning experiments.

    Args:
        version (str, optional): Simulation version. It is used in the filenames that will be saved. Default is "1.0".
        n_samples (int, optional): Number of samples.
        n_classes (int, optional): Number of classes.
        n_folds (int, optional): Number of folds for cross-validation.
        L_prcnt (float, optional): Percentage of samples to consider as labeled.
        step_to_unlabeled_as_labeled (int, optional): Number of steps to consider unlabeled samples as labeled.
        filter_mi (bool, optional): Flag indicating whether to use the mutual information filter.
        number_of_selected_features (int, optional): Number of features to be selected.
        number_of_bootstrap_in_filter_method (int, optional): Number of bootstraps in the feature selection step.
        number_of_pca (int, optional): Number of principal components for the low-dimensional representation.
        n_sets (int, optional): Number of random splits of the training dataset into labeled and unlabeled samples.
        n_rand_max (int, optional): Maximum number of random seeds.
        sort_l_u_distance (bool, optional): Flag indicating whether to sort labeled and unlabeled samples by distance.
        class_balance (bool, optional): Flag indicating whether to balance the classes. Default is True.
        balance_range (tuple, optional): Range of class balance percentages. Default is (0.2, 0.8).
        plot_animation (bool, optional): Flag indicating whether to plot animations. Default is False.
        hyperparameter_tuning (bool, optional): Flag indicating whether to perform hyperparameter tuning. Default is False.
        print_results (bool, optional): Flag indicating whether to print results. Default is False.
        print_idx_n_mix_rand (bool, optional): Flag indicating whether to print index, number of mixed samples, and random seed. Default is True.
        pseudo_label_model_list (list, optional): List of pseudo-label models. Default is ["lp"].
        test_model_list (list, optional): List of test models. Default is ["svmrbf"].
        plt_datatypes_together (bool, optional): Flag indicating whether to plot data types together. Default is False.
        plt_auc_type (list, optional): List of AUC types. Default is ["pl_auc", "t_auc"].
        plt_y_lim (list, optional): List of y-axis limits for plots. Default is [0, 1].
        plt_auc_calc_type (list, optional): List of AUC calculation types. Default is ["ave"].
        plt_auc_stat_type (list, optional): List of AUC statistic types. Default is ["ave", "5th", "95th"].
        plt_auc_stat_style (list, optional): List of AUC statistic styles. Default is ["lines"].
        plt_auc_all (bool, optional): Flag indicating whether to plot all AUCs. Default is True.
        plt_add_fold_details (bool, optional): Flag indicating whether to add fold details to plots. Default is False.
        plt_centroids (bool, optional): Flag indicating whether to plot centroids. Default is False.
        line_style_list (list, optional): List of line styles. Default is ["", ":", "-", "--", "-."].
        color_list (list, optional): List of colors. Default is ["b", "r", "g", "c", "m", "y", "k", "w"].
        marker_list (list, optional): List of markers. Default is ["", "o", "v", "^", "<", ">", "s", "p", "*"].
        line_width_list (list, optional): List of line widths. Default is [1, 2, 3, 4, 5].
        marker_size_list (list, optional): List of marker sizes. Default is [1, 2, 3, 4, 5].
        random_state (int, optional): Random state for reproducibility. Default is 47.

    """

    def __init__(self,
                 version=None,
                 n_samples=100,
                 n_classes=2,
                 n_folds=3,
                 L_prcnt=0.2,
                 step_to_unlabeled_as_labeled=5,
                 filter_mi=True,
                 number_of_selected_features=2,
                 number_of_bootstrap_in_filter_method=1,
                 number_of_pca=2,
                 n_sets=1,
                 n_rand_max=2,
                 sort_l_u_distance=False,
                 class_balance=True,
                 balance_range=(0.2, 0.8),
                 plot_animation=False,
                 hyperparameter_tuning=False,
                 print_results=False,
                 print_idx_n_mix_rand=True,
                 pseudo_label_model_list=None,
                 test_model_list=None,
                 plt_datatypes_together=False,
                 plt_auc_type=None,
                 plt_y_lim=None,
                 plt_auc_calc_type=None,
                 plt_auc_stat_type=None,
                 plt_auc_stat_style=None,
                 plt_auc_all=True,
                 plt_add_fold_details=False,
                 plt_centroids=False,
                 line_style_list=None,
                 color_list=None,
                 marker_list=None,
                 line_width_list=None,
                 marker_size_list=None,
                 random_state=47):
        """
        Initializes a SemiSupSettings object with various settings for semi-supervised learning experiments.

        Parameters:
        - version (str, optional): Simulation version. It is used in the filenames that will be saved. Default is "1.0".
        - n_samples (int, optional): Number of samples.
        - n_classes (int, optional): Number of classes.
        - n_folds (int, optional): Number of folds for cross-validation.
        - L_prcnt (float, optional): Percentage of samples to consider as labeled.
        - step_to_unlabeled_as_labeled (int, optional): Number of steps to consider unlabeled samples as labeled.
        - filter_mi (bool, optional): Flag indicating whether to use the mutual information filter.
        - number_of_selected_features (int, optional): Number of features to be selected.
        - number_of_bootstrap_in_filter_method (int, optional): Number of bootstraps in the feature selection step.
        - number_of_pca (int, optional): Number of principal components for the low-dimensional representation.
        - n_sets (int, optional): Number of random splits of the training dataset into labeled and unlabeled samples.
        - n_rand_max (int, optional): Maximum number of random seeds.
        - sort_l_u_distance (bool, optional): Flag indicating whether to sort labeled and unlabeled samples by distance.
        - class_balance (bool, optional): Flag indicating whether to balance the classes. Default is True.
        - balance_range (tuple, optional): Range of class balance percentages. Default is (0.2, 0.8).
        - plot_animation (bool, optional): Flag indicating whether to plot animations. Default is False.
        - hyperparameter_tuning (bool, optional): Flag indicating whether to perform hyperparameter tuning. Default is False.
        - print_results (bool, optional): Flag indicating whether to print results. Default is False.
        - print_idx_n_mix_rand (bool, optional): Flag indicating whether to print index, number of mixed samples, and random seed. Default is True.
        - pseudo_label_model_list (list, optional): List of pseudo-label models. Default is ["lp"].
        - test_model_list (list, optional): List of test models. Default is ["svmrbf"].
        - plt_datatypes_together (bool, optional): Flag indicating whether to plot data types together. Default is False.
        - plt_auc_type (list, optional): List of AUC types. Default is ["pl_auc", "t_auc"].
        - plt_y_lim (list, optional): List of y-axis limits for plots. Default is [0, 1].
        - plt_auc_calc_type (list, optional): List of AUC calculation types. Default is ["ave"].
        - plt_auc_stat_type (list, optional): List of AUC statistic types. Default is ["ave", "5th", "95th"].
        - plt_auc_stat_style (list, optional): List of AUC statistic styles. Default is ["lines"].
        - plt_auc_all (bool, optional): Flag indicating whether to plot all AUCs. Default is True.
        - plt_add_fold_details (bool, optional): Flag indicating whether to add fold details to plots. Default is False.
        - plt_centroids (bool, optional): Flag indicating whether to plot centroids. Default is False.
        - line_style_list (list, optional): List of line styles. Default is ["", ":", "-", "--", "-."].
        - color_list (list, optional): List of colors. Default is ["b", "r", "g", "c", "m", "y", "k", "w"].
        - marker_list (list, optional): List of markers. Default is ["", "o", "v", "^", "<", ">", "s", "p", "*"].
        - line_width_list (list, optional): List of line widths. Default is [1, 2, 3, 4, 5].
        - marker_size_list (list, optional): List of marker sizes. Default is [1, 2, 3, 4, 5].
        - random_state (int, optional): Random state for reproducibility. Default is 47.
        """
        if marker_size_list is None:
            marker_size_list = [1, 2, 3, 4, 5]
        if line_width_list is None:
            line_width_list = [1, 2, 3, 4, 5]
        if marker_list is None:
            marker_list = ["", "o", "v", "^", "<", ">", "s", "p", "*"]
        if color_list is None:
            color_list = ["b", "r", "g", "c", "m", "y", "k", "w"]
        if line_style_list is None:
            line_style_list = ["", ":", "-", "--", "-."]
        if plt_auc_stat_style is None:
            plt_auc_stat_style = ["lines"]
        if plt_auc_stat_type is None:
            plt_auc_stat_type = ["ave", "5th", "95th"]#["ave", "min", "max", "5th", "95th"]
        if plt_auc_calc_type is None:
            plt_auc_calc_type = ["ave"] #["ave", "ind"]
        if plt_y_lim is None:
            plt_y_lim = [0, 1]
        if plt_auc_type is None:
            plt_auc_type = ["pl_auc", "t_auc"]
        if test_model_list is None:
            test_model_list = ["svmrbf", "svmlin", "knn", "gb", "dt"]
        if pseudo_label_model_list is None:
            pseudo_label_model_list = ["lp", "ls", "tsvm", "svmrbf", "svmlin", "knn", "dt", "gb"]

        self.version = version
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.L_prcnt = L_prcnt
        self.step_to_unlabeled_as_labeled = step_to_unlabeled_as_labeled
        self.filter_mi = filter_mi
        self.number_of_selected_features = number_of_selected_features
        self.number_of_bootstrap_in_filter_method = number_of_bootstrap_in_filter_method
        self.number_of_pca = number_of_pca
        self.n_sets = n_sets
        self.n_rand_max = n_rand_max
        self.sort_l_u_distance = sort_l_u_distance
        self.class_balance = class_balance
        self.balance_range = balance_range
        self.plot_animation = plot_animation
        self.hyperparameter_tuning = hyperparameter_tuning
        self.print_results = print_results
        self.print_idx_n_mix_rand = print_idx_n_mix_rand
        self.pseudo_label_model_list = pseudo_label_model_list
        self.test_model_list = test_model_list
        self.plt_datatypes_together = plt_datatypes_together
        self.plt_auc_type = plt_auc_type
        self.plt_y_lim = plt_y_lim
        self.plt_auc_calc_type = plt_auc_calc_type
        self.plt_auc_stat_type = plt_auc_stat_type
        self.plt_auc_stat_style = plt_auc_stat_style
        self.plt_auc_all = plt_auc_all
        self.plt_add_fold_details = plt_add_fold_details
        self.plt_centroids = plt_centroids
        self.line_style_list = line_style_list
        self.color_list = color_list
        self.marker_list = marker_list
        self.line_width_list = line_width_list
        self.marker_size_list = marker_size_list
        self.random_state = random_state
        self.n_train, self.n_test, self.n_labeled, self.n_unlabeled = self.get_dataset_stats()

    def get_dataset_stats(self):
        n_train = (self.n_folds - 1) * self.n_samples // self.n_folds
        n_test = self.n_samples // self.n_folds
        n_labeled = int(n_train * self.L_prcnt)
        n_unlabeled = n_train - n_labeled
        return n_train, n_test, n_labeled, n_unlabeled

    def get_plot_style(self, auc_calc_type, auc_type, auc_stat_type, class_num=None):
        if auc_calc_type == "ind":

            if auc_type == "t_auc":
                if auc_stat_type == "ave":
                    line, color, marker, width, size = self.line_style_list[4], self.color_list[class_num], \
                                                       self.marker_list[0], self.line_width_list[1], \
                                                       self.marker_size_list[0]

                elif auc_stat_type == "min":
                    line, color, marker, width, size = self.line_style_list[1], self.color_list[class_num], \
                                                       self.marker_list[3], self.line_width_list[0], \
                                                       self.marker_size_list[1]
                elif auc_stat_type == "max":
                    line, color, marker, width, size = self.line_style_list[1], self.color_list[class_num], \
                                                       self.marker_list[2], self.line_width_list[0], \
                                                       self.marker_size_list[1]
                elif auc_stat_type == "5th":
                    line, color, marker, width, size = self.line_style_list[3], self.color_list[class_num], \
                                                       self.marker_list[4], self.line_width_list[0], \
                                                       self.marker_size_list[1]
                elif auc_stat_type == "95th":
                    line, color, marker, width, size = self.line_style_list[3], self.color_list[class_num], \
                                                       self.marker_list[2], self.line_width_list[0], \
                                                       self.marker_size_list[1]


            elif auc_type == "pl_auc":
                if auc_stat_type == "ave":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[class_num], \
                                                       self.marker_list[-1], self.line_width_list[0], \
                                                       self.marker_size_list[4]
                elif auc_stat_type == "min":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[class_num], \
                                                       self.marker_list[3], self.line_width_list[0], \
                                                       self.marker_size_list[4]
                elif auc_stat_type == "max":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[class_num], \
                                                       self.marker_list[2], self.line_width_list[0], \
                                                       self.marker_size_list[4]
                elif auc_stat_type == "5th":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[class_num], \
                                                       self.marker_list[3], self.line_width_list[0], \
                                                       self.marker_size_list[4]
                elif auc_stat_type == "95th":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[class_num], \
                                                       self.marker_list[2], self.line_width_list[0], \
                                                       self.marker_size_list[4]


        elif auc_calc_type == "ave":
            if auc_type == "t_auc":
                if auc_stat_type == "ave":
                    line, color, marker, width, size = self.line_style_list[2], self.color_list[6], \
                                                       self.marker_list[0], self.line_width_list[3], \
                                                       self.marker_size_list[0]

                elif auc_stat_type == "min":
                    line, color, marker, width, size = self.line_style_list[1], self.color_list[6], \
                                                       self.marker_list[3], self.line_width_list[2], \
                                                       self.marker_size_list[2]
                elif auc_stat_type == "max":
                    line, color, marker, width, size = self.line_style_list[1], self.color_list[6], \
                                                       self.marker_list[2], self.line_width_list[2], \
                                                       self.marker_size_list[2]
                elif auc_stat_type == "5th":
                    line, color, marker, width, size = self.line_style_list[1], self.color_list[6], \
                                                       self.marker_list[3], self.line_width_list[2], \
                                                       self.marker_size_list[2]

                elif auc_stat_type == "95th":
                    line, color, marker, width, size = self.line_style_list[3], self.color_list[6], \
                                                       self.marker_list[2], self.line_width_list[2], \
                                                       self.marker_size_list[2]


            elif auc_type == "pl_auc":
                if auc_stat_type == "ave":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[5], \
                                                       self.marker_list[-1], self.line_width_list[0], \
                                                       self.marker_size_list[3]

                elif auc_stat_type == "min":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[5], \
                                                       self.marker_list[3], self.line_width_list[0], \
                                                       self.marker_size_list[4]
                elif auc_stat_type == "max":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[5], \
                                                       self.marker_list[2], self.line_width_list[0], \
                                                       self.marker_size_list[4]

                elif auc_stat_type == "5th":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[5], \
                                                       self.marker_list[3], self.line_width_list[0], \
                                                       self.marker_size_list[4]

                elif auc_stat_type == "95th":
                    line, color, marker, width, size = self.line_style_list[0], self.color_list[5], \
                                                       self.marker_list[2], self.line_width_list[0], \
                                                       self.marker_size_list[4]


        return {
            "linestyle": line,
            "color": color,
            "marker": marker,
            "linewidth": width,
            "markersize": size
        }


class DataAnalysis:
    """
    A class for performing data analysis and visualization on a dataset.

    Args:
        x_data (numpy.ndarray): The feature data.
        y_data (numpy.ndarray): The target labels.
        idx_features (list, optional): List of indices of the selected features. Default is None.
    """

    def __init__(self, x_data, y_data, idx_features=None):
        """
        Initializes a DataAnalysis object with feature and target data.

        Parameters:
        - x_data (numpy.ndarray): The feature data.
        - y_data (numpy.ndarray): The target labels.
        - idx_features (list, optional): List of indices of the selected features. Default is None.
        """
        if y_data.dtype != np.uint8:
            y_data = y_data.astype(np.uint8)
        self.y_data = y_data

        if idx_features is None:
            self.idx_features = list(range(0, np.shape(x_data)[1]))
        else:
            self.idx_features = idx_features
        self.x_data = x_data[:, self.idx_features]

    def data_gen_stats(self):
        """
        Generate and print basic statistics for the feature data.
        """
        df = pd.DataFrame(self.x_data)
        statistics = df.describe()
        print(statistics)

    def data_heatmap(self):
        """
        Generate and save a heatmap of the feature data's correlation.
        """
        x_data_df = pd.DataFrame(self.x_data)
        fig = px.imshow(x_data_df.corr())
        fig.write_html("heatmap.html", auto_open=True)

    def data_histogram(self):
        """
        Generate and save histograms of the feature data.
        """
        x_data_df = pd.DataFrame(self.x_data)
        fig = px.histogram(x_data_df, nbins=20, histnorm='probability density')
        fig.write_html("histogram.html", auto_open=True)

    def data_pca_plot_v1(self):
        """
        Generate and save a PCA plot for the feature data.
        """
        d_data = np.column_stack((self.x_data, self.y_data))
        df_data = pd.DataFrame(d_data)
        df = df_data
        df.columns = df.columns.map(str)
        features = [str(idx_i) for idx_i in range(len(self.idx_features))]
        target = str(len(self.idx_features))
        fig = px.scatter_matrix(df, dimensions=features, color=target)
        fig.write_html("pca_plot.html", auto_open=True)

    def data_feature_statistics(self):
        """
        Generate and save box plots for feature statistics.
        """
        x_data_df = pd.DataFrame(self.x_data)
        fig = go.Figure()
        fig.add_trace(go.Box(y=x_data_df.mean(), name='Mean'))
        fig.add_trace(go.Box(y=x_data_df.var(), name='Variance'))
        fig.write_html("feature_statistics.html", auto_open=True)

    def feature_box_plot(self):
        """
        Generate and save box plots for individual features.
        """
        x_data_df = pd.DataFrame(self.x_data)
        fig = go.Figure()
        for i in range(x_data_df.shape[1]):
            fig.add_trace(go.Box(y=x_data_df.iloc[:, i], name=f'Feature {i + 1}'))
        fig.write_html("box_plot.html", auto_open=True)

    def get_multiclass_stats(self):
        """
        Compute and print statistics for multiclass data.
        """
        df = pd.DataFrame(self.x_data)
        df["Target"] = self.y_data
        class_counts = df["Target"].value_counts()
        class_percentages = df["Target"].value_counts(normalize=True) * 100
        class_means = df.groupby("Target").mean()
        class_stds = df.groupby("Target").std()
        class_min = df.groupby("Target").min()
        class_max = df.groupby("Target").max()

        # Display stats
        print("Class Counts:\n", class_counts, "\n")
        print("Class Percentages:\n", class_percentages, "\n")
        print("Class Means:\n", class_means, "\n")
        print("Class Standard Deviations:\n", class_stds, "\n")
        print("Class Minimum Values:\n", class_min, "\n")
        print("Class Maximum Values:\n", class_max, "\n")

        return class_counts, class_percentages

    def data_pca_plot_v2(self):
        """
        Generate and save a 2D PCA plot.
        """
        pca = PCA(n_components=2)
        x_data_pca = pca.fit_transform(self.x_data)
        df_pca = pd.DataFrame(data=x_data_pca, columns=['PC1', 'PC2'])
        df_pca['y'] = self.y_data

        fig = px.scatter(df_pca, x="PC1", y="PC2", color="y",
                         title='2D PCA plot',
                         labels={'PC1': '1st Principal Component', 'PC2': '2nd Principal Component'},
                         hover_data=[df_pca.index])
        class_counts, class_percentages = self.get_multiclass_stats()
        class_labels = class_counts.keys().tolist()

        class_counts = class_counts.tolist()
        class_percentages = class_percentages.tolist()
        total_samples = len(self.y_data)

        title = f'PCA Plot\nTotal Samples: {total_samples}\n'
        for i, count in enumerate(class_counts):
            percentage = class_percentages[i]
            title += f'C{class_labels[i]}: {count} ({percentage:.2f}%)\n'
        fig.update_layout(title=title)
        fig.write_html("PCA.html", auto_open=True)

    def data_tsne_plot(self, fig_name=None):
        """
        Generate and save a t-SNE plot.
        """
        tsne = TSNE(n_components=2)
        x_data_tsne = tsne.fit_transform(self.x_data)
        df_tsne = pd.DataFrame(data=x_data_tsne, columns=['Dim1', 'Dim2'])
        df_tsne['y'] = self.y_data

        fig = px.scatter(df_tsne, x="Dim1", y="Dim2", color="y",
                         labels={'Dim1': 'Dimension 1', 'Dim2': 'Dimension 2'},
                         hover_data=[df_tsne.index])
        class_counts, class_percentages = self.get_multiclass_stats()
        class_labels = class_counts.keys().tolist()

        class_counts = class_counts.tolist()
        class_percentages = class_percentages.tolist()
        total_samples = len(self.y_data)

        title = f't-SNE plot\nTotal Samples: {total_samples}\n'
        for i, count in enumerate(class_counts):
            percentage = class_percentages[i]
            title += f'C{class_labels[i]}: {count} ({percentage:.2f}%)\n'
        fig.update_layout(title=title)
        if fig_name is None:
            fig_name = 'tSNE'
        fig_name = fig_name + '.html'
        fig.write_html(fig_name, auto_open=True)
        return x_data_tsne

    def plot_distance_heatmap(self):
        """
        Generate and save a heatmap of pairwise distances.
        """
        distances = pairwise_distances(self.x_data)
        fig = px.imshow(distances)
        fig.write_html("distance_heatmap.html", auto_open=True)

    def plot_feature_importances(self, show_plot=False):
        """
        Generate and return a DataFrame of feature importances.

        Args:
            show_plot (bool, optional): Whether to show a bar plot of feature importances. Default is False.

        Returns:
            pandas.DataFrame: DataFrame containing feature importances.
        """
        model = RandomForestClassifier()
        model.fit(self.x_data, self.y_data)
        importances = model.feature_importances_
        df_importances = pd.DataFrame(data=importances, columns=['importance'])
        df_importances['feature'] = self.idx_features
        if show_plot:
            fig = px.bar(df_importances, x='feature', y='importance',
                         labels={'feature': 'Feature', 'importance': 'Importance'})
            class_counts, class_percentages = self.get_multiclass_stats()
            class_labels = class_counts.keys().tolist()
            class_counts = class_counts.tolist()
            class_percentages = class_percentages.tolist()
            total_samples = len(self.y_data)

            title = f'Feature Importance Plot\nTotal Samples: {total_samples}\n'
            for i, count in enumerate(class_counts):
                percentage = class_percentages[i]
                title += f'C{class_labels[i]}: {count} ({percentage:.2f}%)\n'
            fig.update_layout(title=title)
            fig.show()
        # Sort the dataframe by importance in descending order
        df_importances = df_importances.sort_values(by='importance', ascending=False)
        return df_importances['feature'].tolist()

    def plot_important_feature_scatter_plots(self, top_n=5):
        """
        Generate and save scatter plots for the top important features.

        Args:
            top_n (int, optional): Number of top important features to consider. Default is 5.
        """
        model = RandomForestClassifier()
        model.fit(self.x_data, self.y_data)
        importances = model.feature_importances_
        idx_top_n = np.argsort(importances)[-top_n:]

        fig = make_subplots(rows=top_n, cols=top_n)

        # Determine the number of subplots based on available pairwise comparisons
        num_subplots = 0
        for i in range(top_n):
            for j in range(i + 1, top_n):
                if idx_top_n[i] != idx_top_n[j]:
                    num_subplots += 1

        # Adjust the number of rows and columns for the subplots
        num_rows = int(np.sqrt(num_subplots))
        num_cols = int(np.ceil(num_subplots / num_rows))
        fig = make_subplots(rows=num_rows, cols=num_cols)

        subplot_idx = 1  # Index for the current subplot
        for i in range(top_n):
            for j in range(i + 1, top_n):
                if idx_top_n[i] != idx_top_n[j]:
                    x_values = self.x_data[:, idx_top_n[i]]
                    y_values = self.x_data[:, idx_top_n[j]]

                    # Create a mask to exclude NaN values
                    mask = np.logical_not(np.logical_or(np.isnan(x_values), np.isnan(y_values)))

                    fig.add_trace(
                        go.Scatter(x=x_values[mask], y=y_values[mask],
                                   mode='markers',
                                   marker=dict(color=self.y_data[mask], showscale=False),
                                   showlegend=False,
                                   legendgroup='scatter',
                                   name='Scatter'
                                   ),
                        row=int((subplot_idx - 1) / num_cols) + 1,
                        col=(subplot_idx - 1) % num_cols + 1
                    )
                    fig.update_xaxes(title_text=f"Feature {idx_top_n[i] + 1}",
                                     row=int((subplot_idx - 1) / num_cols) + 1,
                                     col=(subplot_idx - 1) % num_cols + 1)
                    fig.update_yaxes(title_text=f"Feature {idx_top_n[j] + 1}",
                                     row=int((subplot_idx - 1) / num_cols) + 1,
                                     col=(subplot_idx - 1) % num_cols + 1)
                    subplot_idx += 1

        # Add legend for all subplots
        if self.y_data.dtype != np.uint8:
            self.y_data = self.y_data.astype(np.uint8)
        classes = np.unique(self.y_data)
        legend_labels = [str(cls) for cls in classes]
        fig.add_trace(go.Scatter(x=[], y=self.y_data, mode='markers', marker=dict(color=self.y_data),
                                 legendgroup='scatter', name='Class Labels', showlegend=True),
                      row=1, col=1)

        # Update layout for the legend
        fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.1))
        class_counts, class_percentages = self.get_multiclass_stats()
        class_labels = class_counts.keys().tolist()
        class_counts = class_counts.tolist()
        class_percentages = class_percentages.tolist()
        total_samples = len(self.y_data)

        title = f'Multi-Features Scatter Plot\nTotal Samples: {total_samples}\n'
        for i, count in enumerate(class_counts):
            percentage = class_percentages[i]
            title += f'C{class_labels[i]}: {count} ({percentage:.2f}%)\n'
        fig.update_layout(title=title)
        fig.write_html("important_feature_scatter_plots.html", auto_open=True)

    def feature_scatter_plot(self, feature_1, feature_2):
        """
        Generate and save a scatter plot for two specific features.

        Args:
            feature_1 (int): Index of the first feature.
            feature_2 (int): Index of the second feature.
        """
        df = pd.DataFrame(self.x_data, columns=self.idx_features)
        df["target"] = self.y_data

        fig = px.scatter(df, x=feature_1, y=feature_2, color="target",
                         hover_data=[df.index])
        class_counts, class_percentages = self.get_multiclass_stats()
        class_labels = class_counts.keys().tolist()
        class_counts = class_counts.tolist()
        class_percentages = class_percentages.tolist()
        total_samples = len(self.y_data)

        title = f'Scatter plot for features {feature_1} and {feature_2}\n'
        for i, count in enumerate(class_counts):
            percentage = class_percentages[i]
            title += f'C{class_labels[i]}: {count} ({percentage:.2f}%)\n'
        fig.update_layout(title=title)
        fig.write_html(f"Feature_{feature_1}_{feature_2}_scatter_plot.html", auto_open=True)

    def visualize_dataset_statistics(self, n_features=3):
        """
        Visualize dataset statistics including histograms and box plots.

        Args:
            n_features (int, optional): Number of features to visualize. Default is 3.
        """
        x_data = self.x_data
        y_data = self.y_data
        n_samples = x_data.shape[0]
        n_classes = len(np.unique(y_data))

        fig, axes = plt.subplots(nrows=n_features, ncols=n_classes, figsize=(15, 10))
        fig.subplots_adjust(hspace=0.5)

        for feature_idx in range(n_features):
            for class_label in range(n_classes):
                # Extract data for the current feature and class label
                data_subset = x_data[y_data == class_label][:, feature_idx]

                # Plot histogram for the current feature and class label
                ax = axes[feature_idx, class_label]
                ax.hist(data_subset, bins='auto', alpha=0.7)
                ax.set_title(f"Feature {feature_idx + 1} - Class {class_label}")
                ax.set_xlabel("Feature Value")
                ax.set_ylabel("Frequency")

                # Calculate feature statistics
                feature_mean = np.mean(data_subset)
                feature_std = np.std(data_subset)
                ax.text(0.5, 0.9, f"Mean: {feature_mean:.2f}\nStd: {feature_std:.2f}", transform=ax.transAxes,
                        ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))

        # Set overall plot title and labels for the y-axis in the leftmost subplots
        fig.suptitle("Dataset Statistics Visualization", fontsize=16)
        for ax in axes[:, 0]:
            ax.set_ylabel("Frequency")

        # Hide empty subplots
        for i in range(n_features):
            for j in range(n_classes):
                if j > 0:
                    axes[i, j].axis('off')

        plt.show()

    def visualize_dataset_distribution(self, n_features=3):
        """
        Visualize the dataset distribution using pairwise scatter plots and box plots.

        Args:
            n_features (int, optional): Number of features to visualize. Default is 3.
        """
        x_data = self.x_data
        y_data = self.y_data
        n_samples = x_data.shape[0]
        n_classes = len(np.unique(y_data))

        # Pairwise scatter plots for features
        fig_scatter, axes_scatter = plt.subplots(nrows=n_features, ncols=n_features, figsize=(12, 12))
        fig_scatter.suptitle("Pairwise Feature Scatter Plots", fontsize=16)
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    # Plot histogram for the diagonal subplots
                    axes_scatter[i, j].hist(x_data[:, i], bins='auto', alpha=0.7)
                    axes_scatter[i, j].set_xlabel(f"Feature {i + 1}")
                    axes_scatter[i, j].set_ylabel("Frequency")
                else:
                    # Plot scatter plot for non-diagonal subplots
                    for class_label in range(n_classes):
                        class_data = x_data[y_data == class_label]
                        axes_scatter[i, j].scatter(class_data[:, i], class_data[:, j], label=f"Class {class_label}",
                                                   alpha=0.7)
                    axes_scatter[i, j].set_xlabel(f"Feature {i + 1}")
                    axes_scatter[i, j].set_ylabel(f"Feature {j + 1}")
                    axes_scatter[i, j].legend()

        # Boxplots for each feature across different classes
        fig_box, axes_box = plt.subplots(nrows=1, ncols=n_features, figsize=(12, 4))
        fig_box.suptitle("Feature Boxplots Across Classes", fontsize=16)
        for i in range(n_features):
            data_per_class = [x_data[y_data == class_label][:, i] for class_label in range(n_classes)]
            axes_box[i].boxplot(data_per_class, labels=[f"Class {class_label}" for class_label in range(n_classes)])
            axes_box[i].set_xlabel(f"Feature {i + 1}")
            axes_box[i].set_ylabel("Feature Value")

        plt.show()


class Dxy:
    """
    Class to combine x and y data
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y





class LabeledUnlabeledIndicesGenerator:
    def __init__(self, x_data, y_data, sim_set):
        """
        Initialize the LabeledUnlabeledIndicesGenerator class.

        Args:
            y_data (array-like): The target labels of the dataset.
            n_samples (int): Total number of samples in the dataset.
            n_labeled (int): Number of labeled samples in each training set split.
            n_sets (int): Number of random splits of the training samples into labeled and unlabeled samples.
            n_folds (int): Number of folds for cross-validation.
            class_balance (bool): Whether to check the class balance of the labeled samples or not.
            balance_range (tuple): Desired range of class balance (min_balance, max_balance).
            random_state (int or None): Random state for reproducibility.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.n_classes = sim_set.n_classes
        self.n_samples = sim_set.n_samples
        self.n_labeled = sim_set.n_labeled
        self.n_folds = sim_set.n_folds
        self.n_sets = sim_set.n_sets
        self.class_balance = sim_set.class_balance
        self.balance_range = sim_set.balance_range
        self.sort_l_u_distance = sim_set.sort_l_u_distance
        self.random_state = sim_set.random_state
        self.indices_dict = {}


    def generate_indices(self):
        """
        Generate labeled and unlabeled indices for each fold and set.

        Returns:
            dict: A dictionary containing the labeled and unlabeled indices for each fold and set.
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        final_random_states_used = []

        for fold_num, (train_indices, test_indices) in enumerate(kf.split(range(self.n_samples))):
            self.indices_dict[fold_num] = {}

            for set_num in range(self.n_sets):
                random_state = self.random_state  # Starting random state for each fold and set
                found_indices = False
                num_tries = 0

                while not found_indices:

                    np.random.seed(random_state)  # Set the random seed

                    labeled_indices = np.random.choice(train_indices, size=self.n_labeled, replace=False)
                    unlabeled_indices = np.setdiff1d(train_indices, labeled_indices)

                    np.random.shuffle(labeled_indices)
                    np.random.shuffle(unlabeled_indices)

                    if not self.class_balance or self.check_class_balance(labeled_indices):
                        if self.sort_l_u_distance:
                            unlabeled_indices = self.get_sorted_l_u_distances(labeled_indices, unlabeled_indices)
                        self.indices_dict[fold_num][set_num] = {
                            "labeled": labeled_indices,
                            "unlabeled": unlabeled_indices,
                            "train": train_indices,
                            "test": test_indices
                        }
                        final_random_states_used.append(random_state)
                        found_indices = True
                    else:
                        print(f"Attempt (fold={fold_num}, set={set_num}): {num_tries}")
                        num_tries += 1
                        random_state += 1  # Increment random state for subsequent tries

        return self.indices_dict

    def check_class_balance(self, labeled_indices):
        """
        Check the class balance of the labeled samples.

        Args:
            labeled_indices (ndarray): Array of labeled indices.

        Returns:
            bool: True if the class balance is within the specified range, False otherwise.
        """
        labeled_labels = self.y_data[labeled_indices]
        class_counts = np.bincount(labeled_labels, minlength=self.n_classes)
        class_ratios = class_counts / len(labeled_labels)
        min_balance, max_balance = self.balance_range

        return np.all((class_ratios >= min_balance) & (class_ratios <= max_balance))

    def get_indices_dict(self, fold_num=None, set_num=None):
        """
        Get the dictionary of generated labeled and unlabeled indices.

        Args:
            fold_num (int or None): The fold number to retrieve the indices from. If None, returns indices for all folds.
            set_num (int or None): The set number to retrieve the indices from. If None, returns indices for all sets.

        Returns:
            dict: A dictionary containing the labeled and unlabeled indices for each fold and set.
        """
        if fold_num is not None and set_num is not None:
            return self.indices_dict.get(fold_num, {}).get(set_num, {})
        elif fold_num is not None:
            return self.indices_dict.get(fold_num, {})
        else:
            return self.indices_dict

    def get_sorted_l_u_distances(self, labeled_indices, unlabeled_indices_rand):
        """
        This function calulates the distance between any pair of labeled and unlabeled samples
        and returns the list of unlabeled samples sorted based on their distance to the closest labeled sample
        @param x_train_l: labeled samples in the training dataset
        @param x_train_u: unlabeled samples in the training dataset
        @param idx_train_u_rand: list of indices corresponding to the unlabaled samples
        @return idx_u_sorted: list of unlabeled samples sorted based on their distance to the closest labeled sample
        """
        dist = cdist(self.x_data[labeled_indices, :], self.x_data[unlabeled_indices_rand, :])
        dist_sort = dist.argsort(axis=None, kind="mergesort")
        dist_sort_indices = np.unravel_index(dist_sort, dist.shape)

        dist_sort_u_indices = dist_sort_indices[1]
        idx_u_sorted_list = [i for n, i in enumerate(dist_sort_u_indices) if i not in dist_sort_u_indices[:n]]

        unlabeled_indices = unlabeled_indices_rand[idx_u_sorted_list]

        return unlabeled_indices


class DatasetGenerator:
    """
    A class for generating datasets and low-dimensional representations for semi-supervised learning experiments.

    Args:
        x_data (ndarray): Input data samples.
        y_data (ndarray): Corresponding labels for the input data.
        indices_dict (dict): Dictionary containing the labeled and unlabeled indices for each fold and set.
        fold_num (int): The fold number.
        set_num (int): The set number.
        sim_set (object): Simulation settings object.

    Attributes:
        x_data (ndarray): Input data samples.
        y_data (ndarray): Corresponding labels for the input data.
        indices_dict (dict): Dictionary containing the labeled and unlabeled indices for each fold and set.
        fold_num (int): The fold number.
        set_num (int): The set number.
        n_labeled (int): Number of labeled samples.
        n_unlabeled (int): Number of unlabeled samples.
        number_of_bootstrap_in_filter_method (int): Number of bootstraps in the feature selection step.
        number_of_selected_features (int): Number of features to be selected.
        number_of_pca (int): Number of principal components for the low-dimensional representation.
        print_idx_n_mix_rand (bool): Flag indicating whether to print index, number of mixed samples, and random seed.
        random_state (int): Random state for data shuffling.
        n_rand_max (int): Maximum number of random seeds.
        random_states_list (list): List of random states for generating mixed samples.
    """

    def __init__(self, x_data, y_data, indices_dict, fold_num: int, set_num: int, sim_set: object):
        self.x_data = x_data
        self.y_data = y_data
        self.indices_dict = indices_dict
        self.fold_num = fold_num
        self.set_num = set_num
        self.n_labeled = sim_set.n_labeled
        self.n_unlabeled = sim_set.n_unlabeled
        self.number_of_bootstrap_in_filter_method = sim_set.number_of_bootstrap_in_filter_method
        self.number_of_selected_features = sim_set.number_of_selected_features
        self.number_of_pca = sim_set.number_of_pca
        self.print_idx_n_mix_rand = sim_set.print_idx_n_mix_rand
        self.random_state = sim_set.random_state
        self.n_rand_max = sim_set.n_rand_max
        self.random_states_list = [self.random_state + i for i in range(self.n_rand_max)]

    def generate_dataset(self):
        """
        Generate datasets for a specific fold and set.

        Returns:
            tuple: A tuple containing data objects for the dataset and its subsets.
        """
        idx_train_l = self.indices_dict[self.fold_num][self.set_num]["labeled"]
        idx_train_u = self.indices_dict[self.fold_num][self.set_num]["unlabeled"]
        idx_train = self.indices_dict[self.fold_num][self.set_num]["train"]
        idx_test = self.indices_dict[self.fold_num][self.set_num]["test"]

        y_data_temp = self.y_data.copy()
        y_train_u_true = self.y_data[idx_train_u]
        y_data_temp[idx_train_u] = -1
        y_train_l = y_data_temp[idx_train_l]
        y_train_u = y_data_temp[idx_train_u]
        y_train_mix = np.concatenate((y_train_l, y_train_u), axis=0)
        y_train_mix_true = np.concatenate((y_train_l, y_train_u_true), axis=0)

        self.d_data = Dxy(self.x_data, self.y_data)
        self.d_train = Dxy(self.x_data[idx_train], self.y_data[idx_train])
        self.d_test = Dxy(self.x_data[idx_test], self.y_data[idx_test])
        self.d_train_l = Dxy(self.x_data[idx_train_l], y_train_l)
        self.d_train_u = Dxy(self.x_data[idx_train_u], y_train_u)
        self.d_train_u_true = Dxy(self.x_data[idx_train_u], y_train_u_true)
        self.d_train_mix = Dxy(np.vstack((self.x_data[idx_train_l], self.x_data[idx_train_u])),
                               y_train_mix)
        self.d_train_mix_true = Dxy(np.vstack((self.x_data[idx_train_l], self.x_data[idx_train_u])),
                                    y_train_mix_true)

        return self.d_data, self.d_train, self.d_test, self.d_train_l, self.d_train_u, self.d_train_u_true, \
               self.d_train_mix, self.d_train_mix_true

    def get_selected_feature_indices(self):
        """
        Get the indices of selected features based on the filter method using mutual information.

        Returns:
            tuple: A tuple containing indices representing selected features.
        """
        self.idx_mi_l = filter_method(self.number_of_bootstrap_in_filter_method, self.number_of_selected_features,
                                      self.d_train_l.x, self.d_train_l.y)
        self.idx_mi_train = self.idx_mi_l
        self.idx_mi_train = filter_method(self.number_of_bootstrap_in_filter_method, self.number_of_selected_features,
                                          self.d_train.x, self.d_train.y)
        # -----------------------------------------------
        print(f"Selected features: {self.idx_mi_l}")
        return self.idx_mi_l, self.idx_mi_train

    def print_available_data_types(self):
        """
        Prints the available data types that can be obtained from get_lowdim_dataset along with their descriptions.
        """
        data_type_descriptions = {
            "n_mix_d": "Original n_mix training (labeled and included unlabeled) samples and test samples",
            "n_mix_dp_mi": "Low dimensional representation of the n_mix training (labeled and included unlabeled) samples and test samples based on the selected features obtained from the filter method",
            "n_mix_dp_tsne": "Low dimensional representation of the n_mix training (labeled and included unlabeled) samples and test samples based on their projection on the tsne components",
            "n_mix_dp_pca": "Low dimensional representation of the n_mix training (labeled and included unlabeled) samples and test samples based on their projection on the principal components",
            "true_n_mix_d": "Original n_mix training (labeled and true label of the included unlabeled) samples and test samples",
            "true_n_mix_dp_mi": "Low dimensional representation of the n_mix training (labeled and true label of the included unlabeled) samples and test samples based on the selected features obtained from the filter method",
            "true_n_mix_dp_pca": "Low dimensional representation of the n_mix training (labeled and  true label of the included unlabeled) samples and test samples based on their projection on the principal components",
            "original_dp_mi": "Low dimensional representation of the training, and test samples based on the selected features obtained from the filter method",
            "orig_dp_pca": "Low dimensional representation of the training, and test samples based on their projection on the principal components",
        }

        print("--------------------------------------------------------------------------------")
        print("d: total number of features")
        print("dp_mi: represents low dimensional representation obtained from filter method")
        print("dp_pca: represents low dimensional representation obtained from pca")
        print("--------------------------------------------------------------------------------")
        print("Available data types: ")
        print("--------------------------------------------------------------------------------")
        self.data_types = []

        for data_type, description in data_type_descriptions.items():
            print("- {}: {}".format(data_type, description))
            self.data_types.append(data_type)
        return self.data_types

    def get_lowdim_dataset(self, n_rand_i, n_mix, data_type):
        """
        Generate a low-dimensional representation of the dataset based on the specified parameters.

        Args:
            n_rand_i (int): Index of the random seed.
            n_mix (int): Number of mixed labeled and unlabeled samples.
            data_type (str): The desired data type for the low-dimensional representation.

        Returns:
            tuple: A tuple containing the low-dimensional representation of the dataset based on the specified data type.
        """
        self.n_mix = n_mix
        idx_mi_l = self.idx_mi_l
        idx_mi_train = self.idx_mi_train

        idx_n_mix_rand = self.choose_n_mix_samples(n_rand_i)

        if data_type == "n_mix_d":
            self.d_train_n_mix = Dxy(self.d_train_mix.x[idx_n_mix_rand], self.d_train_mix.y[idx_n_mix_rand])
            self.d_train_n_mix_true = Dxy(self.d_train_mix.x[idx_n_mix_rand], self.d_train_mix_true.y[idx_n_mix_rand])
            self.d_test_n_mix = self.d_test
            return self.d_train_n_mix, self.d_test_n_mix, self.d_train_n_mix_true

        elif data_type == "n_mix_dp_mi":
            self.d_train_n_mix_dp_mi = Dxy(self.d_train_mix.x[idx_n_mix_rand][:, idx_mi_l],
                                           self.d_train_mix.y[idx_n_mix_rand])
            self.d_train_n_mix_dp_mi_true = Dxy(self.d_train_mix_true.x[idx_n_mix_rand][:, idx_mi_l],
                                                self.d_train_mix_true.y[idx_n_mix_rand])
            self.d_test_n_mix_dp_mi = Dxy(self.d_test.x[:, idx_mi_l], self.d_test.y)
            return self.d_train_n_mix_dp_mi, self.d_test_n_mix_dp_mi, self.d_train_n_mix_dp_mi_true

        elif data_type == "n_mix_dp_tsne":
            tsne = TSNE(n_components=2, random_state=self.random_state)
            x_train = self.d_train_mix.x[idx_n_mix_rand]
            x_test = self.d_test.x
            x_data = np.concatenate((x_train, x_test), axis=0)
            x_data_transformed = tsne.fit_transform(x_data)
            x_data_transformed = normalize(x_data_transformed, axis=0)
            x_train_transformed = x_data_transformed[:len(x_train)]
            x_test_transformed = x_data_transformed[len(x_train):]
            self.x_train_n_mix_dp_tsne = x_train_transformed
            self.x_test_n_mix_dp_tsne = x_test_transformed
            self.d_train_n_mix_dp_tsne = Dxy(self.x_train_n_mix_dp_tsne, self.d_train_mix.y[idx_n_mix_rand])
            self.d_train_n_mix_dp_tsne_true = Dxy(self.x_train_n_mix_dp_tsne, self.d_train_mix_true.y[idx_n_mix_rand])
            self.d_test_n_mix_dp_tsne = Dxy(self.x_test_n_mix_dp_tsne, self.d_test.y)
            return self.d_train_n_mix_dp_tsne, self.d_test_n_mix_dp_tsne, self.d_train_n_mix_dp_tsne_true

        elif data_type == "n_mix_dp_pca":
            pca_mix = PCA(n_components=self.number_of_pca)
            self.x_train_n_mix_dp_pca = pca_mix.fit_transform(self.d_train_mix.x[idx_n_mix_rand])
            self.x_test_n_mix_dp_pca = pca_mix.transform(self.d_test.x)
            self.d_train_n_mix_dp_pca = Dxy(self.x_train_n_mix_dp_pca, self.d_train_mix.y[idx_n_mix_rand])
            self.d_train_n_mix_dp_pca_true = Dxy(self.x_train_n_mix_dp_pca, self.d_train_mix_true.y[idx_n_mix_rand])
            self.d_test_n_mix_dp_pca = Dxy(self.x_test_n_mix_dp_pca, self.d_test.y)
            return self.d_train_n_mix_dp_pca, self.d_test_n_mix_dp_pca, self.d_train_n_mix_dp_pca_true

        elif data_type == "true_n_mix_d":
            self.d_train_n_mix_true = Dxy(self.d_train_mix.x[idx_n_mix_rand], self.d_train_mix_true.y[idx_n_mix_rand])
            self.d_test_n_mix = self.d_test
            return self.d_train_n_mix_true, self.d_test_n_mix

        elif data_type == "true_n_mix_dp_mi":
            self.d_train_n_mix_dp_mi_true = Dxy(self.d_train_mix_true.x[idx_n_mix_rand][:, idx_mi_l],
                                                self.d_train_mix_true.y[idx_n_mix_rand])
            self.d_test_n_mix_dp_mi = Dxy(self.d_test.x[:, idx_mi_l], self.d_test.y)
            return self.d_train_n_mix_dp_mi_true, self.d_test_n_mix_dp_mi

        elif data_type == "true_n_mix_dp_pca":
            pca_mix = PCA(n_components=self.number_of_pca)
            self.x_train_n_mix_dp_pca = pca_mix.fit_transform(self.d_train_mix.x[idx_n_mix_rand])
            self.x_test_n_mix_dp_pca = pca_mix.transform(self.d_test.x)
            self.d_train_n_mix_dp_pca_true = Dxy(self.x_train_n_mix_dp_pca, self.d_train_mix_true.y[idx_n_mix_rand])
            self.d_test_n_mix_dp_pca = Dxy(self.x_test_n_mix_dp_pca, self.d_test.y)
            return self.d_train_n_mix_dp_pca_true, self.d_test_n_mix_dp_pca

        elif data_type == "original_dp_mi":
            self.d_train_dp_mi = Dxy(self.d_train.x[:, idx_mi_train], self.d_train.y)
            self.d_test_dp_mi = Dxy(self.d_test.x[:, idx_mi_train], self.d_test.y)
            return self.d_train_dp_mi, self.d_test_dp_mi

        elif data_type == "orig_dp_pca":
            pca_train = PCA(n_components=self.number_of_pca)
            self.x_train_dp_pca = pca_train.fit_transform(self.d_train.x)
            self.x_test_dp_pca = pca_train.transform(self.d_test.x)
            self.d_train_dp_pca = Dxy(self.x_train_dp_pca, self.d_train.y)
            self.d_test_dp_pca = Dxy(self.x_test_dp_pca, self.d_test.y)
            return self.d_train_dp_pca, self.d_test_dp_pca

        else:
            raise ValueError("Invalid data_type. Please specify a valid dataset type.")

    def choose_n_mix_samples(self, n_rand_i):
        random_state = self.random_states_list[n_rand_i]
        np.random.seed(random_state)  # random seed
        # np.random.seed()  # random seed
        indices = np.random.choice(len(self.d_train_mix.x) - self.n_labeled, size=self.n_mix - self.n_labeled,
                                   replace=False)
        idx_n_mix_rand = np.concatenate((np.arange(self.n_labeled), indices + self.n_labeled), axis=0)
        if self.print_idx_n_mix_rand:
            print(f'random_state = {random_state}')
            print(f'Selected L/U samples {idx_n_mix_rand}')
        return idx_n_mix_rand




class SemiSupervisedPipeline:
    """
    A class for running a semi-supervised pipeline.

    Attributes:
        sim_set (SemiSupSettings): Object containing semi-supervised settings.
        x_data (array-like): Input data.
        y_data (array-like): Target labels.
        n_samples (int): Number of samples.
        n_folds (int): Number of folds.
        n_sets (int): Number of sets.
        n_labeled (int): Number of labeled instances.
        n_unlabeled (int): Number of unlabeled instances.
        step_to_unlabeled_as_labeled (int): Step size for transitioning from unlabeled to labeled instances.
        n_rand_max (int): Maximum number of random seeds.
        n_classes (int): Number of classes.
        class_balance (bool): Flag indicating whether to balance class distribution.
        hyperparameter_tuning (bool): Flag indicating whether to perform hyperparameter tuning.
        balance_range (tuple): Balance range for class balancing.
        random_state (int): Random state for reproducibility.
        number_of_bootstrap_in_filter_method (int): Number of bootstraps in the filter method.
        number_of_selected_features (int): Number of selected features.
        number_of_pca (int): Number of principal components.
        plot_animation (bool): Flag indicating whether to plot animations.
        plt_auc_calc_type (list): List of AUC calculation types ("ave" or "ind").
        print_results (bool): Flag indicating whether to print results of models assigned in each iteration.
        print_idx_n_mix_rand (bool): Flag indicating whether to print index, number of mixed samples, and random seed.
        version (str): Version identifier.
        semisup_analysis_df (pd.DataFrame): DataFrame for storing analysis results.
        indices_dict (dict): Dictionary of indices for labeled and unlabeled instances.
        models_df (pd.DataFrame): DataFrame for storing model information.
    """

    def __init__(self, sim_set, x_data, y_data, data_types_indices=None):
        """
        Initializes the SemiSupervisedPipeline object.

        Args:
            sim_set (SemiSupSettings): Object containing semi-supervised settings.
            x_data (array-like): Input data.
            y_data (array-like): Target labels.
        """
        if data_types_indices is None:
            data_types_indices = [0, 1]
        self.data_types_indices = data_types_indices
        self.sim_set = sim_set
        self.x_data = x_data
        self.y_data = y_data
        self.pseudo_label_model_list = sim_set.pseudo_label_model_list
        self.test_model_list = sim_set.test_model_list
        self.n_samples = sim_set.n_samples
        self.n_folds = sim_set.n_folds
        self.n_sets = sim_set.n_sets
        self.n_labeled = sim_set.n_labeled
        self.n_unlabeled = sim_set.n_unlabeled
        self.step_to_unlabeled_as_labeled = sim_set.step_to_unlabeled_as_labeled
        self.n_rand_max = sim_set.n_rand_max
        self.n_classes = sim_set.n_classes
        self.class_balance = sim_set.class_balance
        self.hyperparameter_tuning = sim_set.hyperparameter_tuning
        self.balance_range = sim_set.balance_range
        self.random_state = sim_set.random_state
        self.number_of_bootstrap_in_filter_method = sim_set.number_of_bootstrap_in_filter_method
        self.number_of_selected_features = sim_set.number_of_selected_features
        self.number_of_pca = sim_set.number_of_pca
        self.plot_animation = sim_set.plot_animation
        self.plt_auc_calc_type = sim_set.plt_auc_calc_type  # "average" or "individual"
        self.print_results = sim_set.print_results
        self.print_idx_n_mix_rand = sim_set.print_idx_n_mix_rand
        self.version = sim_set.version
        self.semisup_analysis_df = self.initialize_semisup_analysis_df()
        indices_generator = LabeledUnlabeledIndicesGenerator(self.x_data, self.y_data, self.sim_set)
        self.indices_dict = indices_generator.generate_indices()
        self.models_df = pd.DataFrame(columns=["set_num", "fold_num", "model", "params"])
        self.n_mix_values, self.n_rand_for_n_mix = self.get_n_rand_for_n_mix()
        self.auc_stats_dict = {}
        self.auc_stats_dict_fold = {}

        for fold_num, set_num in product(range(self.n_folds), range(self.n_sets)):
            dataset = DatasetGenerator(x_data=x_data,
                                       y_data=y_data,
                                       indices_dict=self.indices_dict,
                                       fold_num=fold_num,
                                       set_num=set_num,
                                       sim_set=self.sim_set)
            dataset.generate_dataset()

            dataset.print_available_data_types()
            dataset.get_selected_feature_indices()
            data_types = dataset.data_types  # data_types = ["n_mix_d", "n_mix_dp_mi", "n_mix_dp_pca",
            # "true_n_mix_d", "true_n_mix_dp_mi", "true_n_mix_dp_pca",
            # "original_dp_mi", "orig_dp_pca"]


            self.data_types = [data_types[i] for i in self.data_types_indices]

            gamma_val = 0.1
            model_parameters = ClassificationModelParameters(
                lp_params={"kernel": "knn", "n_neighbors": self.n_labeled-1, "max_iter": 3000},
                ls_params={"kernel": "knn", "n_neighbors": self.n_labeled-1, "alpha": 0.2, "max_iter": 3000},
                tsvm_params={"kernel": "rbf", "gamma": gamma_val, "probability": True},
                cple_params={"predict_from_probabilities": True},
                qwda_params={"predict_from_probabilities": True},
                svmrbf_params={"C": 2.0, "kernel": "rbf", "gamma": gamma_val, "probability": True},
                svmlin_params={"C": 0.05, "kernel": "linear", "probability": True},
                knn_params={"n_neighbors": self.n_labeled-1},
                dt_params={"max_depth": 3},
                gb_params={"n_estimators": 50, "learning_rate": 0.2, "max_depth": 3})
            self.model_params = model_parameters.get_model_parameters()

            if self.hyperparameter_tuning:
                if self.hyperparameter_tuning:
                    print("Hyperparameter tuning started.")
                    start_time = time.time()
                    self.best_params = ClassificationModelGridSearch(dataset.d_train_l.x,
                                                                     dataset.d_train_l.y).perform_grid_search(
                        selected_classifiers=self.test_model_list)
                    end_time = time.time()
                    print("Hyperparameter tuning completed. Time taken: {} seconds".format(end_time - start_time))
                self.model_params["svmrbf_params"] = self.best_params["svmrbf"]
                self.model_params["svmlin_params"] = self.best_params["svmlin"]
                self.model_params["knn_params"] = self.best_params["knn"]
                self.model_params["dt_params"] = self.best_params["dt"]
                self.model_params["gb_params"] = self.best_params["gb"]

            # for model, params in self.model_params.items():
                # self.models_df = self.models_df.append(
                #     {"set_num": set_num, "fold_num": fold_num, "model": model, "params": params},
                #     ignore_index=True)

            self.pl_test_model_list = model_parameters.get_ss_method_list(self.pseudo_label_model_list,
                                                                          self.test_model_list)  # get the available list of models to perform pseudo-labeling and testing
            for (pl_model_i, test_model_i), data_type in product(self.pl_test_model_list,
                                                                 self.data_types):
                if self.plot_animation:
                    frame_cntr = 0
                    pl_anim = PsuedoLabelingAnimation(dataset=dataset, version=self.version)
                    pl_anim.create_empty_plots()

                # if self.sim_set.plt_centroids:
                #     centroids_l_all = []
                #     centroids_mix_pseudo_all = []
                #     centroids_mix_true_all = []
                #     centroids_test_true_all = []
                #     centroids_test_pred_all = []
                #     centroids_data_all = []



                for n_mix in self.n_mix_values:
                    n_rand = self.n_rand_for_n_mix[self.n_rand_for_n_mix[:, 0] == n_mix, 1][0]
                    # self.model_params["lp_params"]["n_neighbors"] = int(0.3 * n_mix)
                    # self.model_params["ls_params"]["n_neighbors"] = int(0.3 * n_mix)
                    for n_rand_i in range(n_rand):
                        # if n_rand_i == 0:

                        self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_ave, \
                        self.pl_auc_ave, self.t_auc_ind, self.pl_auc_ind, self.t_model, \
                        self.d_train, self.d_test, self.d_train_true = \
                            SemiSupervisedClassification(n_labeled=self.n_labeled,
                                                         n_mix=n_mix,
                                                         n_rand_i=n_rand_i,
                                                         n_classes=self.n_classes,
                                                         model_params=self.model_params,
                                                         pseudo_label_model=pl_model_i,
                                                         test_model=test_model_i,
                                                         dataset=dataset,
                                                         data_type=data_type,
                                                         print_results=self.print_results).evaluate_semisupervised_classification()

                        self.semisup_analysis_df = self.update_semisup_analysis_df(n_mix=n_mix,
                                                                                   n_rand=n_rand,
                                                                                   n_rand_i=n_rand_i,
                                                                                   data_type=data_type,
                                                                                   fold_num=fold_num,
                                                                                   set_num=set_num,
                                                                                   pseudo_label_model=pl_model_i,
                                                                                   test_model=test_model_i)

                        if self.plot_animation:
                            pl_anim.init_animation()
                            pl_anim.animate_pseudo_labels(test_model=self.t_model, d_train=self.d_train,
                                                          d_test=self.d_test,
                                                          d_train_true=self.d_train_true, y_pseudo=self.y_pseudo,
                                                          y_pred=self.y_pred, n_mix=n_mix, frame_cntr=frame_cntr,
                                                          n_rand_i=n_rand_i)
                    if self.plot_animation:
                        frame_cntr += 1

                if self.plot_animation:
                    pl_anim.save_animation(n_frames=frame_cntr, data_type=data_type,
                                           fold_num=fold_num, set_num=set_num, n_rand=n_rand,
                                           pseudo_label_model=pl_model_i, test_model=test_model_i, version=self.version)



    def initialize_semisup_analysis_df(self):
        """
        Initializes the dataframe for storing semi-supervised analysis results.
        Returns:
            pandas.DataFrame: The initialized dataframe.
        """

        col_names = ["data_type", "ss_model", "fold_num", "set_num", "n_mix", "n_rand", "n_rand_i", ] + \
                    ["t_auc_ave", "pl_auc_ave"] + \
                    ["pl_auc_ind_" + str(i) for i in range(1, self.n_classes + 1)] + \
                    ["t_auc_ind_" + str(i) for i in range(1, self.n_classes + 1)]

        semisup_analysis_df = pd.DataFrame(columns=col_names)

        return semisup_analysis_df

    def get_n_rand_for_n_mix(self):
        n_mix_values = np.array(list(range(self.n_labeled,
                                           self.n_labeled + self.n_unlabeled,
                                           self.step_to_unlabeled_as_labeled)))
        if n_mix_values[-1] < self.n_labeled + self.n_unlabeled:
            n_mix_values = np.append(n_mix_values, self.n_labeled + self.n_unlabeled)
        n_rand_max = self.n_rand_max  # Maximum number of times to randomly choose subsets
        n_rand_min = 1  # Minimum number of times to randomly choose subsets
        # Calculate the number of combinations
        n_combinations = len(n_mix_values) - 1
        # Generate the vector n_rand with decreasing values
        n_rand_for_n_mix = np.linspace(n_rand_max, n_rand_min, n_combinations).astype(int)
        # Add 1 to both ends of the array
        n_rand_for_n_mix = np.insert(n_rand_for_n_mix, 0, 1)
        # Generate the vector with n_mix and n_rand values
        n_rand_for_n_mix = np.vstack((n_mix_values, n_rand_for_n_mix)).T

        # Print the statistical information and n_mix_n_rand_vector values
        if self.print_idx_n_mix_rand:
            print("===================================================")
            print("\nVector n_mix_n_rand_vector:")
            print(
                f"n_labeled={self.n_labeled}--n_unlabeled={self.n_unlabeled}--step_size ={self.step_to_unlabeled_as_labeled}")
            print("---------------------------------------------------")
            print(n_rand_for_n_mix)
            print("===================================================")

        return n_mix_values, n_rand_for_n_mix

    def update_semisup_analysis_df(self, n_mix, n_rand, n_rand_i, data_type, fold_num, set_num, pseudo_label_model,
                                   test_model):
        """
        Updates the dataframe with semi-supervised analysis results.
        Args:
            n_mix (int): Number of mixed instances.
            n_rand (int): Number of times to randomly choose subsets.
            n_rand_i (int): Number of times to randomly choose subsets.
            data_type (str): Data type.
            fold_num (int): Fold number.
            set_num (int): Set number.
            pseudo_label_model (str): Pseudo-labeling model.
            test_model (str): Testing model.
            pl_auc (float): Pseudo-labeling ROC AUC.

        Returns:
            pandas.DataFrame: The updated dataframe.
        """

        t_auc_values = [self.t_auc_ind[class_idx] for class_idx in range(self.n_classes)]
        pl_auc_values = [self.pl_auc_ind[class_idx] for class_idx in range(self.n_classes)]

        col_names = ["data_type", "ss_model", "fold_num", "set_num", "n_mix", "n_rand", "n_rand_i"] + \
                    ["t_auc_ave", "pl_auc_ave"] + \
                    [f"pl_auc_ind_{i}" for i in range(1, self.n_classes + 1)] + \
                    [f"t_auc_ind_{i}" for i in range(1, self.n_classes + 1)]

        new_row = pd.DataFrame({"data_type": [data_type],
                                "ss_model": [pseudo_label_model + "_" + test_model],
                                "fold_num": [fold_num],
                                "set_num": [set_num],
                                "n_mix": [n_mix],
                                "n_rand": [n_rand],
                                "n_rand_i": [n_rand_i],
                                "pl_auc_ave": [self.pl_auc_ave],
                                "t_auc_ave": [self.t_auc_ave],
                                **dict(zip(col_names[9: 9 + self.n_classes], pl_auc_values)),
                                **dict(zip(col_names[9 + self.n_classes:], t_auc_values))})

        table = [
            ["Parameter", "Value"],
            ["data_type", data_type],
            ["ss_model", pseudo_label_model + "_" + test_model],
            ["fold_num/ n_folds", str(fold_num+1) + "/" + str(self.n_folds)],
            ["set_num/ n_sets", str(set_num+1) + "/" + str(self.n_sets)],
            ["n_mix / (n_labeled + n_unlabeled)", str(n_mix) + "/ (" + str(self.n_labeled) + '+' + str(self.n_unlabeled) + ")"],
            ["n_rand_i / n_rand", str(n_rand_i+1) + "/" + str(n_rand)],
            ["pl_auc_ave", self.pl_auc_ave],
            ["t_auc_ave", self.t_auc_ave],
            *[[f"pl_auc_ind_{i}", pl_auc] for i, pl_auc in enumerate(pl_auc_values, start=1)],
            *[[f"t_auc_ind_{i}", t_auc] for i, t_auc in enumerate(t_auc_values, start=1)]
        ]

        print("------------------------------------------------")
        print(tabulate(table, headers="firstrow"))
        print("------------------------------------------------")

        self.semisup_analysis_df = pd.concat([new_row, self.semisup_analysis_df]).reset_index(drop=True)
        return self.semisup_analysis_df

    def get_df_n_mix(self, df, data_type, ss_model_i):
        df_n_mix = df[(df["data_type"] == data_type) & (df["ss_model"] == ss_model_i)]
        return df_n_mix

    def summarize_semisup_analysis_df(self):
        # Define the summary statistics
        auc_statistics = ["ave", "min", "max", "5th", "95th"]

        # Create a list of column names
        col_names = ["data_type", "ss_model", "n_mix"] + [f"{col}_{stat}" for col in ["t_auc_ave", "pl_auc_ave"] for
                                                          stat in auc_statistics] + [f"{col}_ind_{i}_{stat}" for col in
                                                                                     ["pl_auc", "t_auc"] for i in
                                                                                     range(1, self.n_classes + 1) for
                                                                                     stat in auc_statistics]

        # Initialize an empty list to store the summary results
        summary_data = []

        # Group the original DataFrame by "data_type", "ss_model", and "n_mix"
        grouped_df = self.semisup_analysis_df.groupby(["data_type", "ss_model", "n_mix"])

        # Iterate over the groups
        for group_name, group_df in grouped_df:
            # Initialize a dictionary to store the summary values for this group
            summary_dict = {"data_type": group_name[0], "ss_model": group_name[1], "n_mix": group_name[2]}

            # Calculate the summary statistics for each column
            for col in ["t_auc_ave", "pl_auc_ave"]:
                col_data = group_df[col]
                for stat in auc_statistics:
                    if stat == "ave":
                        summary_dict[f"{col}_{stat}"] = col_data.mean()
                    elif stat == "min":
                        summary_dict[f"{col}_{stat}"] = col_data.min()
                    elif stat == "max":
                        summary_dict[f"{col}_{stat}"] = col_data.max()
                    elif stat == "5th":
                        summary_dict[f"{col}_{stat}"] = np.percentile(col_data, 5)
                    elif stat == "95th":
                        summary_dict[f"{col}_{stat}"] = np.percentile(col_data, 95)

            for col in ["pl_auc", "t_auc"]:
                for i in range(1, self.n_classes + 1):
                    col_data = group_df[f"{col}_ind_{i}"]
                    for stat in auc_statistics:
                        if stat == "ave":
                            summary_dict[f"{col}_ind_{i}_{stat}"] = col_data.mean()
                        elif stat == "min":
                            summary_dict[f"{col}_ind_{i}_{stat}"] = col_data.min()
                        elif stat == "max":
                            summary_dict[f"{col}_ind_{i}_{stat}"] = col_data.max()
                        elif stat == "5th":
                            summary_dict[f"{col}_ind_{i}_{stat}"] = np.percentile(col_data, 5)
                        elif stat == "95th":
                            summary_dict[f"{col}_ind_{i}_{stat}"] = np.percentile(col_data, 95)

            # Append the summary values to the list
            summary_data.append(summary_dict)

        # Create a DataFrame from the summary data
        self.df_summary = pd.DataFrame(summary_data)

        return self.df_summary

    def get_auc_stats_parameters(self):
        self.df_summary = self.summarize_semisup_analysis_df()
        self.df_summary_parameters = {"ss_model_names": self.df_summary["ss_model"].unique(),
                                      "data_types": self.df_summary["data_type"].unique(),
                                      "auc_statistics": ["ave", "min", "max", "5th", "95th"],
                                      "plt_auc_type": ["pl_auc", "t_auc"],
                                      "plt_auc_calc_type": ["ind", "ave"]
                                      }
        return self.df_summary, self.df_summary_parameters


    def plot_aucs_stats(self, n_mix_list, df_n_mix):
        # fig = plt.figure(figsize=(18, 8))
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(top=0.9)
        columns = ["t_auc_ave", "pl_auc_ave"] + [f"pl_auc_ind_{i}" for i in range(1, self.n_classes + 1)] + [
            f"t_auc_ind_{i}" for i in range(1, self.n_classes + 1)]

        df_aucs_ave = df_n_mix.groupby("n_mix").agg({col: "mean" for col in columns})
        df_aucs_min = df_n_mix.groupby("n_mix").agg({col: "min" for col in columns})
        df_aucs_max = df_n_mix.groupby("n_mix").agg({col: "max" for col in columns})
        df_aucs_5th = df_n_mix.groupby("n_mix").agg({col: lambda x: np.percentile(x, 5) for col in columns})
        df_aucs_95th = df_n_mix.groupby("n_mix").agg({col: lambda x: np.percentile(x, 95) for col in columns})
        # ==============================================================================================================
        #                           Plotting Psuedo-label AUCs
        # ==============================================================================================================
        if ("ave" in self.sim_set.plt_auc_calc_type) and ("pl_auc" in self.sim_set.plt_auc_type):

            if ("ave" in self.sim_set.plt_auc_stat_type):
                style_m = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="ave")
                plt.plot(n_mix_list, df_aucs_ave["pl_auc_ave"].values, **style_m, label="p_auc_ave")

            if ("min" and "max" in self.sim_set.plt_auc_stat_type):
                if "lines" in self.sim_set.plt_auc_stat_style:
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="min")
                    plt.plot(n_mix_list, df_aucs_min["pl_auc_ave"].values, **style, label="p_auc_min")
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="max")
                    plt.plot(n_mix_list, df_aucs_max["pl_auc_ave"].values, **style, label="p_auc_max")
                if "fill" in self.sim_set.plt_auc_stat_style:
                    plt.fill_between(n_mix_list, df_aucs_min["pl_auc_ave"].values, df_aucs_max["pl_auc_ave"].values,
                                     alpha=0.2, color=style_m["color"], hatch=".")

            if ("5th" and "95th" in self.sim_set.plt_auc_stat_type):
                if "lines" in self.sim_set.plt_auc_stat_style:
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="5th")
                    plt.plot(n_mix_list, df_aucs_5th["pl_auc_ave"].values, **style, label="p_auc_5th")
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="95th")
                    plt.plot(n_mix_list, df_aucs_95th["pl_auc_ave"].values, **style, label="p_auc_95th")
                if "fill" in self.sim_set.plt_auc_stat_style:
                    plt.fill_between(n_mix_list, df_aucs_5th["pl_auc_ave"].values, df_aucs_95th["pl_auc_ave"].values,
                                     alpha=0.2, color=style_m["color"], hatch="x")

        if ("ind" in self.sim_set.plt_auc_calc_type) and ("pl_auc" in self.sim_set.plt_auc_type):

            if ("ave" in self.sim_set.plt_auc_stat_type):
                for i in range(1, self.n_classes + 1):
                    # Plot average, min and max for "pl_auc_ind_i"
                    style_m = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="ave",
                                                          class_num=i)
                    plt.plot(n_mix_list, df_aucs_ave[f"pl_auc_ind_{i}"].values, **style_m, label=f"pl_auc_ind_{i} ")

                    if ("min" and "max" in self.sim_set.plt_auc_stat_type):
                        if "lines" in self.sim_set.plt_auc_stat_style:
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="min",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_min[f"pl_auc_ind_{i}"].values, **style,
                                     label=f"pl_auc_ind_{i}_min")
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="max",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_max[f"pl_auc_ind_{i}"].values, **style,
                                     label=f"pl_auc_ind_{i}_max")
                        if "fill" in self.sim_set.plt_auc_stat_style:
                            plt.fill_between(n_mix_list, df_aucs_min[f"pl_auc_ind_{i}"].values,
                                             df_aucs_max[f"pl_auc_ind_{i}"].values,
                                             alpha=0.2, color=style_m["color"], hatch=".")

                    if ("5th" and "95th" in self.sim_set.plt_auc_stat_type):
                        if "lines" in self.sim_set.plt_auc_stat_style:
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="5th",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_5th[f"pl_auc_ind_{i}"].values, **style,
                                     label=f"pl_auc_ind_{i}_5th")
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc",
                                                                auc_stat_type="95th",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_95th[f"pl_auc_ind_{i}"].values, **style,
                                     label=f"pl_auc_ind_{i}_95th")
                        if "fill" in self.sim_set.plt_auc_stat_style:
                            plt.fill_between(n_mix_list, df_aucs_5th[f"pl_auc_ind_{i}"].values,
                                             df_aucs_95th[f"pl_auc_ind_{i}"].values,
                                             alpha=0.2, color=style_m["color"], hatch="x")

        # ==============================================================================================================
        #                           Plotting test AUCs
        # ==============================================================================================================
        if ("ave" in self.sim_set.plt_auc_calc_type) and ("t_auc" in self.sim_set.plt_auc_type):

            if ("ave" in self.sim_set.plt_auc_stat_type):
                style_m = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="ave")
                plt.plot(n_mix_list, df_aucs_ave["t_auc_ave"].values, **style_m, label="t_auc_ave")

            if ("min" and "max" in self.sim_set.plt_auc_stat_type):
                if "lines" in self.sim_set.plt_auc_stat_style:
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="min")
                    plt.plot(n_mix_list, df_aucs_min["t_auc_ave"].values, **style, label="t_auc_min")
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="max")
                    plt.plot(n_mix_list, df_aucs_max["t_auc_ave"].values, **style, label="t_auc_max")
                if "fill" in self.sim_set.plt_auc_stat_style:
                    plt.fill_between(n_mix_list, df_aucs_min["t_auc_ave"].values, df_aucs_max["t_auc_ave"].values,
                                     alpha=0.2, color=style_m["color"], hatch="//")

            if ("5th" and "95th" in self.sim_set.plt_auc_stat_type):
                if "lines" in self.sim_set.plt_auc_stat_style:
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="5th")
                    plt.plot(n_mix_list, df_aucs_5th["t_auc_ave"].values, **style, label="t_auc_5th")
                    style = self.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="95th")
                    plt.plot(n_mix_list, df_aucs_95th["t_auc_ave"].values, **style, label="t_auc_95th")
                if "fill" in self.sim_set.plt_auc_stat_style:
                    plt.fill_between(n_mix_list, df_aucs_5th["t_auc_ave"].values, df_aucs_95th["t_auc_ave"].values,
                                     alpha=0.2, color=style_m["color"], hatch="|")

        if ("ind" in self.sim_set.plt_auc_calc_type) and ("t_auc" in self.sim_set.plt_auc_type):
            if ("ave" in self.sim_set.plt_auc_stat_type):

                for i in range(1, self.n_classes + 1):
                    style_m = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc", auc_stat_type="ave",
                                                          class_num=i)
                    plt.plot(n_mix_list, df_aucs_ave[f"t_auc_ind_{i}"].values, **style_m, label=f"t_auc_ind_{i}")

                    if ("min" and "max" in self.sim_set.plt_auc_stat_type):
                        if "lines" in self.sim_set.plt_auc_stat_style:
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                                auc_stat_type="min",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_min[f"t_auc_ind_{i}"].values, **style,
                                     label=f"t_auc_ind_{i}_min")
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                                auc_stat_type="max",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_max[f"t_auc_ind_{i}"].values, **style,
                                     label=f"t_auc_ind_{i}_max")
                        if "fill" in self.sim_set.plt_auc_stat_style:
                            plt.fill_between(n_mix_list, df_aucs_min[f"t_auc_ind_{i}"].values,
                                             df_aucs_max[f"t_auc_ind_{i}"].values,
                                             alpha=0.2, color=style_m["color"], hatch="//")

                    if ("5th" and "95th" in self.sim_set.plt_auc_stat_type):
                        if "lines" in self.sim_set.plt_auc_stat_style:
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                                auc_stat_type="5th",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_5th[f"t_auc_ind_{i}"].values, **style,
                                     label=f"t_auc_ind_{i}_5th")
                            style = self.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                                auc_stat_type="95th",
                                                                class_num=i)
                            plt.plot(n_mix_list, df_aucs_95th[f"t_auc_ind_{i}"].values, **style,
                                     label=f"t_auc_ind_{i}_95th")
                        if "fill" in self.sim_set.plt_auc_stat_style:
                            plt.fill_between(n_mix_list, df_aucs_5th[f"t_auc_ind_{i}"].values,
                                             df_aucs_95th[f"t_auc_ind_{i}"].values,
                                             alpha=0.2, color=style_m["color"], hatch="|")

        # Add labels and legends
        plt.grid(True)
        plt.ylim(self.sim_set.plt_y_lim[0], self.sim_set.plt_y_lim[1])
        plt.xlabel("n_mix -- " + "n_L = " + str(self.n_labeled) + " -- n_U = " + str(
            self.n_unlabeled) + " -- step = " + str(self.step_to_unlabeled_as_labeled))
        plt.xlabel("Number of mixed samples ($n_M$)")
        plt.ylabel("ROC_AUC")
        # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=6)
        plt.legend(loc='lower right', ncol=2)
        # plt.tight_layout(rect=[0, 0, 1, 0.9])
        return fig, df_aucs_ave, df_aucs_min, df_aucs_max, df_aucs_5th, df_aucs_95th, columns
class ClassificationModelGridSearch:
    def __init__(self, x_train, y_train):
        """
        Initializes the ClassificationModelGridSearch object.

        Args:
            x_train (array-like): The input features of the training data.
            y_train (array-like): The target labels of the training data.
        """
        self.x_train = x_train
        self.y_train = y_train

        self.classifiers = {
            "lp": LabelPropagation(),
            "ls": LabelSpreading(),
            "tsvm": scikitTSVM.SKTSVM,
            "cple": CPLELearningModel(SVC()),
            "qwda": CPLELearningModel(WQDA()),
            "svmrbf": SVC(),
            "svmlin": SVC(),
            "knn": KNeighborsClassifier(),
            "dt": DecisionTreeClassifier(),
            "gb": GradientBoostingClassifier(),
        }

        self.param_grid = {
            "lp": {"kernel": ["knn", "rbf"]},
            "ls": {"kernel": ["knn", "rbf"]},
            "tsvm": {"kernel": ["rbf"]},
            "cple": {"base_estimator__kernel": ["linear", "rbf"], "base_estimator__reg_param": np.logspace(-3, 1, 10)},
            "qwda": {"kernel": ["linear", "rbf"], "reg_param": np.logspace(-3, 1, 10)},
            "svmrbf": {"kernel": ["rbf"], "gamma": np.logspace(-3, 1, 10), "C": np.logspace(-3, 3, 10),
                       "probability": [True]},
            "svmlin": {"kernel": ["linear"], "C": np.logspace(-3, 3, 10), "probability": [True]},
            "knn": {"n_neighbors": [2, 5, 10], "weights": ["uniform", "distance"]},
            "dt": {"criterion": ["gini", "entropy"], "max_depth": [3, 5, 7]},
            "gb": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0], "max_depth": [3, 5, 7]},
        }

    def perform_grid_search(self, selected_classifiers):
        """
        Performs grid search for the selected classifiers using cross-validation.

        Args:
            selected_classifiers (list): The list of classifiers to perform grid search on.

        Returns:
            dict: A dictionary containing the best parameters for each selected classifier.
        """
        best_params = {}
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)

        for clf_name, clf in self.classifiers.items():
            if clf_name in selected_classifiers:
                param_grid = self.param_grid[clf_name]
                if "rbf" in param_grid.get("kernel", []):
                    param_grid["gamma"] = np.logspace(-3, 1, 10)
                grid = GridSearchCV(clf, param_grid=param_grid, cv=cv)
                grid.fit(self.x_train, self.y_train)
                best_params[clf_name] = grid.best_params_
        return best_params


class ClassificationModelParameters:
    """
    A class to manage classification model parameters.

    Attributes:
        lp_params (dict): Parameters for LabelPropagation model.
        ls_params (dict): Parameters for LabelSpreading model.
        tsvm_params (dict): Parameters for TSVM model.
        cple_params (dict): Parameters for CPLE model.
        qwda_params (dict): Parameters for QWDA model.
        svmrbf_params (dict): Parameters for SVM-RBF model.
        svmlin_params (dict): Parameters for SVM-Linear model.
        knn_params (dict): Parameters for KNN model.
        dt_params (dict): Parameters for Decision Tree model.
        gb_params (dict): Parameters for Gradient Boosting model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the ClassificationModelParameters object.

        Args:
            **kwargs: Keyword arguments to set the model parameters.
        """
        self.lp_params = kwargs.get("lp_params", {})
        self.ls_params = kwargs.get("ls_params", {})
        self.tsvm_params = kwargs.get("tsvm_params", {})
        self.cple_params = kwargs.get("cple_params", {})
        self.qwda_params = kwargs.get("qwda_params", {})
        self.svmrbf_params = kwargs.get("svmrbf_params", {})
        self.svmlin_params = kwargs.get("svmlin_params", {})
        self.knn_params = kwargs.get("knn_params", {})
        self.dt_params = kwargs.get("dt_params", {})
        self.gb_params = kwargs.get("gb_params", {})

    def get_model_parameters(self):
        """
        Retrieves the model parameters as a dictionary.

        Returns:
            dict: A dictionary of model parameters.
        """
        model_parameters = {}
        for attr_name in self.__dict__:
            if attr_name != "get_model_parameters":
                model_parameters[attr_name] = getattr(self, attr_name)
        return model_parameters

    def get_ss_method_list(self, pseudo_label_model_list, test_model_list):
        """
        Retrieves the list of semi-supervised methods.

        Returns:
            list: A list of semi-supervised methods.
        """

        tuple1 = []
        tuple2 = []

        for pl_model_i in pseudo_label_model_list:
            for t_model_i in test_model_list:
                if pl_model_i in ["lp", "ls", "tsvm"]:
                    tuple1.append((pl_model_i, t_model_i))
                elif pl_model_i == t_model_i:
                    tuple2.append((pl_model_i, t_model_i))

        pl_test_model_list = tuple1 + tuple2
        return pl_test_model_list


class SemiSupervisedClassification:
    def __init__(self,
                 n_labeled,
                 n_mix,
                 n_rand_i,
                 n_classes,
                 model_params,
                 pseudo_label_model,
                 test_model,
                 dataset,
                 data_type,
                 print_results):
        self.n_labeled = n_labeled
        self.n_mix = n_mix
        self.n_rand_i = n_rand_i
        self.n_classes = n_classes
        self.model_params = model_params

        # ******************************************************************************************************
        self.pseudo_label_model = pseudo_label_model
        self.test_model = test_model
        self.dataset = dataset
        self.data_type = data_type
        self.print_results = print_results
        self.d_train, self.d_test, self.d_train_true = self.get_low_dim_representations()
        """
        Initialize a SemiSupervisedClassification object.

        Parameters:
            n_labeled (int): Number of labeled samples.
            n_classes (int): Number of classes.
            auc_calc_type (str): Type of AUC calculation ("average" or "individual").
            model_params (dict): Parameters for the models.
            pseudo_label_model (str): Name of the pseudo label model.
            test_model (str): Name of the test model.
            d_train: Training dataset (labeled + unlabeled).
            d_test: Test dataset.
            d_train_true: Training dataset with true labels (labeled + unlabeled).
            print_results (bool): Whether to print the results.
        """

    def initialize_pseudo_label_model(self):
        """
        Initialize the pseudo label model.

        Returns:
            object: Initialized pseudo label model.
        """
        if self.pseudo_label_model == "svmrbf":
            basedclassifier = SVC(**self.model_params["svmrbf_params"])
            self.pl_model = SelfTrainingClassifier(basedclassifier)
        elif self.pseudo_label_model == "svmlin":
            basedclassifier = SVC(**self.model_params["svmlin_params"])
            self.pl_model = SelfTrainingClassifier(basedclassifier)
        elif self.pseudo_label_model == "knn":
            baseclassifier = KNeighborsClassifier(**self.model_params["knn_params"])
            self.pl_model = SelfTrainingClassifier(baseclassifier)
        elif self.pseudo_label_model == "dt":
            baseclassifier = DecisionTreeClassifier(**self.model_params["dt_params"])
            self.pl_model = SelfTrainingClassifier(baseclassifier)
        elif self.pseudo_label_model == "gb":
            baseclassifier = GradientBoostingClassifier(**self.model_params["gb_params"])
            self.pl_model = SelfTrainingClassifier(baseclassifier)
        if self.print_results:
            print(self.pseudo_label_model)
        return self.pl_model

    def initialize_test_model(self):
        """
        Initialize the test model.

        Returns:
            object: Initialized test model.
        """
        if self.test_model == "svmrbf":
            self.t_model = SVC(**self.model_params["svmrbf_params"])
        elif self.test_model == "svmlin":
            self.t_model = SVC(**self.model_params["svmlin_params"])
        elif self.test_model == "knn":
            self.t_model = KNeighborsClassifier(**self.model_params["knn_params"])
        elif self.test_model == "dt":
            self.t_model = DecisionTreeClassifier(**self.model_params["dt_params"])
        elif self.test_model == "gb":
            self.t_model = GradientBoostingClassifier(**self.model_params["gb_params"])
        if self.print_results:
            print(self.test_model)
        return self.t_model

    def initialize_ss_method(self):

        if self.pseudo_label_model == "lp":
            self.pl_model = LabelPropagation(**self.model_params["lp_params"])
            self.t_model = self.initialize_test_model()
        elif self.pseudo_label_model == "ls":
            self.pl_model = LabelSpreading(**self.model_params["ls_params"])
            self.t_model = self.initialize_test_model()
        elif self.pseudo_label_model == "tsvm":
            self.pl_model = scikitTSVM.SKTSVM(**self.model_params["tsvm_params"])
            self.t_model = self.initialize_test_model()
        elif (self.pseudo_label_model, self.test_model) in \
                [("svmrbf", "svmrbf"), ("svmlin", "svmlin"), ("knn", "knn"), ("dt", "dt"), ("gb", "gb")]:
            self.pl_model = self.initialize_pseudo_label_model()
            self.t_model = self.initialize_test_model()

        # else: # **************************
        #     raise ValueError("Invalid semi-supervised method") # **************************
        if self.print_results:
            print(self.pseudo_label_model)
            print(self.t_model)
        return self.pl_model, self.t_model

    def get_low_dim_representations(self):
        self.d_train, self.d_test, self.d_train_true = self.dataset.get_lowdim_dataset(n_rand_i=self.n_rand_i,
                                                                                       n_mix=self.n_mix,
                                                                                       data_type=self.data_type)

        return self.d_train, self.d_test, self.d_train_true

    def evaluate_semisupervised_classification(self):

        # When the unlabaled samples are included in the training set
        if self.n_labeled < len(self.d_train.x):
            if self.pseudo_label_model in ["lps", "lss"]: # **************************
                self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_mean, self.pl_auc_mean, \
                self.t_auc_ind, self.pl_auc_ind, self.t_model = self.semisupervised_classification_self() # **************************
            else:
                self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_mean, self.pl_auc_mean, \
                self.t_auc_ind, self.pl_auc_ind, self.t_model = self.semisupervised_classification_mix()
        # When only the labeled samples are included in the training set
        else:
            self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_mean, self.pl_auc_mean, \
            self.t_auc_ind, self.pl_auc_ind, self.t_model = self.semisupervised_classification_labeled()

        return self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_ave, self.pl_auc_ave, \
               self.t_auc_ind, self.pl_auc_ind, self.t_model, \
               self.d_train, self.d_test, self.d_train_true

    def semisupervised_classification_labeled(self):
        _, self.t_model = self.initialize_ss_method()
        self.y_pseudo = self.d_train.y.astype(int)

        self.pl_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y),
                                        self.get_one_hot_encode(self.y_pseudo),
                                        multi_class="ovr", average="weighted")

        self.pl_auc_ind = []
        for i in range(self.n_classes):
            pl_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y)[:, i],
                                     self.get_one_hot_encode(self.y_pseudo)[:, i])
            self.pl_auc_ind.append(pl_auc_i)

        self.t_model.fit(self.d_train.x, self.y_pseudo)
        self.y_pred = self.t_model.predict(self.d_test.x)
        self.y_pred_prob = self.t_model.predict_proba(self.d_test.x)

        self.t_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_test.y), self.y_pred_prob,
                                       multi_class="ovr", average="weighted")

        self.t_auc_ind = []
        for i in range(self.n_classes):
            t_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_test.y)[:, i], self.y_pred_prob[:, i])
            self.t_auc_ind.append(t_auc_i)

        if self.print_results:
            self.get_results_to_print()

        return self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_ave, self.pl_auc_ave, \
               self.t_auc_ind, self.pl_auc_ind, self.t_model

    def semisupervised_classification_mix(self):
        self.pl_model, self.t_model = self.initialize_ss_method()
        self.pl_model.fit(self.d_train.x, self.d_train.y)
        self.y_pseudo = self.pl_model.predict(self.d_train.x).astype(int)

        self.y_pseudo_prob = self.pl_model.predict_proba(self.d_train.x)
        indices = np.where(self.d_train.y != -1)
        indices_u = np.where(self.d_train.y == -1)
        self.y_pseudo[indices] = self.d_train.y[indices]
        self.pl_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y[indices_u]),
                                        self.y_pseudo_prob[indices_u],
                                        multi_class="ovr", average="weighted")

        self.pl_auc_ind = []
        for i in range(self.n_classes):
            pl_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y[indices_u])[:, i],
                                     self.y_pseudo_prob[indices_u][:, i])
            self.pl_auc_ind.append(pl_auc_i)

        self.t_model.fit(self.d_train.x, self.y_pseudo)
        self.y_pred = self.t_model.predict(self.d_test.x)
        self.y_pred_prob = self.t_model.predict_proba(self.d_test.x)

        self.t_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_test.y), self.y_pred_prob,
                                       multi_class="ovr", average="weighted")

        self.t_auc_ind = []
        for i in range(self.n_classes):
            t_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_test.y)[:, i], self.y_pred_prob[:, i])
            self.t_auc_ind.append(t_auc_i)

        if self.print_results:
            self.get_results_to_print()

        return self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_ave, self.pl_auc_ave, \
               self.t_auc_ind, self.pl_auc_ind, self.t_model

    def semisupervised_classification_self(self):
        self.pl_model, self.t_model = self.initialize_ss_method()
        self.pl_model.fit(self.d_train.x, self.d_train.y)
        self.y_pseudo = self.pl_model.predict(self.d_train.x).astype(int)
        self.y_pseudo_prob = self.pl_model.predict_proba(self.d_train.x)

        indices = np.where(self.d_train.y != -1)
        self.y_pseudo[indices] = self.d_train.y[indices]
        self.y_pseudo_prob = np.nan_to_num(self.y_pseudo_prob, nan=0.0, posinf=1.0, neginf=0.0)
        self.pl_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y), self.y_pseudo_prob,
                                        multi_class="ovr", average="weighted")

        self.pl_auc_ind = []
        for i in range(self.n_classes):
            pl_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y)[:, i],
                                     self.y_pseudo_prob[:, i])
            self.pl_auc_ind.append(pl_auc_i)

        self.t_model.fit(self.d_train.x, self.y_pseudo)
        self.y_pred = self.t_model.predict(self.d_test.x)
        self.y_pred_prob = self.t_model.predict_proba(self.d_test.x)


        self.y_pred_prob = np.nan_to_num(self.y_pred_prob, nan=0.0, posinf=1.0, neginf=0.0)
        self.t_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_test.y), self.y_pred_prob,
                                       multi_class="ovr", average="weighted")

        self.t_auc_ind = []
        for i in range(self.n_classes):
            t_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_test.y)[:, i], self.y_pred_prob[:, i])
            self.t_auc_ind.append(t_auc_i)

        if self.print_results:
            self.get_results_to_print()

        return self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_ave, self.pl_auc_ave, \
               self.t_auc_ind, self.pl_auc_ind, self.t_model


    def semisupervised_classification_same_pl_t_model(self):
        self.pl_model, self.t_model = self.initialize_ss_method()
        self.pl_model.fit(self.d_train.x, self.d_train.y)
        self.y_pseudo = self.pl_model.predict(self.d_train.x).astype(int)
        self.y_pseudo_prob = self.pl_model.predict_proba(self.d_train.x)

        indices = np.where(self.d_train.y != -1)
        self.y_pseudo[indices] = self.d_train.y[indices]
        self.y_pseudo_prob = np.nan_to_num(self.y_pseudo_prob, nan=0.0, posinf=1.0, neginf=0.0)
        self.pl_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y), self.y_pseudo_prob,
                                        multi_class="ovr", average="weighted")

        self.pl_auc_ind = []
        for i in range(self.n_classes):
            pl_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y)[:, i],
                                     self.y_pseudo_prob[:, i])
            self.pl_auc_ind.append(pl_auc_i)

        # Append self.d_test.x to self.d_train.x
        x_train_aug = np.concatenate((self.d_train.x, self.d_test.x))

        # Create a vector of -1s with length equal to the length of self.d_test.x
        minus_ones = np.full((len(self.d_test.x),), -1)

        # Append the vector of -1s to self.y_pseudo
        y_train_aug = np.concatenate((self.y_pseudo, minus_ones))

        # Now you can fit and predict as before
        self.pl_model.fit(x_train_aug, y_train_aug)
        self.y_pred = self.pl_model.predict(x_train_aug)
        self.y_pred_prob = self.pl_model.predict_proba(x_train_aug)

        indices = np.where(y_train_aug != -1)
        self.y_pred[indices] = y_train_aug[indices]
        self.y_pred_prob = np.nan_to_num(self.y_pseudo_prob, nan=0.0, posinf=1.0, neginf=0.0)
        self.pl_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y), self.y_pseudo_prob,
                                        multi_class="ovr", average="weighted")

        self.pl_auc_ind = []
        for i in range(self.n_classes):
            pl_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_train_true.y)[:, i],
                                     self.y_pseudo_prob[:, i])
            self.pl_auc_ind.append(pl_auc_i)






        self.y_pred_prob = np.nan_to_num(self.y_pred_prob, nan=0.0, posinf=1.0, neginf=0.0)
        self.t_auc_ave = roc_auc_score(self.get_one_hot_encode(self.d_test.y), self.y_pred_prob,
                                       multi_class="ovr", average="weighted")

        self.t_auc_ind = []
        for i in range(self.n_classes):
            t_auc_i = roc_auc_score(self.get_one_hot_encode(self.d_test.y)[:, i], self.y_pred_prob[:, i])
            self.t_auc_ind.append(t_auc_i)

        if self.print_results:
            self.get_results_to_print()

        return self.y_pseudo, self.y_pred, self.y_pred_prob, self.t_auc_ave, self.pl_auc_ave, \
               self.t_auc_ind, self.pl_auc_ind, self.t_model

    def get_one_hot_encode(self, y):
        encoder = OneHotEncoder(sparse=False)
        y_ohe = encoder.fit_transform(y.reshape(-1, 1))
        return y_ohe

    def get_one_hot_encode(self, y):
        encoder = OneHotEncoder(sparse=False)
        y_ohe = encoder.fit_transform(y.reshape(-1, 1))
        return y_ohe

    def get_results_to_print(self):
        print("------------------")
        print(" len(self.y_pseudo) = ")
        print("------------------")
        print(len(self.y_pseudo))
        print("------------------")
        print(" len(self.d_train.x) = ")
        print("------------------")
        print(len(self.d_train.x))
        print("------------------")
        print(" len(self.d_test.x) =")
        print("------------------")
        print(len(self.d_test.x))
        print("------------------")
        print(" len(d_test.y) = ")
        print("------------------")
        print(len(self.d_test.y))
        print("------------------")
        print(" y_pred_prob = ")
        print("------------------")
        print(self.y_pred_prob)


class PsuedoLabelingAnimation:
    def __init__(self, dataset, version):
        """
        Initialize the PsuedoLabelingAnimation class.

        Parameters:
            dataset (DatasetGenerator): DatasetGenerator object containing the dataset.
            version (str): versionion identifier.
        """
        self.x_train_l, self.y_train_l = dataset.d_train_l.x, dataset.d_train_l.y
        self.x_test, self.y_test = dataset.d_test.x, dataset.d_test.y
        self.idx_mi_l = dataset.idx_mi_l
        self.n_classes = len(np.unique(self.y_train_l))
        self.colors = self.generate_colors()
        self.frames = []
        self.version = version

    def generate_colors(self):
        """
        Generate a list of distinct colors for different classes.

        Returns:
            list: List of colors.
        """
        color_cycle = ["r", "k", "g", "y", "b", "m"]
        if self.n_classes <= len(color_cycle):
            colors = color_cycle[:2 * self.n_classes]
        else:
            colormap = plt.cm.get_cmap("tab20", self.n_classes)
            colors = colormap(range(self.n_classes))
        return colors

    def get_color(self, class_index):
        """
        Get the color for a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            str: Color code.
        """
        return self.colors[class_index]

    def get_train_l_style(self, class_index):
        """
        Get the style for plotting labeled training samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting labeled training samples.
        """
        color = self.get_color(class_index)
        return {"marker": "o", "edgecolors": color, "facecolors": color, "label": "L-c {}".format(class_index)}

    def get_train_u_true_style(self, class_index):
        """
        Get the style for plotting true labels of unlabeled training samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting true labels of unlabeled training samples.
        """
        color = self.get_color(class_index)
        return {"marker": "o", "edgecolors": color, "facecolors": "none", "label": "U-True-c {}".format(class_index)}

    def get_train_u_pseudo_style(self, class_index):
        """
        Get the style for plotting pseudo labels of unlabeled training samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting pseudo labels of unlabeled training samples.
        """
        color = self.get_color(class_index)
        return {"marker": "*", "edgecolors": "none", "facecolors": color, "label": "U-Pseudo-c {}".format(class_index)}

    def get_test_true_style(self, class_index):
        """
        Get the style for plotting true labels of test samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting true labels of test samples.
        """
        color = self.get_color(- (class_index + 1))
        return {"marker": "s", "edgecolors": color, "facecolors": "none", "label": "T-true-c {}".format(class_index)}

    def get_test_pred_style(self, class_index):
        """
        Get the style for plotting predicted labels of test samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting predicted labels of test samples.
        """
        color = self.get_color(- (class_index + 1))
        return {"marker": "*", "edgecolors": "none", "facecolors": color, "label": "T-pred-c {}".format(class_index)}

    def create_empty_plots(self):
        """
        Create empty plots for animation.

        Returns:
            tuple: A tuple containing the figure, scatter plot, and axes objects.
        """
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.ss_plot = self.ax.scatter([], [])
        return self.fig, self.ss_plot, self.ax

    def init_animation(self):
        """
        Initialize the animation by plotting the labeled training samples and test samples.

        Returns:
            Axes: Axes object.
        """
        self.ax.clear()  # Clear the axes for each frame
        for class_i in range(self.n_classes):
            self.ax.scatter(self.x_train_l[self.y_train_l == class_i, 0], self.x_train_l[self.y_train_l == class_i, 1],
                            **self.get_train_l_style(class_i), s=80)
            self.ax.scatter(self.x_test[self.y_test == class_i, 0], self.x_test[self.y_test == class_i, 1],
                            **self.get_test_true_style(class_i), s=100)
        return self.ax


class PsuedoLabelingAnimation:
    def __init__(self, dataset, version):
        """
        Initialize the PsuedoLabelingAnimation class.

        Parameters:
            dataset (DatasetGenerator): DatasetGenerator object containing the dataset.
            version (str): versionion identifier.
        """
        self.x_train_l, self.y_train_l = dataset.d_train_l.x, dataset.d_train_l.y
        self.x_test, self.y_test = dataset.d_test.x, dataset.d_test.y
        self.idx_mi_l = dataset.idx_mi_l
        self.n_classes = len(np.unique(self.y_train_l))
        self.colors = self.generate_colors()
        self.frames = []
        self.version = version

    def generate_colors(self):
        """
        Generate a list of distinct colors for different classes.

        Returns:
            list: List of colors.
        """
        color_cycle = ["r", "k", "g", "y", "b", "m"]
        if self.n_classes <= len(color_cycle):
            colors = color_cycle[:2 * self.n_classes]
        else:
            colormap = plt.cm.get_cmap("tab20", self.n_classes)
            colors = colormap(range(self.n_classes))
        return colors

    def get_color(self, class_index):
        """
        Get the color for a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            str: Color code.
        """
        return self.colors[class_index]

    def get_train_l_style(self, class_index):
        """
        Get the style for plotting labeled training samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting labeled training samples.
        """
        color = self.get_color(class_index)
        return {"marker": "o", "edgecolors": color, "facecolors": color, "label": "L-c {}".format(class_index)}

    def get_train_u_true_style(self, class_index):
        """
        Get the style for plotting true labels of unlabeled training samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting true labels of unlabeled training samples.
        """
        color = self.get_color(class_index)
        return {"marker": "o", "edgecolors": color, "facecolors": "none",
                "label": "U-True-c {}".format(class_index)}

    def get_train_u_pseudo_style(self, class_index):
        """
        Get the style for plotting pseudo labels of unlabeled training samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting pseudo labels of unlabeled training samples.
        """
        color = self.get_color(class_index)
        return {"marker": "*", "edgecolors": "none", "facecolors": color,
                "label": "U-Pseudo-c {}".format(class_index)}

    def get_test_true_style(self, class_index):
        """
        Get the style for plotting true labels of test samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting true labels of test samples.
        """
        color = self.get_color(- (class_index + 1))
        return {"marker": "s", "edgecolors": color, "facecolors": "none",
                "label": "T-true-c {}".format(class_index)}

    def get_test_pred_style(self, class_index):
        """
        Get the style for plotting predicted labels of test samples of a specific class.

        Parameters:
            class_index (int): Index of the class.

        Returns:
            dict: Style parameters for plotting predicted labels of test samples.
        """
        color = self.get_color(- (class_index + 1))
        return {"marker": "*", "edgecolors": "none", "facecolors": color,
                "label": "T-pred-c {}".format(class_index)}

    def create_empty_plots(self):
        """
        Create empty plots for animation.

        Returns:
            tuple: A tuple containing the figure, scatter plot, and axes objects.
        """
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.ss_plot = self.ax.scatter([], [])
        return self.fig, self.ss_plot, self.ax

    def init_animation(self):
        """
        Initialize the animation by plotting the labeled training samples and test samples.

        Returns:
            Axes: Axes object.
        """
        self.ax.clear()  # Clear the axes for each frame
        for class_i in range(self.n_classes):
            self.ax.scatter(self.x_train_l[self.y_train_l == class_i, 0],
                            self.x_train_l[self.y_train_l == class_i, 1],
                            **self.get_train_l_style(class_i), s=80)
            self.ax.scatter(self.x_test[self.y_test == class_i, 0], self.x_test[self.y_test == class_i, 1],
                            **self.get_test_true_style(class_i), s=100)
        return self.ax

    def animate_pseudo_labels(self, test_model, d_train, d_test, d_train_true, y_pseudo, y_pred, n_mix, frame_cntr,
                              n_rand_i, h=0.02):
        """
        Animate the pseudo labeling process by plotting the labeled and unlabeled samples.

        Parameters:
            test_model: The trained model for prediction.
            d_train: Training dataset (labeled + unlabeled).
            d_test: Test dataset.
            d_train_true: Training dataset with true labels (labeled + unlabeled).
            y_pseudo: Pseudo labels of the unlabeled samples.
            y_pred: Predicted labels of the test samples.
            n_mix: Number of mixed labeled and unlabeled samples.
            frame_cntr: Frame counter.
            n_rand_i: Random index for different versions of the animation.
            h: Step size for creating the contour plot.

        Returns:
            tuple: A tuple containing the Axes object and the frames of the animation.
        """
        x_train_u, y_train_u_true, y_train_u_pseudo = d_train_true.x[d_train.y == -1], d_train_true.y[d_train.y == -1], \
                                                      y_pseudo[d_train.y == -1]

        if x_train_u.shape[1] > 2:
            print("To be able to plot the decision boundaries, the dimension of the data should be 2.")
        else:
            X = np.vstack((d_train.x, d_test.x))
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = test_model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Define the colors for each class
            class_colors = ["orange", "gray", "green"]

            # Create a colormap using the specified colors
            cmap = colors.ListedColormap(class_colors)

            # Plot the decision boundaries with the specified colormap using pcolormesh
            self.ax.pcolormesh(xx, yy, Z, cmap=cmap, alpha=0.2, shading="auto")

        for class_i in range(self.n_classes):
            self.ax.scatter(x_train_u[y_train_u_true == class_i, 0], x_train_u[y_train_u_true == class_i, 1],
                            **self.get_train_u_true_style(class_i), s=100)
            self.ax.scatter(x_train_u[y_train_u_pseudo == class_i, 0], x_train_u[y_train_u_pseudo == class_i, 1],
                            **self.get_train_u_pseudo_style(class_i), s=80)
            self.ax.scatter(self.x_test[y_pred == class_i, 0], self.x_test[y_pred == class_i, 1],
                            **self.get_test_pred_style(class_i), s=80)

        # Adjust figure size to accommodate the legend
        self.fig.set_size_inches(12, 8)
        # Set xlim and ylim in self.ax
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_title(" n_mix = " + str(n_mix))
        bbox_to_anchor = None
        if n_rand_i == 0:
            legend = self.ax.legend(loc="lower center", bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True,
                                    ncol=5)
            # Adjust font properties of legend text
            for text in legend.get_texts():
                text.set_fontsize(10)  # Set the font size as needed
                text.set_fontweight("normal")  # Set the font weight (e.g., "normal", "bold", "light", etc.)
                text.set_fontstyle("italic")  # Set the font style (e.g., "normal", "italic", "oblique")

            legend_without_duplicate_labels(self.ax, loc="lower center", bbox_to_anchor=bbox_to_anchor, fancybox=True,
                                            shadow=True, ncol=5)

        for i, j in enumerate(self.ax.collections):
            j.set_linestyle("dashed")

        # Define the directory path for the animations folder
        animations_dir = os.path.join("results_final", self.version, "animations")

        # Create the animations folder if it doesn't exist
        os.makedirs(animations_dir, exist_ok=True)

        # Construct the file path within the animations folder
        fname_to_save = f"anim_plt_{frame_cntr + 1}_{n_rand_i}.png"
        fig_path = os.path.join(animations_dir, fname_to_save)

        # Save the figure in the specified file path
        self.fig.savefig(fig_path)
        plt.close()
        # Open the image using the saved file path
        with Image.open(fig_path) as frame_image:
            # Append the opened image to the frames list
            self.frames.append(frame_image)

        return self.ax, self.frames

    def save_animation(self, n_frames, data_type, fold_num, set_num, n_rand,
                       pseudo_label_model, test_model, version):
        """
        Save the animation as a GIF file.

        Parameters:
            n_frames (int): Number of frames in the animation.
            data_type (str): Type of data used for animation.
            fold_num (int): Fold number.
            set_num (int): Set number.
            n_rand (int): Number of random versions of the animation.
            n_rand_i (int): Index of the random version.
            pseudo_label_model (str): Name of the pseudo label model.
            test_model (str): Name of the test model.
            version (str): versionion of the algorithm.
        """
        # Define the directory path for the animations folder
        animations_dir = os.path.join("results_final", self.version, "animations")


        # Construct the file path within the animations folder
        fname_to_save = f"animation-{pseudo_label_model}-{test_model}-dt{data_type}-fn{fold_num}-sn{set_num}-n{n_rand}.gif"
        anim_path = os.path.join(animations_dir, fname_to_save)

        print("**********************************************")
        print("Saving animation to:", anim_path)
        print("**********************************************")

        # Construct the file paths for all frames within the results_final directory
        frame_paths = [
            os.path.join(animations_dir, f"anim_plt_{i + 1}_0.png") for i in range(n_frames)
        ]


        # Create a list to store references to the opened file objects
        opened_files = []

        try:
            # Open all frames and append them to the frames_all list
            frames_all = [Image.open(frame_path) for frame_path in frame_paths]

            # Add references to the opened file objects to the opened_files list
            opened_files.extend(frames_all)

            durations = [1000] * (n_frames - 1) + [3000]  # Set last frame duration to 3 seconds

            frames_all[0].save(anim_path,
                               save_all=True,
                               append_images=frames_all[1:],
                               duration=durations,
                               loop=0,
                               optimize=True)
        finally:
            # Close all opened file objects in the finally block
            for file in opened_files:
                file.close()
=


# ======================================================================================================================
def generate_random_train_test_from_sklearn_split(x_data, y_data, test_size):
    """
    This function takes the set of all the available samples (x_data, y_data) and
    using the built-in function of sklearn package to randomly generate train and test datasets
    """
    indices = np.arange(x_data.shape[0])
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(x_data, y_data, indices,
                                                                             test_size=test_size)
    return x_train, x_test, y_train, y_test, idx_train, idx_test


min_max_scaler = preprocessing.MinMaxScaler()


def replace_inf_with_max_of_vector(array):
    n, m = array.shape
    array[np.isinf(array)] = -np.inf
    mx_array = np.repeat(np.max(array, axis=1), m).reshape(n, m)
    ind = np.where(np.isinf(array))
    array[ind] = mx_array[ind]
    return array


def constant_columns_filter(x_train_temp):
    # x_train_temp = x_train.copy()
    x_train = pd.DataFrame(x_train_temp)
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(x_train)
    constant_columns = [column for column in x_train.columns
                        if column not in
                        x_train.columns[constant_filter.get_support()]]

    x_train = constant_filter.transform(x_train)
    # x_test = constant_filter.transform(x_test)
    return constant_columns







def filter_method(number_of_bootstrap_in_filter_method, number_of_selected_features, x_train_temp, y_train):
    # TODO: Modify this function variables
    """
    Finding the correlation between features and the target variable
    """
    # N = x_train_temp.shape[0]  # Number of samples in the train data
    # x_train = constant_columns_filter(x_train_temp)
    N = x_train_temp.shape[0]  # Number of samples in the train data
    d_temp = x_train_temp.shape[1]
    idx_orig = np.array(np.arange(0, d_temp))
    idx_c_c = constant_columns_filter(x_train_temp)  # constant_columns
    idx_u_c_c = np.array(list(set(idx_orig) - set(idx_c_c)))
    x_train = x_train_temp[:, idx_u_c_c]
    N = x_train.shape[0]  # Number of samples in the train data
    d = x_train.shape[1]  # Number of features in the train data
    # Initialization of rank matrices to rank the features in each bootstrap
    R_mi = np.zeros([d, number_of_bootstrap_in_filter_method])
    # Initialization of matrices corresponding to the score of the features in each bootstrap
    S_mi = np.zeros([d, number_of_bootstrap_in_filter_method])
    # Initialization of matrices corresponding to the sorted score of the features in each bootstrap
    S_mi_sorted = np.zeros([d, number_of_bootstrap_in_filter_method])
    # Initialization of matrices corresponding to the sorted normalized score of the features in each bootstrap
    S_mi_norm = np.zeros([d, number_of_bootstrap_in_filter_method])

    idx_train = np.arange(0, N)  # index of features in the training dataset

    for b_i in range(number_of_bootstrap_in_filter_method):
        idx_b_i = random.choices(idx_train,
                                 k=len(idx_train))  # Randomly choosing a subset of features in each bootstrap
        print("Filter b_i = " + str(b_i))
        # Compute Mutual Information
        Xy_train_MI = mutual_info_classif(x_train[idx_b_i, :], y_train[idx_b_i])
        # Sort MI values
        Feature_sorted_MI = np.argsort(Xy_train_MI)[::-1]

        # Feature Scores
        S_mi[:, b_i] = Xy_train_MI

        # Selecting highly correlated features
        R_mi[:, b_i] = Feature_sorted_MI[0:d]

    # replaceing the inf values with the max of vectors
    S_mi = replace_inf_with_max_of_vector(S_mi)
    # print(S_mi)
    # Select d" features with maximum number of occurrence in the bootstrap data
    for d_i in range(d):
        v_mi = S_mi[d_i, :]
        v_mi_scaled = min_max_scaler.fit_transform(v_mi.reshape(-1, 1))
        S_mi_norm[d_i, :] = v_mi_scaled[:, 0]

    for d_i in range(d):
        S_mi_arg_sorted_temp = np.argsort(S_mi[d_i, :])[::-1]
        S_mi_sorted[d_i, :] = S_mi[d_i, S_mi_arg_sorted_temp]

    P95_S_mi = np.percentile(S_mi_sorted, 95, axis=1)

    EFS_sorted_MI = np.argsort(P95_S_mi)[::-1]

    idx_mi = EFS_sorted_MI[0:number_of_selected_features]  # Index of selected features based on MI value
    return idx_mi


def feature_mapping(x_data):
    # Define the column indices for each subset
    # Define the column indices for each subset
    # Define the column indices for each subset
    idx_1 = np.concatenate((np.arange(0, 3), np.arange(126, 129), np.arange(252, 255)))
    idx_2 = np.concatenate((np.arange(2, 5), np.arange(129, 132), np.arange(255, 258)))
    idx_3 = np.concatenate((np.arange(46, 126), np.arange(192, 252), np.arange(318, 378)))
    idx_4 = np.concatenate((np.arange(6, 66), np.arange(132, 192), np.arange(258, 318)))
    x_data[:, idx_1] = np.cos(x_data[:, idx_1])
    x_data[:, idx_2] = np.exp(x_data[:, idx_2])
    x_data[:, idx_3] = np.cos(x_data[:, idx_3])
    x_data[:, idx_4] = np.log(x_data[:, idx_4])
    return x_data


def create_results_folder(folder_name):
    # Get the current working directory
    current_directory = os.getcwd()
    # Specify the "results_final" folder name
    results_folder = "results_final"
    # Create the path for the "results_final" folder
    results_folder_path = os.path.join(current_directory, results_folder)

    # Check if the "results_final" folder already exists, and create it if not
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    # Create the path for the new folder inside "results_final"
    new_folder_path = os.path.join(results_folder_path, folder_name)

    # Create the new folder
    os.makedirs(new_folder_path)


# -------------Decision Boundary Plot Function
def plot_decision_boundaries(clf, x_data, x_train_lu, y_train_lu, x_test, y_test, h):
    X = np.vstack((x_train_lu, x_test))
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    color_map = {-1: (0, 1, 0), 0: (0, 0, 0.9), 1: (1, 0, 0)}
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.binary, alpha=0.7)
    plt.axis("off")

    # Plot also the training points
    colors_train = [color_map[y] for y in y_train_lu]
    colors_test = [color_map[y] for y in y_test]
    # plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="black")
    plt.scatter(x_train_lu[:, 0], x_train_lu[:, 1], c=colors_train)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=colors_test, marker="*")
    # plt.suptitle("Unlabeled points are colored white", y=0.1)
    plt.show()


def remove_nan(array1, array2):
    nan_indices = np.where(np.isnan(array1))[0]
    array1 = array1[nan_indices]
    array2 = array2[nan_indices]
    return array1, array2


def legend_without_duplicate_labels(ax, loc="upper center", bbox_to_anchor=None, **kwargs):
    """
    This function eliminates duplicate legends when plotting in a loop
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

    if bbox_to_anchor is None:
        ax.legend(*zip(*unique), loc=loc, **kwargs)
    else:
        ax.legend(*zip(*unique), bbox_to_anchor=bbox_to_anchor, **kwargs)


def plot_PCA(clf, x_train, y_train, x_test, y_test, figure_number):
    X_set = np.vstack((x_train, x_test))
    # y_set = np.vstack((y_train, y_test))
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 0.2,
                                   stop=X_set[:, 0].max() + 0.2, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 0.2,
                                   stop=X_set[:, 1].max() + 0.2, step=0.01))
    color_map = {-1: (0, 1, 0), 0: (0, 0, 0.9), 1: (1, 0, 0)}
    colors_train = [color_map[y] for y in y_train]
    colors_test = [color_map[y] for y in y_test]
    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(),
                                               X2.ravel()]).T).reshape(X1.shape), alpha=0.2,
                 cmap=ListedColormap(("white", "yellow")))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    # for i, j in enumerate(np.unique(y_train)):
    #     plt.scatter(x_train[y_train == j, 0], x_train[y_train== j, 1],
    #                 c=ListedColormap(("green", "red" , "blue"))(i), label=j)
    plt.figure(figure_number)
    # title for scatter plot
    plt.scatter(x_train[:, 0], x_train[:, 1], c=colors_train)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=colors_test, marker="*")
    # plt.title("Logistic Regression (training set)")
    plt.xlabel("PC1")  # for Xlabel
    plt.ylabel("PC2")  # for Ylabel
    # plt.legend()

    # show scatter plot
    plt.show()

    # def data_pca_plot(self):
    #     # idx_mi_train = [0, 1, 2, 3]
    #     d_data = np.column_stack((self.x_data, self.y_data))
    #     df_data = pd.DataFrame(d_data)
    #     df = df_data
    #     df.columns = df.columns.map(str)
    #     features = [str(idx_i) for idx_i in range(len(self.idx_features))]
    #     target = str(len(self.idx_features))
    #     fig = px.scatter_matrix(
    #         df,
    #         dimensions=features,
    #         color=df[target]
    #     )
    #     fig.update_traces(diagonal_visible=False)
    #     fig.write_html("Features.html", auto_open=True)
    #     fig.show()
    #     pca = PCA()
    #     components = pca.fit_transform(df[features])
    #     labels = {
    #         str(i): f"PC {i + 1} ({var:.1f}%)"
    #         for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    #     }
    #     fig = px.scatter_matrix(
    #         components,
    #         labels=labels,
    #         dimensions=range(len(features)),
    #         color=df[target]
    #     )
    #     fig.update_traces(diagonal_visible=False)
    #     fig.write_html("PCA.html", auto_open=True)
    #     fig.show()


def calculate_roc_auc(model, x_train, y_train, x_test, y_test):
    """
    This function uses the training samples, (x_train, y_train) to learn a classification model (model),
    and returns the (roc_auc) score of the model on the test samples (x_test, y_test), as well as the predicted labels
    (y_pred) and the corresponding prediction probabilities (y_pred).
    @return: roc_auc, y_pseudo, y_pred
    """
    model.fit(x_train, y_train)
    # y_pseudo = model_fit.transduction_
    y_pseudo = model.predict(x_train)
    y_pred = model.predict(x_test)
    y_pred_prob = model.predict_proba(x_test)
    y_pred_prob = y_pred_prob[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred, average="weighted")
    # roc_auc = balanced_accuracy_score(y_test, y_pred)
    return roc_auc, y_pseudo, y_pred  # , y_pred_prob


def get_sorted_l_u_distances(x_train_l, x_train_u, idx_train_u_rand):
    """
    This function calulates the distance between any pair of labeled and unlabeled samples
    and returns the list of unlabeled samples sorted based on their distance to the closest labeled sample
    @param x_train_l: labeled samples in the training dataset
    @param x_train_u: unlabeled samples in the training dataset
    @param idx_train_u_rand: list of indices corresponding to the unlabaled samples
    @return idx_u_sorted: list of unlabeled samples sorted based on their distance to the closest labeled sample
    """
    dist = cdist(x_train_l, x_train_u)
    dist_sort = dist.argsort(axis=None, kind="mergesort")
    dist_sort_indices = np.unravel_index(dist_sort, dist.shape)

    dist_sort_u_indices = dist_sort_indices[1]
    idx_u_sorted_list = [i for n, i in enumerate(dist_sort_u_indices) if i not in dist_sort_u_indices[:n]]

    idx_u_sorted = idx_train_u_rand[idx_u_sorted_list]
    # ------------- Uncomment the following print function to see how the unlabaled samples are sorted -----------------
    # print(idx_train_u_rand)
    # print(idx_u_sorted)
    return idx_u_sorted


def add_label_noise(y_train_l, label_noise_prcnt):
    """
    This function adds label noise to the samples
    @param y_train_l: labels corresponding to the labeled samples in the training dataset
    @param label_noise_prcnt: percentage of samples that we want to change their labels
    @return: y_train_l_noisy: vector of changed labels corresponding to the labeled samples
    """
    y_train_l_noisy = y_train_l.copy()
    n_label = len(y_train_l_noisy)
    n_chnge_label = int(n_label * label_noise_prcnt)
    idx_chnge_label = random.sample(range(0, n_label), n_chnge_label)
    for i in idx_chnge_label:
        # Get the current label
        current_label = y_train_l_noisy[i]
        # Get the set of other possible labels
        other_labels = np.unique(y_train_l_noisy)
        other_labels = other_labels[other_labels != current_label]
        # Randomly select a new label from the set of other labels
        new_label = random.choice(other_labels)
        # Assign the new label to the sample
        y_train_l_noisy[i] = new_label
    return y_train_l_noisy


def PCA_detail_plot(D, idx_mi_train):
    idx_mi_train = [0, 1, 2, 3]
    d_data = np.column_stack((D.D_data.x[:, idx_mi_train], D.D_data.y))
    df_data = pd.DataFrame(d_data)
    df = df_data
    df.columns = df.columns.map(str)
    features = [str(idx_i) for idx_i in range(len(idx_mi_train))]
    target = str(len(idx_mi_train))
    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color=df[target]
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_html("Features.html", auto_open=True)
    fig.show()

    pca = PCA()
    components = pca.fit_transform(df[features])
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(len(features)),
        color=df[target]
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_html("PCA.html", auto_open=True)
    fig.show()


def normalize_distances(distances):
    # Create a scaler
    scaler = MinMaxScaler()

    # Reshape the distances to fit the scaler"s requirements
    distances = distances.reshape(-1, 1)

    # Fit and transform the data
    normalized_distances = scaler.fit_transform(distances)

    # Reshape the distances back to the original shape
    normalized_distances = normalized_distances.reshape(distances.shape[0], distances.shape[1])

    return normalized_distances


def remove_outliers(x_data, y_data, percentage):
    # Compute pairwise distances
    distances = pairwise_distances(x_data)
    # distances = normalize_distances(distances)
    # Compute mean distances for each sample
    mean_distances = distances.mean(axis=1)
    # Compute the threshold as the (1 - percentage) quantile of mean distances
    threshold = np.percentile(mean_distances, 100 - percentage)
    # Identify outliers: samples whose mean distance to other samples is greater than the threshold
    outliers = mean_distances > threshold

    while outliers.all():
        print(f'All samples are considered outliers. Decreasing outlier_remove_prcnt from {percentage} to {percentage - 0.5}')
        percentage -= 0.5
        # Compute the threshold as the (1 - percentage) quantile of mean distances
        threshold = np.percentile(mean_distances, 100 - percentage)
        # Identify outliers: samples whose mean distance to other samples is greater than the threshold
        outliers = mean_distances > threshold


    # Remove outliers
    x_data_clean = x_data[~outliers]
    y_data_clean = y_data[~outliers]
    return x_data_clean, y_data_clean





def find_representative_samples(x_data, y_data, n_samples):

    df = pd.DataFrame(x_data)
    df["Target"] = y_data
    class_counts = df["Target"].value_counts()
    class_labels = class_counts.keys().tolist()

    # class_counts_prcnt = np.array(class_counts.tolist()) / len(y_data)
    class_counts_prcnt = np.ones(len(class_labels)) / len(class_labels)
    class_counts_new = (class_counts_prcnt * n_samples).astype(int)
    if sum(class_counts_new) != n_samples:
        diff = n_samples - sum(class_counts_new)
        class_counts_new[0] = class_counts_new[0] + diff

    representative_indices = []
    class_i = 0
    for class_label in class_labels:
        # get all samples of this class
        class_samples = x_data[y_data == class_label]

        # compute the centroid
        centroid = np.mean(class_samples, axis=0)

        # find the indices of the samples closest to the centroid
        distances_to_centroid = pairwise_distances(class_samples, [centroid])
        closest_indices = np.argsort(distances_to_centroid, axis=0)[:class_counts_new[class_i]].flatten()

        # store the indices of representative samples in the original dataset
        representative_indices.extend(np.where(y_data == class_label)[0][closest_indices])
        class_i += 1

    # get the representative samples
    x_representative = x_data[representative_indices]
    y_representative = y_data[representative_indices]

    return x_representative, y_representative











def print_size(x, x_name):
    print(f"---------------- {x_name} ----------------")
    print(f"n_rows = {x.shape[0]}")
    print(f"n_cols = {x.shape[1]}")
    print(f"----------------------------------------")


def adjust_proba(y_pseudo_prob, epsilon=1e-4):
    # Create boolean masks for 0's
    zeros = y_pseudo_prob == 0

    # Add epsilon from 0's
    y_pseudo_prob = y_pseudo_prob + epsilon * zeros

    # Renormalize each row to sum to 1
    y_pseudo_prob = y_pseudo_prob / y_pseudo_prob.sum(axis=1, keepdims=True)

    return y_pseudo_prob



def plot_auc_stats_comparison(SSEI,
                              fname_to_save=None,
                              plt_title=None,
                              ss_model_names=None,
                              data_types=None,
                              auc_statistics=None,
                              plt_auc_type=None,
                              plt_auc_calc_type=None,
                              n_col=None):

    df_summary, df_summary_parameters = SSEI.get_auc_stats_parameters()

    if fname_to_save is None:
        fname_to_save = "compare_all.jpg"
    if ss_model_names is None:
        ss_model_names = df_summary_parameters["ss_model_names"]
    #         fname_to_save = f"compare-{'_'.join(ss_model_names)}.jpg"


    if data_types is None:
        data_types = df_summary_parameters["data_types"]
    if auc_statistics is None:
        auc_statistics = df_summary_parameters["auc_statistics"]
    if plt_auc_type is None:
        plt_auc_type = df_summary_parameters["plt_auc_type"]
    if plt_auc_calc_type is None:
        plt_auc_calc_type = df_summary_parameters["plt_auc_calc_type"]

    n_mix_list = df_summary["n_mix"].unique()
    n_mix_list = n_mix_list.astype(float)
    n_mix_list = np.sort(n_mix_list)
    color_map = cm.get_cmap("tab20")
    n_curves = len(ss_model_names) * len(plt_auc_type) * len(auc_statistics)
    if n_col is None:
        if n_curves % 4 == 0:
            n_col = 4
        elif n_curves % 5 == 0:
            n_col = 5
        else:
            n_col = 5



    # n_col = 4
    n_row = int(np.ceil(n_curves / n_col))
    print(f"n_curves: {n_curves} -- n_row: {n_row}")
    n_row_max = 8

    color_list = [color_map(i) for i in range(20)]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("Number of mixed (labeled + unlabeled) samples", fontsize=16)
    # ax.set_ylabel("ROC AUC (5th percentile)", fontsize=16)
    ax.set_ylabel("ROC AUC ", fontsize=16)
    mpl.rcParams['font.size'] = 16  # Adjust the font size as needed
    legend_box_y = 0.2
    if n_row == 1:
        legend_box_y = 0.2
    else:
        legend_box_y = 0.1 * n_row


    # plt.subplots_adjust(top=0.9)
    color_cntr = 0

    for data_type, ss_model_i in product(data_types, ss_model_names):
        df_n_mix = SSEI.get_df_n_mix(df_summary, data_type, ss_model_i)
        for col1, col2, col3 in product(plt_auc_type, plt_auc_calc_type, auc_statistics):
            if col2 == "ave":
                col = f"{col1}_{col2}_{col3}"
            elif col2 == "ind":
                col = [f"{col1}_{col2}_{i}_{col3}" for i in range(1, SSEI.n_classes + 1)]
            if len(df_n_mix[col].values) != 0:
                if "lss" in ss_model_i:
                    model_name = ss_model_i.replace("lss", "ls")
                elif "svmlin" in ss_model_i:
                    model_name = ss_model_i.replace("svmlin", "svml")
                elif "svmrbf" in ss_model_i:
                    model_name = ss_model_i.replace("svmrbf", "svmr")
                else:
                    model_name = ss_model_i
                model_name = model_name.replace("_", ",").upper()
                if col1 == "pl_auc":
                    auc_type = "p_auc"
                else:
                    auc_type = col1
                if len(plt_auc_type) == 1:
                    # label = f"({model_name})"
                    label = f"auc_{col3}: ({model_name})"
                else:
                    # label = f"{auc_type}_{col3}: ({model_name})"
                    label = f"auc_{col3}: ({model_name})"
                label = f"{col3}: ({model_name})"
                style = SSEI.sim_set.get_plot_style(auc_calc_type=col2, auc_type=col1, auc_stat_type=col3)
                if 19 <= color_cntr:
                    print(color_cntr)
                    color_cntr = 0
                style["color"] = color_list[color_cntr]

                # # Apply random noise to each column based on its magnitude
                # for col, percentage in noise_percentage.items():
                #     magnitude = df[col].abs()
                #     noise = (percentage / 100) * magnitude
                #     df[col] += np.random.uniform(low=-noise, high=noise, size=len(df))
                # if "5th" in col:
                #     magnitude = df_n_mix[col].values[:-1]
                #     noise = (1 / 100) * magnitude
                #     magnitude += np.random.uniform(low=-noise, high=noise, size=len(magnitude))
                # elif "95th" in col:
                #
                # else




                plt.plot(n_mix_list[:-1], df_n_mix[col].values[:-1], **style, label=label)
            mpl.rcParams['font.size'] = 16  # Adjust the font size as needed
        color_cntr += 2

    plt.grid(True)

    # plt.xlabel("Number of mixed (labeled + unlabeled) samples")
    # plt.ylabel("ROC AUC (5th percentile)")
    # plot_title = f"Compare -{', '.join(data_types)} - {', '.join(ss_model_names)} -" \
    #              f" {', '.join(auc_statistics)} - {', '.join(plt_auc_type)} - {', '.join(plt_auc_calc_type)}"
    plot_title = plt_title
    wrapped_title = textwrap.fill(plot_title, 100)
    plt.title(wrapped_title)
    plt.ylim(bottom=0.5, top=1)
    plt.xlim(left=n_mix_list[0], right=n_mix_list[-2])

    fig_path = os.path.join("results_final", SSEI.version, fname_to_save)
    fig_dir = os.path.dirname(fig_path)
    os.makedirs(fig_dir, exist_ok=True)
    # if 1 < n_row:
    #     plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1 + n_row*0.1), ncol=n_col)
    # else:
    #     plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1 + 0.2), ncol=n_col)
    # mpl.rcParams['font.size'] = 16  # Adjust the font size as needed
    # plt.rcParams['font.size'] = 16
    plt.legend(loc='best', ncol=2)
    print("Saving plot to:", fig_path)
    plt.savefig(fig_path)
    plt.close(fig)
    # plt.show()



def plot_aucs_stats(SSEI, n_mix_list, df_n_mix):
    # fig = plt.figure(figsize=(18, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(top=0.9)
    columns = ["t_auc_ave", "pl_auc_ave"] + [f"pl_auc_ind_{i}" for i in range(1, SSEI.n_classes + 1)] + [
        f"t_auc_ind_{i}" for i in range(1, SSEI.n_classes + 1)]

    df_aucs_ave = df_n_mix.groupby("n_mix").agg({col: "mean" for col in columns})
    df_aucs_min = df_n_mix.groupby("n_mix").agg({col: "min" for col in columns})
    df_aucs_max = df_n_mix.groupby("n_mix").agg({col: "max" for col in columns})
    df_aucs_5th = df_n_mix.groupby("n_mix").agg({col: lambda x: np.percentile(x, 5) for col in columns})
    df_aucs_95th = df_n_mix.groupby("n_mix").agg({col: lambda x: np.percentile(x, 95) for col in columns})
    # ==============================================================================================================
    #                           Plotting Psuedo-label AUCs
    # ==============================================================================================================
    if ("ave" in SSEI.sim_set.plt_auc_calc_type) and ("pl_auc" in SSEI.sim_set.plt_auc_type):

        if ("ave" in SSEI.sim_set.plt_auc_stat_type):
            style_m = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="ave")
            plt.plot(n_mix_list, df_aucs_ave["pl_auc_ave"].values, **style_m, label="p_auc_ave")

        if ("min" and "max" in SSEI.sim_set.plt_auc_stat_type):
            if "lines" in SSEI.sim_set.plt_auc_stat_style:
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="min")
                plt.plot(n_mix_list, df_aucs_min["pl_auc_ave"].values, **style, label="p_auc_min")
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="max")
                plt.plot(n_mix_list, df_aucs_max["pl_auc_ave"].values, **style, label="p_auc_max")
            if "fill" in SSEI.sim_set.plt_auc_stat_style:
                plt.fill_between(n_mix_list, df_aucs_min["pl_auc_ave"].values, df_aucs_max["pl_auc_ave"].values,
                                 alpha=0.2, color=style_m["color"], hatch=".")

        if ("5th" and "95th" in SSEI.sim_set.plt_auc_stat_type):
            if "lines" in SSEI.sim_set.plt_auc_stat_style:
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="5th")
                plt.plot(n_mix_list, df_aucs_5th["pl_auc_ave"].values, **style, label="p_auc_5th")
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="pl_auc", auc_stat_type="95th")
                plt.plot(n_mix_list, df_aucs_95th["pl_auc_ave"].values, **style, label="p_auc_95th")
            if "fill" in SSEI.sim_set.plt_auc_stat_style:
                plt.fill_between(n_mix_list, df_aucs_5th["pl_auc_ave"].values, df_aucs_95th["pl_auc_ave"].values,
                                 alpha=0.2, color=style_m["color"], hatch="x")

    if ("ind" in SSEI.sim_set.plt_auc_calc_type) and ("pl_auc" in SSEI.sim_set.plt_auc_type):

        if ("ave" in SSEI.sim_set.plt_auc_stat_type):
            for i in range(1, SSEI.n_classes + 1):
                # Plot average, min and max for "pl_auc_ind_i"
                style_m = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="ave",
                                                      class_num=i)
                plt.plot(n_mix_list, df_aucs_ave[f"pl_auc_ind_{i}"].values, **style_m, label=f"pl_auc_ind_{i} ")

                if ("min" and "max" in SSEI.sim_set.plt_auc_stat_type):
                    if "lines" in SSEI.sim_set.plt_auc_stat_style:
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="min",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_min[f"pl_auc_ind_{i}"].values, **style,
                                 label=f"pl_auc_ind_{i}_min")
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="max",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_max[f"pl_auc_ind_{i}"].values, **style,
                                 label=f"pl_auc_ind_{i}_max")
                    if "fill" in SSEI.sim_set.plt_auc_stat_style:
                        plt.fill_between(n_mix_list, df_aucs_min[f"pl_auc_ind_{i}"].values,
                                         df_aucs_max[f"pl_auc_ind_{i}"].values,
                                         alpha=0.2, color=style_m["color"], hatch=".")

                if ("5th" and "95th" in SSEI.sim_set.plt_auc_stat_type):
                    if "lines" in SSEI.sim_set.plt_auc_stat_style:
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc", auc_stat_type="5th",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_5th[f"pl_auc_ind_{i}"].values, **style,
                                 label=f"pl_auc_ind_{i}_5th")
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="pl_auc",
                                                            auc_stat_type="95th",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_95th[f"pl_auc_ind_{i}"].values, **style,
                                 label=f"pl_auc_ind_{i}_95th")
                    if "fill" in SSEI.sim_set.plt_auc_stat_style:
                        plt.fill_between(n_mix_list, df_aucs_5th[f"pl_auc_ind_{i}"].values,
                                         df_aucs_95th[f"pl_auc_ind_{i}"].values,
                                         alpha=0.2, color=style_m["color"], hatch="x")

    # ==============================================================================================================
    #                           Plotting test AUCs
    # ==============================================================================================================
    if ("ave" in SSEI.sim_set.plt_auc_calc_type) and ("t_auc" in SSEI.sim_set.plt_auc_type):

        if ("ave" in SSEI.sim_set.plt_auc_stat_type):
            style_m = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="ave")
            plt.plot(n_mix_list, df_aucs_ave["t_auc_ave"].values, **style_m, label="t_auc_ave")

        if ("min" and "max" in SSEI.sim_set.plt_auc_stat_type):
            if "lines" in SSEI.sim_set.plt_auc_stat_style:
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="min")
                plt.plot(n_mix_list, df_aucs_min["t_auc_ave"].values, **style, label="t_auc_min")
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="max")
                plt.plot(n_mix_list, df_aucs_max["t_auc_ave"].values, **style, label="t_auc_max")
            if "fill" in SSEI.sim_set.plt_auc_stat_style:
                plt.fill_between(n_mix_list, df_aucs_min["t_auc_ave"].values, df_aucs_max["t_auc_ave"].values,
                                 alpha=0.2, color=style_m["color"], hatch="//")

        if ("5th" and "95th" in SSEI.sim_set.plt_auc_stat_type):
            if "lines" in SSEI.sim_set.plt_auc_stat_style:
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="5th")
                plt.plot(n_mix_list, df_aucs_5th["t_auc_ave"].values, **style, label="t_auc_5th")
                style = SSEI.sim_set.get_plot_style(auc_calc_type="ave", auc_type="t_auc", auc_stat_type="95th")
                plt.plot(n_mix_list, df_aucs_95th["t_auc_ave"].values, **style, label="t_auc_95th")
            if "fill" in SSEI.sim_set.plt_auc_stat_style:
                plt.fill_between(n_mix_list, df_aucs_5th["t_auc_ave"].values, df_aucs_95th["t_auc_ave"].values,
                                 alpha=0.2, color=style_m["color"], hatch="|")

    if ("ind" in SSEI.sim_set.plt_auc_calc_type) and ("t_auc" in SSEI.sim_set.plt_auc_type):
        if ("ave" in SSEI.sim_set.plt_auc_stat_type):

            for i in range(1, SSEI.n_classes + 1):
                style_m = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc", auc_stat_type="ave",
                                                      class_num=i)
                plt.plot(n_mix_list, df_aucs_ave[f"t_auc_ind_{i}"].values, **style_m, label=f"t_auc_ind_{i}")

                if ("min" and "max" in SSEI.sim_set.plt_auc_stat_type):
                    if "lines" in SSEI.sim_set.plt_auc_stat_style:
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                            auc_stat_type="min",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_min[f"t_auc_ind_{i}"].values, **style,
                                 label=f"t_auc_ind_{i}_min")
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                            auc_stat_type="max",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_max[f"t_auc_ind_{i}"].values, **style,
                                 label=f"t_auc_ind_{i}_max")
                    if "fill" in SSEI.sim_set.plt_auc_stat_style:
                        plt.fill_between(n_mix_list, df_aucs_min[f"t_auc_ind_{i}"].values,
                                         df_aucs_max[f"t_auc_ind_{i}"].values,
                                         alpha=0.2, color=style_m["color"], hatch="//")

                if ("5th" and "95th" in SSEI.sim_set.plt_auc_stat_type):
                    if "lines" in SSEI.sim_set.plt_auc_stat_style:
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                            auc_stat_type="5th",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_5th[f"t_auc_ind_{i}"].values, **style,
                                 label=f"t_auc_ind_{i}_5th")
                        style = SSEI.sim_set.get_plot_style(auc_calc_type="ind", auc_type="t_auc",
                                                            auc_stat_type="95th",
                                                            class_num=i)
                        plt.plot(n_mix_list, df_aucs_95th[f"t_auc_ind_{i}"].values, **style,
                                 label=f"t_auc_ind_{i}_95th")
                    if "fill" in SSEI.sim_set.plt_auc_stat_style:
                        plt.fill_between(n_mix_list, df_aucs_5th[f"t_auc_ind_{i}"].values,
                                         df_aucs_95th[f"t_auc_ind_{i}"].values,
                                         alpha=0.2, color=style_m["color"], hatch="|")

    # Add labels and legends
    plt.grid(True)
    plt.ylim(SSEI.sim_set.plt_y_lim[0], SSEI.sim_set.plt_y_lim[1])
    plt.xlabel("n_mix -- " + "n_L = " + str(SSEI.n_labeled) + " -- n_U = " + str(
        SSEI.n_unlabeled) + " -- step = " + str(SSEI.step_to_unlabeled_as_labeled))
    plt.ylabel("ROC_AUC")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=6)
    # plt.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, df_aucs_ave, df_aucs_min, df_aucs_max, df_aucs_5th, df_aucs_95th, columns

def plot_auc_all(SSEI, df_n_mix, fold_num, set_num):
    # Filter the dataframe based on fold_num and set_num
    filtered_df = df_n_mix[(df_n_mix["fold_num"] == fold_num) & (df_n_mix["set_num"] == set_num)]

    # Iterate over the unique n_mix values
    for n_mix in filtered_df["n_mix"].unique():
        # Filter the data for the current n_mix
        n_mix_data = filtered_df[filtered_df["n_mix"] == n_mix]

        # Check if n_rand is not equal to 1 for the current n_mix
        if n_mix_data["n_rand"].nunique() > 1:
            # Compute the average auc values for the specific n_mix
            pl_auc_ave = n_mix_data.groupby("n_mix")["pl_auc_ave"].mean().values[0]
            t_auc_ave = n_mix_data.groupby("n_mix")["t_auc_ave"].mean().values[0]

            # Compute the average auc values for pl_auc_ind_i and t_auc_ind_i
            pl_auc_ind_avg = []
            t_auc_ind_avg = []
            for i in range(1, SSEI.n_classes + 1):
                pl_auc_ind_avg.append(n_mix_data.groupby("n_mix")["pl_auc_ind_" + str(i)].mean().values[0])
                t_auc_ind_avg.append(n_mix_data.groupby("n_mix")["t_auc_ind_" + str(i)].mean().values[0])
        else:
            # Use the auc values for the specific n_rand_i
            pl_auc_ave = n_mix_data["pl_auc_ave"].values[0]
            t_auc_ave = n_mix_data["t_auc_ave"].values[0]

            # Use the auc values for pl_auc_ind_i and t_auc_ind_i
            pl_auc_ind_avg = [n_mix_data["pl_auc_ind_" + str(i)].values[0] for i in range(1, SSEI.n_classes + 1)]
            t_auc_ind_avg = [n_mix_data["t_auc_ind_" + str(i)].values[0] for i in range(1, SSEI.n_classes + 1)]

        # Plot n_mix vs pl_auc_ave
        plt.plot(n_mix, pl_auc_ave, "ro--", label="p_auc_ave")

        # Plot n_mix vs t_auc_ave
        plt.plot(n_mix, t_auc_ave, "b*--", label="t_auc_ave")

        # Plot n_mix vs pl_auc_ind_i
        for i, pl_auc_ind in enumerate(pl_auc_ind_avg):
            plt.plot(n_mix, pl_auc_ind, "go--", label=f"pl_auc_ind_{i + 1}")

        # Plot n_mix vs t_auc_ind_i
        for i, t_auc_ind in enumerate(t_auc_ind_avg):
            plt.plot(n_mix, t_auc_ind, "k*--", label=f"t_auc_ind_{i + 1}")



def plot_auc_stat_summary(SSEI):

        df = SSEI.semisup_analysis_df
        ss_model_names = df["ss_model"].unique()
        data_types = df["data_type"].unique()
        n_data_types = len(data_types)
        n_folds = len(df["fold_num"].unique())
        n_mix_list = df["n_mix"].unique()
        n_mix_list = n_mix_list.astype(float)  # convert to float
        n_mix_list = np.sort(n_mix_list)  # sort the list
        n_sets = len(df["set_num"].unique())
        SSEI.sim_set.plt_auc_calc_type = ["ave", "ind"]

        df_aucs_ave_list = []  # List to store df_aucs_ave at each iteration
        df_aucs_min_list = []  # List to store df_aucs_min at each iteration
        df_aucs_max_list = []  # List to store df_aucs_max at each iteration
        df_aucs_5th_list = []  # List to store df_aucs_5th at each iteration
        df_aucs_95th_list = []  # List to store df_aucs_95th at each iteration

        # for ss_model_i, data_type in product(ss_model_names, data_types):

        for ss_model_i, data_type in product(ss_model_names, data_types):

            SSEI.sim_set.plt_auc_calc_type = ["ave", "ind"]
            df_n_mix = SSEI.get_df_n_mix(df, data_type, ss_model_i)
            plot, df_aucs_ave, df_aucs_min, df_aucs_max, df_aucs_5th, df_aucs_95th, columns \
                = SSEI.plot_aucs_stats(n_mix_list, df_n_mix)

            df_aucs_ave_list.append(df_aucs_ave)  # Append df_aucs_ave to the list
            df_aucs_min_list.append(df_aucs_min)  # Append df_aucs_min to the list
            df_aucs_max_list.append(df_aucs_max)  # Append df_aucs_max to the list
            df_aucs_5th_list.append(df_aucs_5th)  # Append df_aucs_5th to the list
            df_aucs_95th_list.append(df_aucs_95th)  # Append df_aucs_95th to the list

            # Add title to the plot
            fname_to_save = "AUC-" + ss_model_i + "-" + data_type + "-detail.jpg"
            print("Saving plot to:", fname_to_save)
            plt_title = f"Model={ss_model_i}-- n_folds={n_folds}-n_steps={len(n_mix_list)}--n_sets={n_sets}--n_rand_max={SSEI.n_rand_max}"
            fig_path = os.path.join("results_final", SSEI.version, fname_to_save)
            plot.suptitle(plt_title)
            plt.savefig(fig_path)
            plt.show()
            plt.close()

            SSEI.sim_set.plt_auc_calc_type = ["ave"]
            df_n_mix = SSEI.get_df_n_mix(df, data_type, ss_model_i)
            plot, df_aucs_ave, df_aucs_min, df_aucs_max, df_aucs_5th, df_aucs_95th, columns \
                = SSEI.plot_aucs_stats(n_mix_list, df_n_mix)
            # Add title to the plot
            fname_to_save = "AUC-" + ss_model_i + "-" + data_type + "-ave.jpg"
            print("Saving plot to:", fname_to_save)
            # plt_title = f"Model={ss_model_i}-- n_folds={n_folds}-n_steps={len(n_mix_list)}--n_sets={n_sets}--n_rand_max={SSEI.n_rand_max}"

            if "svmlin" in ss_model_i:
                model_name = ss_model_i.replace("svmlin", "svml")
            elif "svmrbf" in ss_model_i:
                model_name = ss_model_i.replace("svmrbf", "svmr")
            else:
                model_name = ss_model_i
            model_name = model_name.replace("_", ",").upper()
            plt_title = f"Model: ($F_p$, $F_t$) = ({model_name})"

            fig_path = os.path.join("results_final", SSEI.version, fname_to_save)
            plot.suptitle(plt_title)
            plt.savefig(fig_path)
            plt.rcParams['font.size'] = 18
            plt.show()
            plt.close()

            if SSEI.sim_set.plt_add_fold_details:
                for fold_num in range(n_folds):
                    df_n_mix_fold = df_n_mix[df_n_mix["fold_num"] == fold_num]
                    SSEI.sim_set.plt_auc_calc_type = ["ave"]
                    plot_f, df_aucs_ave, df_aucs_min, df_aucs_max, df_aucs_5th, df_aucs_95th, columns \
                        = SSEI.plot_aucs_stats(n_mix_list, df_n_mix_fold)

                    # Add title to the plot
                    fname_to_save = "AUC-" + ss_model_i + "-" + data_type + "-ave- fold-" + str(fold_num) + ".jpg"
                    print("Saving plot to:", fname_to_save)

                    plt_title = f"Model={ss_model_i}-- fold_num={fold_num}/{n_folds}-n_steps={len(n_mix_list)}--n_sets={n_sets}--n_rand_max={SSEI.n_rand_max}"
                    # plt_title = "Model= " + ss_model_i + "- n_steps " + str(len(n_mix_list))
                    # plot.suptitle(f"AUC Statistics - {ss_model_i} - {data_type}")
                    # # Save the figure
                    # plot.savefig(f"auc_statistics_{ss_model_i}_{data_type}.png")
                    fig_path = os.path.join("results_final", SSEI.version, fname_to_save)
                    plot_f.suptitle(plt_title)
                    plt.savefig(fig_path)
                    plt.show()
                    plt.close()



def calculate_combinations(n_samples, n_labeled, n_unlabeled, n_test):
    # Check if the input values are valid
    if n_labeled + n_unlabeled + n_test != n_samples:
        print("Error: n_labeled + n_unlabeled + n_test should be equal to n_samples")
        return

    # Calculate the combinations
    combination_1 = factorial(n_samples) // (factorial(n_labeled) * factorial(n_samples - n_labeled))
    combination_2 = factorial(n_samples - n_labeled) // (factorial(n_unlabeled) * factorial(n_samples - n_labeled - n_unlabeled))
    combination_3 = factorial(n_samples - n_labeled - n_unlabeled) // (factorial(n_test) * factorial(n_samples - n_labeled - n_unlabeled - n_test))

    # Calculate the total combinations
    total_combinations = combination_1 * combination_2 * combination_3

    # Print the results
    print("Number of combinations for labeled samples:", combination_1)
    print("Number of combinations for unlabeled samples:", combination_2)
    print("Number of combinations for test samples:", combination_3)
    print("Total number of combinations:", total_combinations)




def get_tsne_centroids(x, y, random_state=0):
    tsne = TSNE(n_components=2, random_state=random_state)
    x_tsne = tsne.fit_transform(x)
    # Compute cluster centroids based on true labels
    kmeans_true = KMeans(n_clusters=np.unique(y).shape[0], random_state=random_state)
    kmeans_true.fit(x_tsne)

    cluster_centroids = kmeans_true.cluster_centers_
    return cluster_centroids


def calculate_centroids(n_mix, x_train_l, y_train_l, x_train_u, y_train_u_true, y_train_u_pseudo, x_test, y_test_true,
                  y_test_pred, x_data, y_data, random_state=0):


    x_train_mix = np.vstack((x_train_l, x_train_u))
    y_train_mix_true = np.concatenate((y_train_l, y_train_u_true), axis=0).astype(int)
    y_train_mix_pseudo = np.concatenate((y_train_l, y_train_u_pseudo), axis=0).astype(int)


    centroids_l = get_tsne_centroids(x_train_l, y_train_l, random_state=random_state)
    centroids_mix_pseudo = get_tsne_centroids(x_train_mix, y_train_mix_pseudo, random_state=random_state)
    centroids_mix_true = get_tsne_centroids(x_train_mix, y_train_mix_true, random_state=random_state)
    centroids_test_true = get_tsne_centroids(x_test, y_test_true, random_state=random_state)
    centroids_test_pred = get_tsne_centroids(x_test, y_test_pred, random_state=random_state)
    centroids_data = get_tsne_centroids(x_data, y_data, random_state=random_state)

    return centroids_l, centroids_mix_pseudo, centroids_mix_true, centroids_test_true, centroids_test_pred, centroids_data

