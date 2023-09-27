# ======================================================================================================================
#                                       Inputs for the event generation module
# ======================================================================================================================
# specify minor version of the PSSE software, e.g., X.4
psse_set_minor = 4
# To initialize the PSSE environment with specific parameters.
psseinit = 5000
# PSSE directory
psse_path = r"C:\Program Files\PTI\PSSE35\35.4\PSSPY39"
# The project path
project_path = r"""C:\Users\ntaghip1\PycharmProjects\Event_Generation_private\PSMLEI"""
# Specifies the directory of the synthetic network .raw and .dyr files
system_files_path = project_path + r"""\inputs\system_data"""
raw_file_name = "ACTIVSg500.raw"
dyr_file_name = "ACTIVSg500_dynamics.dyr"
# Specify the PSSE version of the .raw and .dyr files
psse_files_vers = r"""35"""

# If true, plots the graph representation of the network
plot_network_flag = False
"""
------------------------------------------------------------------------------------------------------------------------
Specify a list of event types to be considered for synthetic eventful pmu data generation
Currently, the following event types are supported: generation loss, load loss, line trip, line fault, bus fault
*** Note: use the following convention to specify the types of events ***
"generation_loss", "load_loss", "line_trip", "line_fault", "bus_fault"
------------------------------------------------------------------------------------------------------------------------
Ex: 
event_types_list = ["load_loss", "bus_fault"]
"""
event_types_list = ["generation_loss", "load_loss", "line_trip", "line_fault", "bus_fault"]


"""
------------------------------------------------------------------------------------------------------------------------
We arrange the system components in order of their magnitude. For each specific event type, please indicate the range
(or list) of components (e.g., generators, lines, loads, buses) on which you intend to apply a disturbance.
Note that in the range(from, to) specification below, the min/max values of 'from' and 'to' arguments correspond to the
minimum and maximum magnitudes of the components (loads, generators, lines, and buses) within the network.
------------------------------------------------------------------------------------------------------------------------
Example:
event_load_list = range(from, to)
event_gen_list = range(from, to)
event_line_list = range(from, to)
event_bus_list = range(from, to)
------------------------------------------------------------------------------------------------------------------------
"""
event_load_list = range(0, 1)
event_gen_list = range(0, 1)
event_line_list = range(0, 1)
event_bus_list = range(0, 1)


"""
------------------------------------------------------------------------------------------------------------------------
Specify Subset of Buses for PMU Placement.
Three methods are available for selecting PMU buses:
- "random_in_range" generates a list of bus numbers with PMUs that spans from bus #1 to bus #n_bus with n_pmus
  installed on every n_bus/n_pmus buses in the system.
- "random_in_buses" randomly selects n_pmus buses out of n_bus buses in the system.
- "specify_buses" allows user-defined lists of buses with PMUs. If this method is used, a pmu_bus_num list is required.
------------------------------------------------------------------------------------------------------------------------
Example1:
pmu_bus_select_method = "random_in_range"
pmu_bus_num = None 
n_pmus = 95

Example2:
pmu_bus_select_method = "random_in_buses"
pmu_bus_num = None 
n_pmus = 95

Example3: 
pmu_bus_select_method = "specify_buses"
pmu_bus_num = [bus_num_1, bus_num_2, ...]
n_pmus = len(pmu_bus_num)
------------------------------------------------------------------------------------------------------------------------
"""
pmu_bus_select_method = "random_in_range"
pmu_bus_num = None  # list of bus numbers with pmus - required if "specify_buses" method is selected
n_pmus = 95



"""
------------------------------------------------------------------------------------------------------------------------
PSSE dynamic simulation setting
flat_run_time: Time (in seconds) for flat run dynamic simulation after performing power flow and before applying the disturbance
remove_fault_time: For bus fault and line fault events, disturbance is cleared at 'remove_fault_time' seconds.
simulation_time: Total dynamic simulation time (in seconds)
------------------------------------------------------------------------------------------------------------------------
"""
flat_run_time = 1
remove_fault_time = 1.05
simulation_time = 20

"""
Generate various system loading condition scenarios:
------------------------------------------------------------------------------------------------------------------------
For each individual load_i, the consumption S_i (MVA) is defined as S_i = P_i (MW) + j Q_i (MVAR).
The total system loading S_tot is the sum of all individual loads.

generate_loading_scenarios: If set to True, various loading condition scenarios will be generated.
If set to False, the system will operate under normal conditions without disturbances.

User-defined parameters for generating various loading conditions:
---------------------------------------------------------------
- start_load_level: The lower bound in system loading condition scenarios, represented as (min S_tot)/(nominal S_tot).

- end_load_level: The upper bound in system loading condition scenarios, represented as (max S_tot)/(nominal S_tot).

- n_scenarios: The number of different loading condition scenarios to generate.
  For each loading condition scenario j = 1, ..., n_scenarios, the load consumption for each load_i is calculated as:
  S_i[j] = (nominal S_i) + (nominal S_i) * [(end_load_level - start_load_level) / n_scenario] * j

- min_rand and max_rand: Given these bounds, random load change ratios are generated within this range to change
  each individual load_i at loading scenario j, denoted as S_i[j]:
  S_i[j] = S_i[j] + rand(min_rand, max_rand) * S_i[j]
"""
generate_loading_scenarios = True

start_load_level = 0.99  # Lower bound in system loading condition scenarios (min S_tot)/(nominal S_tot)
end_load_level = 1.01  # Upper bound in system loading condition scenarios (max S_tot)/(nominal S_tot)
n_scenarios = 1  # Number of different loading condition scenarios to generate
min_rand, max_rand = -0.02, 0.02  # Range for random load change ratios



