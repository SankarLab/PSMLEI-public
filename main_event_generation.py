from config_event_generation import *
from core.utils import *

create_environment(psse_path)
import psse35
import psspy
psspy.psseinit(5000)
import pssplot
import redirect
import dyntools


sim_setting = simulation_setting(system_files_path=system_files_path,
                                 project_path=project_path,
                                 raw_file_name=raw_file_name,
                                 dyr_file_name=dyr_file_name,
                                 psse_files_vers=psse_files_vers,
                                 pmu_bus_select_method=pmu_bus_select_method,
                                 pmu_bus_num=pmu_bus_num,
                                 psse_set_minor=psse_set_minor,
                                 psseinit=psseinit,
                                 flat_run_time=flat_run_time,
                                 remove_fault_time=remove_fault_time,
                                 simulation_time=simulation_time,
                                 generate_loading_scenarios=generate_loading_scenarios,
                                 start_load_level=start_load_level,
                                 end_load_level=end_load_level,
                                 n_scenarios=n_scenarios,
                                 min_rand=min_rand,
                                 max_rand=max_rand,
                                 event_types_list=event_types_list)


psse35.set_minor(sim_setting.psse_set_minor)
psspy.psseinit(sim_setting.psseinit)

sc = SystemComponents(sim_setting.raw_path, psspy, sim_setting.psse_vers)
if plot_network_flag==True:
    plot_network(sc)

load_change, load_chng_prcnt = get_load_change_prcnt(sc,
                                                     sim_setting)


"""
Select a subset of buses with PMUs
"""
start_range = 1
end_range = len(sc.busnumbers)


pmu_bus_num = get_pmu_bus_nums(sim_setting, start_range, end_range, n_pmus)



"""
Load Loss scenarios (event_label = 1)
"""
event_num = 1
event_label = []
if sim_setting.generate_load_loss:
    for load_i in event_load_list:  # range(len(Mach_No)):#range(len(Mach_No))
        for scenario_number in range(1, sim_setting.n_scenarios + 1):
            print("Load count:" + str(load_i) + 'loss of load on bus:' + str(
                sc.load_bus_num_sorted[load_i]) + '--Scen_Num: ' + str(scenario_number))
            import psse35

            psse35.set_minor(sim_setting.psse_set_minor)
            redirect.psse2py()
            psspy.psseinit(sim_setting.psseinit)
            psspy.set_NaN_python()
            psspy.readrawversion(0, sim_setting.psse_vers, sim_setting.raw_path)
            if sim_setting.generate_loading_scenarios:
                apply_load_change(psspy, sc, load_change[scenario_number, :], load_chng_prcnt[scenario_number, :])
            psse_powerflow_fact_tysl(psspy)
            psspy.dyre_new([1, 1, 1, 1], sim_setting.dyr_path, "", "", "")
            psse_channel_setup(psspy, pmu_bus_num)
            apply_load_loss(psspy, sim_setting.loadloss_res_path, sc.load_bus_num_sorted[load_i],
                            sc.load_id_sorted[load_i], sim_setting.flat_run_time, sim_setting.simulation_time,
                            scenario_number)
            print("Load count:" + str(load_i) + 'loss of load on bus:' + str(
                sc.load_bus_num_sorted[load_i]) + '--Scen_Num: ' + str(scenario_number))
            save_load_loss(dyntools, pssplot, sim_setting.loadloss_res_path, sc.load_bus_num_sorted[load_i],
                           scenario_number)
            temp_data_array = save_load_loss_as_mat(sim_setting.loadloss_res_path, n_pmus, event_num,
                                                    sc.load_bus_num_sorted[load_i], scenario_number)
            psspy.delete_all_plot_channels()
            event_num += 1
            event_label.append(1)

"""
Generation Loss scenarios (event_label = 2)
"""
if sim_setting.generate_generation_loss:
    for m_i in event_gen_list:  # len(sc.gen_bus_num_sorted)):#range(len(Mach_No))
        for scenario_number in range(1, sim_setting.n_scenarios + 1):
            print("Gen count:" + str(m_i) + 'loss of generator on bus:' + str(
                sc.gen_bus_num_sorted[m_i]) + '--Scen_Num: ' + str(scenario_number))
            import psse35
            redirect.psse2py()
            psspy.psseinit(sim_setting.psseinit)
            psspy.readrawversion(0, sim_setting.psse_vers, sim_setting.raw_path)
            if sim_setting.generate_loading_scenarios:
                apply_load_change(psspy, sc, load_change[scenario_number, :], load_chng_prcnt[scenario_number, :])
            psse_powerflow_fact_tysl(psspy)
            psspy.dyre_new([1, 1, 1, 1], sim_setting.dyr_path, "", "", "")
            psse_channel_setup(psspy, pmu_bus_num)
            # psspy.set_relang(1, 9, r"""1""")
            apply_generation_loss(psspy, sim_setting.gen_res_path, sc.gen_bus_num_sorted[m_i],
                                  sim_setting.flat_run_time, sim_setting.simulation_time, scenario_number)
            print("Gen count:" + str(m_i) + 'loss of generator on bus:' + str(
                sc.gen_bus_num_sorted[m_i]) + '--Scen_Num: ' + str(scenario_number))
            save_generation_loss(dyntools, pssplot, sim_setting.gen_res_path, sc.gen_bus_num_sorted[m_i],
                                 scenario_number)
            save_generation_loss_as_mat(sim_setting.gen_res_path, n_pmus, event_num, sc.gen_bus_num_sorted[m_i],
                                        scenario_number)
            psspy.delete_all_plot_channels()
            event_num += 1
            event_label.append(2)

"""
Line Trip scenarios (event_label = 3)
"""
if sim_setting.generate_line_trip:
    for l_i in event_line_list:
        for scenario_number in range(1, sim_setting.n_scenarios + 1):
            print("Line count:" + str(l_i) + 'Line trip on:' + str(sc.line_from_bus_sorted[l_i]) + '--' + str(
                sc.line_to_bus_sorted[l_i]) + '--Scen_Num: ' + str(scenario_number))
            import psse35
            psse35.set_minor(sim_setting.psse_set_minor)
            redirect.psse2py()
            psspy.psseinit(sim_setting.psseinit)
            psspy.readrawversion(0, sim_setting.psse_vers, sim_setting.raw_path)
            if sim_setting.generate_loading_scenarios:
                apply_load_change(psspy, sc, load_change[scenario_number, :], load_chng_prcnt[scenario_number, :])
            psse_powerflow_fact_tysl(psspy)
            psspy.dyre_new([1, 1, 1, 1], sim_setting.dyr_path, "", "", "")
            # psspy.set_relang(1, 9, r"""1""")
            psse_channel_setup(psspy, pmu_bus_num)
            apply_line_trip(psspy, sim_setting.linetrip_res_path, sc.line_from_bus_sorted[l_i],
                            sc.line_to_bus_sorted[l_i], sim_setting.flat_run_time, sim_setting.simulation_time,
                            scenario_number)
            print("Line count:" + str(l_i) + 'Line trip on:' + str(sc.line_from_bus_sorted[l_i]) + '--' + str(
                sc.line_to_bus_sorted[l_i]) + '--Scen_Num: ' + str(scenario_number))
            save_line_trip(dyntools, pssplot, sim_setting.linetrip_res_path, sc.line_from_bus_sorted[l_i],
                           sc.line_to_bus_sorted[l_i], scenario_number)
            save_line_trip_as_mat(sim_setting.linetrip_res_path, n_pmus, event_num, sc.line_from_bus_sorted[l_i],
                                  sc.line_to_bus_sorted[l_i], scenario_number)
            psspy.delete_all_plot_channels()
            event_num += 1
            event_label.append(3)
"""
Line Fault scenarios (event_label = 4)
"""
if sim_setting.generate_line_fault:
    for l_i in event_line_list:  # 700):#number_of_linefault_scenarios):
        for scenario_number in range(1, sim_setting.n_scenarios + 1):
            print("Line count:" + str(l_i) + 'Line Fault on:' + str(sc.line_from_bus_sorted[l_i]) + '--' + str(
                sc.line_to_bus_sorted[l_i]) + '--Scen_Num: ' + str(scenario_number))
            import psse35
            psse35.set_minor(sim_setting.psse_set_minor)
            redirect.psse2py()
            psspy.psseinit(sim_setting.psseinit)
            psspy.readrawversion(0, sim_setting.psse_vers, sim_setting.raw_path)
            if sim_setting.generate_loading_scenarios:
                apply_load_change(psspy, sc, load_change[scenario_number, :], load_chng_prcnt[scenario_number, :])
            psse_powerflow_fact_tysl(psspy)
            psspy.dyre_new([1, 1, 1, 1], sim_setting.dyr_path, "", "", "")
            # psspy.set_relang(1, 9, r"""1""")
            psse_channel_setup(psspy, pmu_bus_num)
            apply_line_fault(psspy, sim_setting.line_res_path, sc.line_from_bus_sorted[l_i],
                             sc.line_to_bus_sorted[l_i], sim_setting.flat_run_time, sim_setting.remove_fault_time,
                             sim_setting.simulation_time,
                             scenario_number)
            print("Line count:" + str(l_i) + 'Line Fault on:' + str(sc.line_from_bus_sorted[l_i]) + '--' + str(
                sc.line_to_bus_sorted[l_i]) + '--Scen_Num: ' + str(scenario_number))
            save_line_fault(dyntools, pssplot, sim_setting.line_res_path, sc.line_from_bus_sorted[l_i],
                            sc.line_to_bus_sorted[l_i], scenario_number)
            save_line_fault_as_mat(sim_setting.line_res_path, n_pmus, event_num, sc.line_from_bus_sorted[l_i],
                                   sc.line_to_bus_sorted[l_i], scenario_number)
            psspy.delete_all_plot_channels()
            event_num += 1
            event_label.append(4)

"""
Bus Fault Scenarios (event_label = 5)
"""
if sim_setting.generate_bus_fault:
    for scenario_number in range(1, n_scenarios + 1):
        for b_i in event_bus_list:  # len(sc.busnumbers)):
            print("Bus count:" + str(b_i) + 'Bus Fault on:' + str(sc.busnumbers[b_i]) + '--Scen_Num: ' + str(
                scenario_number))
            import psse35
            psse35.set_minor(sim_setting.psse_set_minor)
            redirect.psse2py()
            psspy.psseinit(sim_setting.psseinit)
            psspy.readrawversion(0, sim_setting.psse_vers, sim_setting.raw_path)
            if sim_setting.generate_loading_scenarios:
                apply_load_change(psspy, sc, load_change[scenario_number, :], load_chng_prcnt[scenario_number, :])
            psse_powerflow_fact_tysl(psspy)
            psspy.dyre_new([1, 1, 1, 1], sim_setting.dyr_path, "", "", "")
            # psspy.set_relang(1, 9, r"""1""")
            psse_channel_setup(psspy, pmu_bus_num)
            apply_bus_fault(psspy, sim_setting.bus_res_path, sc.busnumbers[b_i], sim_setting.flat_run_time,
                            sim_setting.remove_fault_time, sim_setting.simulation_time,
                            scenario_number)
            print("Bus count:" + str(b_i) + 'Bus Fault on:' + str(sc.busnumbers[b_i]) + '--Scen_Num: ' + str(
                scenario_number))
            save_bus_fault(dyntools, pssplot, sim_setting.bus_res_path, sc.busnumbers[b_i], scenario_number)
            save_bus_fault_as_mat(sim_setting.bus_res_path, n_pmus, event_num, sc.busnumbers[b_i], scenario_number)
            psspy.delete_all_plot_channels()
            event_num += 1
            event_label.append(5)

"""
save event labels as mat file:
Event labels will be saved as follows:
Load loss: 1
Generation loss: 2
Line trip: 3
Line fault: 4
Bus fault: 5
"""
save_event_labels_as_mat(event_label, project_path)




