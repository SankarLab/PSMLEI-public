from config_feature_extraction import *
from core.utils import *

FS = FeatureExtraction(project_path=project_path,
                       generated_events_path=generated_events_path,
                       first_event=first_event,
                       last_event=last_event,
                       first_sample=first_sample,
                       last_sample=last_sample,
                       n_pmus=n_pmus,
                       n_pmus_prime=n_pmus_prime,
                       Rank=Rank,
                       Ts=Ts,
                       p=p,
                       p_prime=p_prime,
                       decimal_tol=decimal_tol)

raw_vm, raw_va, raw_f = FS.import_raw_data()
detrend_raw_vm, detrend_raw_va, detrend_raw_f = FS.detrend_raw_data(raw_vm, raw_va, raw_f)
vm_data, va_data, f_data = FS.define_data_window(detrend_raw_vm, detrend_raw_va, detrend_raw_f)

# sort PMUs based on highest energies
energy_vm, idx_vm = FS.energy_sort(vm_data)
energy_va, idx_va = FS.energy_sort(va_data)
energy_f, idx_f = FS.energy_sort(f_data)

# extract feature vectors for each channel
ch_vm, ch_va, ch_f = 'Vm', 'Va', 'F'
x_vm_features, ia_vm = FS.MPM(vm_data, ch_vm, idx_vm)
x_va_features, ia_va = FS.MPM(va_data, ch_va, idx_va)
x_f_features, ia_f = FS.MPM(f_data, ch_f, idx_f)

x_data = FS.return_event_features_matrix(x_vm_features, x_va_features, x_f_features)
FS.save_event_features_matrix_as_mat(x_data)





