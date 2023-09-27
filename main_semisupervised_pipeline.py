from config_semisupervised_pipeline import *
from core.utils import *


dataset = Dataset(dataset_name=dataset_name,
                  x_data_path=x_data_path,
                  y_data_path=y_data_path,
                  class_labels=class_labels,
                  selected_class_labels=selected_class_labels,
                  shuffle_data=shuffle_data,
                  undersampling_method=undersampling_method,
                  random_state=random_state)
n_classes = dataset.n_classes

x_data, y_data, x_data_orig, y_data_orig, data_object, data_orig_object, class_labels_list = dataset._load_dataset()

# tsne = TSNE(n_components=2, random_state=random_state)
# x_data_transformed = tsne.fit_transform(x_data_temp)
# x_data = x_data_transformed
# x_data = normalize(x_data, axis=0)

n_samples = len(x_data)

version = (
    version_specification +        # A user-specified version specification or identifier
    dataset_name +                # The name or identifier of the dataset being used
    '-' + class_labels_list +     # List of class labels (or class names) in the dataset
    ')-nS(' + str(n_samples) +   # Number of total samples in the dataset
    ')-nK(' + str(n_folds) +      # Number of folds used for cross-validation
    ')-LUp(' + str(L_prcnt) +    # Percentage of labeled data (L) in the dataset
    ')-dU(' + str(step_to_unlabeled_as_labeled) +  # Number of steps to transition from labeled to unlabeled data
    ')-nF(' + str(number_of_selected_features) +  # Number of selected features for analysis
    ')-nQ(' + str(n_sets) +       # Number of sets or scenarios generated
    ')-nR(' + str(n_rand_max) +   # Maximum number of random seeds used
    ')-HT('  + str(hyperparameter_tuning) + ')'  # Hyperparameter tuning setting (e.g., True or False)
    '-rs(' + str(random_state) + ')'  # Random seed used for reproducibility
)
print('----------------------------------------------------')
print(f'Version: {version}')
print('----------------------------------------------------')
# simulation settings
sim_set = SemiSupSettings(version=version,
                          n_samples=n_samples,
                          n_classes=n_classes,
                          n_folds=n_folds,
                          L_prcnt=L_prcnt,
                          step_to_unlabeled_as_labeled=step_to_unlabeled_as_labeled,
                          n_sets=n_sets,
                          n_rand_max=n_rand_max,
                          number_of_selected_features=number_of_selected_features,
                          number_of_bootstrap_in_filter_method=number_of_bootstrap_in_filter_method,
                          plot_animation=plot_animation,
                          balance_range=balance_range,
                          hyperparameter_tuning=hyperparameter_tuning,
                          sort_l_u_distance=sort_l_u_distance,
                          print_idx_n_mix_rand=print_idx_n_mix_rand,
                          pseudo_label_model_list=pseudo_label_model_list,
                          test_model_list=test_model_list)


create_results_folder(folder_name=version)

SSEI = SemiSupervisedPipeline(sim_set=sim_set,
                              x_data=x_data,
                              y_data=y_data,
                              data_types_indices=data_types_indices)
df = SSEI.semisup_analysis_df
df_summary, auc_stats_params = SSEI.get_auc_stats_parameters()

plot_auc_stat_summary(SSEI)


# Object saving
obj_name = 'SSEI.pkl'
obj_path = os.path.join("results_final", SSEI.version, obj_name)

# Check if the directory exists and create if not
os.makedirs(os.path.dirname(obj_path), exist_ok=True)

# Saving the object
with open(obj_path, 'wb') as f:
    pickle.dump(SSEI, f)

# Version saving
version_path = os.path.join("results_final", SSEI.version, "version.txt")
with open(version_path, 'w') as f:
    f.write(SSEI.version)


print('----------------------------------------------------')
print(f'Version: {version}')
print('----------------------------------------------------')




#
label_spreading_models = ['ls_svmlin', 'ls_svmrbf', 'lss_knn', 'ls_dt', 'ls_gb']
fname_to_save = 'ls_all.jpg'
plt_tile = 'Label Spreading (LS)'
# data_types = ['n_mix_dp_tsne']
plot_auc_stats_comparison(SSEI,
                          fname_to_save=fname_to_save,
                          plt_title=plt_tile,
                          ss_model_names=label_spreading_models,
                          auc_statistics=['5th'],
                          plt_auc_type=['t_auc'],
                          plt_auc_calc_type=['ave'])

#
tsvm_models = ['tsvm_svmlin',  'tsvm_svmrbf', 'tsvm_knn','tsvm_dt', 'tsvm_gb']
fname_to_save = 'tsvm_all.jpg'
plt_tile = 'Transductive support vector machines (TSVM)'
plot_auc_stats_comparison(SSEI,
                          fname_to_save=fname_to_save,
                          plt_title=plt_tile,
                          ss_model_names=tsvm_models,
                          auc_statistics=['5th'],
                          plt_auc_type=['t_auc'],
                          plt_auc_calc_type=['ave'])


self_training_models = ['svmlin_svmlin','svmrbf_svmrbf', 'knn_knn', 'dt_dt', 'gb_gb']
fname_to_save = 'selfs_all.jpg'
plt_tile = 'Self-training'
plot_auc_stats_comparison(SSEI,
                          fname_to_save=fname_to_save,
                          plt_title=plt_tile,
                          ss_model_names=self_training_models,
                          auc_statistics=['5th'],
                          plt_auc_type=['t_auc'],
                          plt_auc_calc_type=['ave'])
# #

comparison_models = ['gb_gb', 'lss_knn']
fname_to_save = 'compare_gb_vs_ls_with_pl.jpg'
plt_tile = 'Comparison: (GB,GB) vs. (LS, kNN)'
plot_auc_stats_comparison(SSEI,
                          fname_to_save=fname_to_save,
                          plt_title=plt_tile,
                          ss_model_names=comparison_models,
                          auc_statistics=['ave', '5th', '95th'],
                          plt_auc_type=['t_auc'],
                          plt_auc_calc_type=['ave'],
                          n_col=2)
