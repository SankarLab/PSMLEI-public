
version_specification = "test_1"

"""
Dataset Configuration.
If you have used the event_generation and feature_extraction modules create the dataset, the x_data_path, y_data_path 
will be save in the project_directory\results\generated_events. 
For testing, I have shared x_data and y_data for a sample circles dataset in the project_directory\results\sample_circles_dataset
"""
#
dataset_name = "circles"  # Name of the dataset.
x_data_path = r'C:\Users\ntaghip1\PycharmProjects\Event_Generation_private\PSMLEI\results\sample_circles_dataset\x_data.mat'
y_data_path = r'C:\Users\ntaghip1\PycharmProjects\Event_Generation_private\PSMLEI\results\sample_circles_dataset\y_data.mat'
class_labels = [0, 1]  # List of all possible class labels in the dataset.
selected_class_labels = [0, 1]  # List of selected class labels to include in the dataset.
shuffle_data = True  # Whether to shuffle the data before splitting into train and test sets.
random_state = 110  # Random seed for data shuffling.
# ======================================================================================================================
# Under-sampling Methods:
"""
imblearn.under_sampling import (
    TomekLinks, NeighbourhoodCleaningRule, AllKNN, EditedNearestNeighbours,
    CondensedNearestNeighbour, ClusterCentroids, NearMiss, RepeatedEditedNearestNeighbours
)
1. Neighbourhood Cleaning Rule (NCR)
2. Condensed Nearest Neighbour (CNN)
3. All K-Nearest Neighbors (AllKNN)
4. Edited Nearest Neighbours (ENN)
5. Cluster Centroids (CC)
6. Tomek Links (TL)
7. NearMiss (NM)
8. Repeated Edited Nearest Neighbours (RENN)
"""
# Example usage:
# undersampling_method = NearMiss(version=3)
undersampling_method = None
# ======================================================================================================================
'''
d: total number of features
dp_mi: represents low dimensional representation obtained from filter method"
dp_pca: represents low dimensional representation obtained from pca"


data_types_indices = [0, 1, 3, ..]
0 -> "n_mix_d": "Original n_mix training (labeled and included unlabeled) samples and test samples",
1 -> "n_mix_dp_mi": "Low dimensional representation of the n_mix training (labeled and included unlabeled) samples and test samples based on the selected features obtained from the filter method",
2 -> "n_mix_dp_tsne": "Low dimensional representation of the n_mix training (labeled and included unlabeled) samples and test samples based on their projection on the t-SNE components",
3 -> "n_mix_dp_pca": "Low dimensional representation of the n_mix training (labeled and included unlabeled) samples and test samples based on their projection on the principal components",
4 -> "true_n_mix_d": "Original n_mix training (labeled and true label of the included unlabeled) samples and test samples",
5 -> "true_n_mix_dp_mi": "Low dimensional representation of the n_mix training (labeled and true label of the included unlabeled) samples and test samples based on the selected features obtained from the filter method",
6 -> "true_n_mix_dp_pca": "Low dimensional representation of the n_mix training (labeled and true label of the included unlabeled) samples and test samples based on their projection on the principal components",
7 -> "original_dp_mi": "Low dimensional representation of the training, and test samples based on the selected features obtained from the filter method",
8 -> "orig_dp_pca": "Low dimensional representation of the training, and test samples based on their projection on the principal components",

data_types_indices allows you to specify which low-dimensional representation to use. 
Simply provide a list of indices corresponding to the desired data types from the descriptions above. 
For example, if you want to use "n_mix_dp_mi," set data_types_indices = [1]. 
You can change the list to include multiple indices if needed.
'''

# Indices corresponding to selected data types
data_types_indices = [1]  # You can select the desired data type index here
# ======================================================================================================================
# Variables for Data Splits and Labeled/Unlabeled Data Settings:

# Number of folds for cross-validation. Determines data partitioning into subsets for cross-validation.
n_folds = 3
# Percentage of labeled data. Represents the labeled data proportion; the rest is unlabeled.
L_prcnt = 0.2
# Number of steps to transition from labeled to unlabeled data during training.
step_to_unlabeled_as_labeled = 50
# Number of sets to be generated, influencing dataset variations.
n_sets = 1
# Maximum number of random selection of subset of unlabeled samples at each step of including more unlabeled samples.
n_rand_max = 1
# Number of features to be selected in the feature selection step.
number_of_selected_features = 2

# Number of bootstrap iterations in the feature selection process.
number_of_bootstrap_in_filter_method = 20

# Flag to print index of the selected subset of labeled and unlabeled samples at each step of the simulation
print_idx_n_mix_rand = False

# Range of balance ratios of the selected subset of labeled samples.
balance_range = (0.2, 0.8)

# Flag to enable sorting based on labeled-unlabeled data distance.
sort_l_u_distance = False

# ======================================================================================================================
# Model settings:
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
# Example
hyperparameter_tuning = True
plot_animation = True
pseudo_label_model_list = ["ls", "tsvm", "svmrbf", "svmlin", "knn", "dt", "gb"] # ["ls", "tsvm", "gb"]
test_model_list = ["svmrbf", "svmlin", "knn", "dt", "gb"] #["svmrbf"] ["knn", "gb"]



