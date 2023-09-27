I will consistently strive to keep the repository up to date and provide further documentation.

# Power System Machine Learning based Event Identification (PSMLEI)

This package provides tools for power system event analysis using machine learning techniques. It consists of three main modules:


## Event generation, feature extraction, and semisupervised pipeline setting
### Event Generation
- `config_event_generation.py`: Configuration file for event generation.
- `main_event_generation.py`: Main script for event generation.

### Feature Extraction
- `config_feature_extraction.py`: Configuration file for feature extraction.
- `main_feature_extraction.py`: Main script for feature extraction.

### Semi-Supervised Pipeline
- `config_semisupervised_pipeline.py`: Configuration file for the semi-supervised learning pipeline.
- `main_semisupervised_pipeline.py`: Main script for the semi-supervised learning pipeline.

These modules can be used together in a pipeline or independently for specific tasks. For detailed information and usage instructions for each module, please refer to their respective `config_x.py` and `main_x.py` files.

Feel free to explore and utilize these modules according to your power system event analysis needs.

###  `config_event_generation.py`

#### PSSE Configuration 
- `psse_set_minor`: Minor version of the PSSE software (e.g., X.4).
- `psseinit`: Parameters to initialize the PSSE environment.
- `psse_path`: Path to the PSSE installation directory.
- `project_path`: Path to the project directory.
- `system_files_path`: Directory containing synthetic network .raw and .dyr files.
- `raw_file_name`: Name of the .raw file.
- `dyr_file_name`: Name of the .dyr file.
- `psse_files_vers`: PSSE version of the .raw and .dyr files.
- `plot_network_flag`: Flag to plot the graph representation of the network.

#### Event Types
- `event_types_list`: List of event types for synthetic event data generation.

#### Event Component Ranges
- `event_load_list`: Range of components for load events.
- `event_gen_list`: Range of components for generation events.
- `event_line_list`: Range of components for line events.
- `event_bus_list`: Range of components for bus events.

#### PMU Placement
- `pmu_bus_select_method`: Method for PMU placement (random_in_range, random_in_buses, specify_buses).
- `pmu_bus_num`: List of PMU bus numbers (required for specify_buses method).
- `n_pmus`: Number of PMUs to place.

#### PSSE Dynamic Simulation Settings
- `flat_run_time`: Time for flat run dynamic simulation (seconds).
- `remove_fault_time`: Time to clear disturbances for bus and line fault events (seconds).
- `simulation_time`: Total dynamic simulation time (seconds).

#### System Loading Scenarios
- `generate_loading_scenarios`: Generate various system loading conditions (True/False).
- `start_load_level`: Lower bound for loading scenarios.
- `end_load_level`: Upper bound for loading scenarios.
- `n_scenarios`: Number of loading scenarios.
- `min_rand` and `max_rand`: Range for random load change ratios.



### `config_feature_extraction.py`

- `project_path`: Root directory of the Event Generation project.
- `generated_events_path`: Directory to store generated event data.
- `first_event`: First event number for data generation.
- `last_event`: Last event number for data generation.
- `first_sample`: Index of the first sample in the signal data.
- `last_sample`: Index of the last sample in the signal data.
- `n_pmus`: Total number of PMUs used in the simulation.
- `n_pmus_prime`: Number of PMUs with the largest energy of the signal.
- `Rank`: Rank approximation of the Hankel matrix for mode decomposition.
- `Ts`: Sampling rate of PMU data (seconds).
- `p`: Number of modes used in the mode decomposition via the matrix pencil method.
- `p_prime`: Number of distinct modes (considering one complex conjugate pair).
- `decimal_tol`: Decimal tolerance for numerical value rounding.


### `config_semisupervised_pipeline.py`


- `version_specification`: Minor version specification for the pipeline.

#### Dataset Configuration
- `dataset_name`: Name of the dataset.
- `x_data_path`: Path to the feature data file (e.g., x_data.mat).
- `y_data_path`: Path to the label data file (e.g., y_data.mat).
- `class_labels`: List of all possible class labels in the dataset.
- `selected_class_labels`: List of selected class labels to include in the dataset.
- `shuffle_data`: Whether to shuffle the data before splitting.
- `random_state`: Random seed for data shuffling.

#### Under-sampling Methods
- `undersampling_method`: Method for undersampling the data (e.g., Neighbourhood Cleaning Rule, Tomek Links). Set to `None` for no undersampling.

#### Data Types Selection
- `data_types_indices`: Indices corresponding to selected low-dimensional data representations (e.g., "n_mix_dp_mi").

#### Data Splits and Labeled/Unlabeled Data Settings
- `n_folds`: Number of folds for cross-validation.
- `L_prcnt`: Percentage of labeled data.
- `step_to_unlabeled_as_labeled`: Number of steps to transition from labeled to unlabeled.
- `n_sets`: Number of dataset variations.
- `n_rand_max`: Maximum random selections of unlabeled samples.
- `number_of_selected_features`: Number of features to select.
- `number_of_bootstrap_in_filter_method`: Number of bootstrap iterations.
- `print_idx_n_mix_rand`: Flag to print index of selected subsets.
- `balance_range`: Range of balance ratios for labeled samples.
- `sort_l_u_distance`: Flag to sort data by labeled-unlabeled distance.

#### Model Settings
- `hyperparameter_tuning`: Enable hyperparameter tuning.
- `plot_animation`: Enable plot animations.
- `pseudo_label_model_list`: List of pseudo-label models.
- `test_model_list`: List of test models.



## Other functionalities of the PSMLEI repositories 

### Dataset Class

The `Dataset` class is a versatile tool for handling and processing datasets. It offers functionality to load feature and label data, perform resampling, apply undersampling methods, and conduct data analysis. This class is particularly useful for machine learning tasks where data preprocessing is a crucial step.

#### Initialization

To create a `Dataset` object, you can provide the following parameters:

- `dataset_name` (str): Name of the dataset.
- `x_data_path` (str): Path to the feature data file (e.g., x_data.mat).
- `y_data_path` (str): Path to the label data file (e.g., y_data.mat).
- `class_labels` (list): List of all possible class labels in the dataset.
- `selected_class_labels` (list): List of selected class labels to include in the dataset.
- `shuffle_data` (bool): Whether to shuffle the data.
- `undersampling_method` (object): Undersampling method to apply to the data.
- `random_state` (int): Random seed for data shuffling.

#### Attributes

The `Dataset` object includes various attributes, including:

- `dataset_name`: Name of the dataset.
- `x_data_path`: Path to the feature data file.
- `y_data_path`: Path to the label data file.
- `class_labels`: List of all possible class labels in the dataset.
- `selected_class_labels`: List of selected class labels included in the dataset.
- `shuffle_data`: Whether to shuffle the data.
- `undersampling_method`: Undersampling method applied to the data.
- `random_state`: Random seed for data shuffling.
- `n_classes`: Number of classes in the dataset.
- `x_data_orig`: Original feature data.
- `y_data_orig`: Original label data.
- `data_orig_object`: Data analysis object for the original data.
- `x_data`: Processed feature data.
- `y_data`: Processed label data.
- `n_samples`: Number of samples in the dataset.
- `data_object`: Data analysis object for processed data.
- `class_labels_list`: String representation of selected class labels.

#### Resampling and Data Loading

The `Dataset` class provides methods to apply resampling techniques and load dataset files. The `_apply_resampling` method allows you to apply undersampling methods and shuffle the data. The `_load_dataset` method loads the dataset, applies resampling, and performs data analysis.

#### Example Usage

Here's an example of how to create and use a `Dataset` object:

```python
from core.utils import Dataset

# Initialize the Dataset object
dataset = Dataset(
    dataset_name="MyDataset",
    x_data_path="path/to/x_data.mat",
    y_data_path="path/to/y_data.mat",
    class_labels=[0, 1, 2, 3, 4],
    selected_class_labels=[0, 1, 2],
    shuffle_data=True,
    undersampling_method=None,
    random_state=42
)

# Load and preprocess the dataset
x_data, y_data, x_data_orig, y_data_orig, data_object, data_orig_object, class_labels_list = dataset._load_dataset()

# Access dataset attributes
print("Number of classes:", dataset.n_classes)
print("Number of samples:", dataset.n_samples)

# Perform data analysis
data_object.analyze_data()
data_orig_object.analyze_data()

# Display selected class labels
print("Selected Class Labels:", class_labels_list)
```




### DataAnalysis Class

The `DataAnalysis` class is designed for performing data analysis and visualization on a dataset. It includes various methods for generating statistical insights and visualizing data distributions.

#### Usage


```python
# Example Usage
from core.utils import DataAnalysis

# Create an instance of DataAnalysis with feature and target data
data_analyzer = DataAnalysis(x_data, y_data)

# Generate and print basic statistics for the feature data
data_analyzer.data_gen_stats()

# Generate and save a heatmap of the feature data's correlation
data_analyzer.data_heatmap()

# Generate and save histograms of the feature data
data_analyzer.data_histogram()

# Generate and save a PCA plot for the feature data
data_analyzer.data_pca_plot_v1()

# Genertate and save box plots for feature statistics
data_analyzer.data_feature_statistics()

# Generate and save box plots for individual features
data_analyzer.feature_box_plot()

# Compute and print statistics for multiclass data
data_analyzer.get_multiclass_stats()

# Generate and save a 2D PCA plot
data_analyzer.data_pca_plot_v2()

# Generate and save a t-SNE plot
data_analyzer.data_tsne_plot()

# Generate and save a heatmap of pairwise distances
data_analyzer.plot_distance_heatmap()

# Generate and return a DataFrame of feature importances
feature_importances = data_analyzer.plot_feature_importances()

# Generate and save scatter plots for the top important features
data_analyzer.plot_important_feature_scatter_plots()

# Generate and save a scatter plot for two specific features
data_analyzer.feature_scatter_plot(feature_1, feature_2)

# Visualize dataset statistics including histograms and box plots
data_analyzer.visualize_dataset_statistics()

# Visualize the dataset distribution using pairwise scatter plots and box plots
data_analyzer.visualize_dataset_distribution()
```



### LabeledUnlabeledIndicesGenerator Class

The `LabeledUnlabeledIndicesGenerator` class is designed for generating labeled and unlabeled indices for semi-supervised learning experiments. It allows users to control class balance and sort unlabeled samples based on distance to labeled samples.

#### Usage

```python
# Example Usage
from core.utils import LabeledUnlabeledIndicesGenerator

# Create an instance of LabeledUnlabeledIndicesGenerator with feature and target data
indices_generator = LabeledUnlabeledIndicesGenerator(x_data, y_data, sim_set)

# Generate labeled and unlabeled indices for each fold and set
indices_dict = indices_generator.generate_indices()

# Check the class balance of the labeled samples
is_balanced = indices_generator.check_class_balance(labeled_indices)

# Get the dictionary of generated labeled and unlabeled indices for specific fold and set
indices = indices_generator.get_indices_dict(fold_num, set_num)

# Sort unlabeled samples based on their distance to the closest labeled sample
sorted_unlabeled_indices = indices_generator.get_sorted_l_u_distances(labeled_indices, unlabeled_indices_rand)
```

### DatasetGenerator Class

The `DatasetGenerator` class is a versatile tool designed for generating datasets and low-dimensional representations for semi-supervised learning experiments. This class is particularly useful for researchers and practitioners in the field of machine learning who need to create custom datasets and perform feature selection and dimensionality reduction.

#### Constructor

To create a `DatasetGenerator` object, you can provide the following parameters:

- `x_data (ndarray)`: Input data samples.
- `y_data (ndarray)`: Corresponding labels for the input data.
- `indices_dict (dict)`: Dictionary containing labeled and unlabeled indices for each fold and set.
- `fold_num (int)`: The fold number.
- `set_num (int)`: The set number.
- `sim_set (object)`: Simulation settings object containing configuration parameters.

#### Attributes

The `DatasetGenerator` object includes various attributes, including:

- `x_data`: Input data samples.
- `y_data`: Corresponding labels for the input data.
- `indices_dict`: Dictionary containing labeled and unlabeled indices for each fold and set.
- `fold_num` and `set_num`: Fold and set identifiers.
- `n_labeled` and `n_unlabeled`: Number of labeled and unlabeled samples.
- `number_of_bootstrap_in_filter_method`: Number of bootstraps used in the feature selection step.
- `number_of_selected_features`: Number of features to be selected.
- `number_of_pca`: Number of principal components for low-dimensional representation.
- `print_idx_n_mix_rand`: Flag to control printing of selected samples.
- `random_state`, `n_rand_max`, `random_states_list`: Parameters for data shuffling and randomization.

#### Methods

The `DatasetGenerator` class provides several methods, including:

- `generate_dataset()`: Generate datasets for a specific fold and set.
- `get_selected_feature_indices()`: Get the indices of selected features based on the filter method using mutual information.
- `print_available_data_types()`: Prints available data types and their descriptions.
- `get_lowdim_dataset(n_rand_i, n_mix, data_type)`: Generate a low-dimensional representation of the dataset based on specified parameters.
- `choose_n_mix_samples(n_rand_i)`: Choose a subset of mixed samples for low-dimensional representation.

#### Example Usage

Here's an example of how to create and use a `DatasetGenerator` object:

```python
# Initialize the DatasetGenerator object
dataset_generator = DatasetGenerator(
    x_data=x_data, y_data=y_data, indices_dict=indices_dict, fold_num=1, set_num=1, sim_set=sim_set
)

# Generate datasets for a specific fold and set
d_data, d_train, d_test, d_train_l, d_train_u, d_train_u_true, d_train_mix, d_train_mix_true = dataset_generator.generate_dataset()

# Get the indices of selected features using the filter method
idx_mi_l, idx_mi_train = dataset_generator.get_selected_feature_indices()

# Print available data types for low-dimensional representation
data_types = dataset_generator.print_available_data_types()

# Generate a low-dimensional representation of the dataset
n_rand_i = 0
n_mix = 100
data_type = "n_mix_d"
d_train_n_mix, d_test_n_mix, d_train_n_mix_true = dataset_generator.get_lowdim_dataset(n_rand_i, n_mix, data_type)
```

### PsuedoLabelingAnimation Class

The `PsuedoLabelingAnimation` class is designed for creating animated visualizations of the pseudo-labeling process in machine learning. It visualizes how pseudo-labels are assigned to unlabeled data and how they change over time during the training process.

#### Constructor

To create a `PsuedoLabelingAnimation` object, you can provide the following parameters:

- `dataset (DatasetGenerator)`: DatasetGenerator object containing the dataset.
- `version (str)`: Version identifier.

#### Attributes

The `PsuedoLabelingAnimation` object includes various attributes, including:

- `x_train_l` and `y_train_l`: Labeled training data.
- `x_test` and `y_test`: Test data.
- `idx_mi_l`: Indices of selected features based on the filter method.
- `n_classes`: Number of unique classes in the labeled data.
- `colors`: List of distinct colors for different classes.
- `frames`: List to store animation frames.
- `version`: Version identifier.

#### Methods

The `PsuedoLabelingAnimation` class provides several methods, including:

- `generate_colors()`: Generates a list of distinct colors for different classes.
- `get_color(class_index)`: Gets the color for a specific class.
- `get_train_l_style(class_index)`: Gets the style for plotting labeled training samples of a specific class.
- `get_train_u_true_style(class_index)`: Gets the style for plotting true labels of unlabeled training samples of a specific class.
- `get_train_u_pseudo_style(class_index)`: Gets the style for plotting pseudo labels of unlabeled training samples of a specific class.
- `get_test_true_style(class_index)`: Gets the style for plotting true labels of test samples of a specific class.
- `get_test_pred_style(class_index)`: Gets the style for plotting predicted labels of test samples of a specific class.
- `create_empty_plots()`: Creates empty plots for animation.
- `init_animation()`: Initializes the animation by plotting the labeled training samples and test samples.
- `animate_pseudo_labels()`: Animates the pseudo-labeling process.
- `save_animation()`: Saves the animation as a GIF file.

#### Example Usage

Here's an example of how to create and use a `PsuedoLabelingAnimation` object:

```python
# Initialize the PsuedoLabelingAnimation object
animation = PsuedoLabelingAnimation(dataset, version)

# Create empty plots for animation
fig, ss_plot, ax = animation.create_empty_plots()

# Initialize the animation
animation.init_animation()

# Animate the pseudo-labeling process
animation.animate_pseudo_labels(test_model, d_train, d_test, d_train_true, y_pseudo, y_pred, n_mix, frame_cntr, n_rand_i)

# Save the animation as a GIF file
animation.save_animation(n_frames, data_type, fold_num, set_num, n_rand, pseudo_label_model, test_model, version)
```





### Acknowledgment

We acknowledge that in the Semi-supervised pipeline module, we have integrated the semi-supervised methods from the following repositories for a more comprehensive comparison between various semi-supervised algorithms:

- [semisup-learn](https://github.com/tmadl/semisup-learn)
- [Implementation-of-Transductive-SVM-Sklearn-Compatible](https://github.com/d12306/Implementation-of-Transductive-SVM-Sklearn-Compatible)

We greatly appreciate the contributions of these repositories to our project.


