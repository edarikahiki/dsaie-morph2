# This module stores the functions needed for loading and accessing the data used by Jagers (2003) to train the Neural Network

import torch #type: ignore

import numpy as np
import netCDF4 as nc #type: ignore
import torch.nn as nn #type: ignore
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, average_precision_score, PrecisionRecallDisplay

from postprocessing.metrics import compute_metrics

grey_cmap = ListedColormap(['palegoldenrod', 'navy']) # ['black', 'white']
diff_cmap = ListedColormap(['red', 'white', 'green'])
grey_diff_cmap = ListedColormap(['black', 'white'])

def load_ncd_tensor(path=r'data\jagers_nn\bsi_dump.nc', convert_tensor=True):
    '''
    Load dataset and convert it to tensor

    Inputs:
           path = str, path where file is stored.
                  default: r'data\jagers_nn\bsi_dump.nc'
           convert_tensor = bool, if True create tensor from file variables
                            Default: True
    
    Output:
           vars_tensor (pytorch.tensor) or dataset (nc.Dataset)
    '''
    dataset = nc.Dataset(path, mode='r')
    
    if convert_tensor:
        # extract variables
        variables = []
        for var_name in dataset.variables:
            var_data = np.array(dataset.variables[var_name])  # convert to numpy array
            variables.append(np.transpose(var_data)) # transpose to have correct orientation

        variables = variables[2:] # remove (x, y) coords
        vars_tensor = torch.tensor(variables)
        
        return vars_tensor 
    else:
        return dataset

def get_inp_tar_pred(vars_tensor=None, path=r'data\jagers_nn\bsi_dump.nc', var=[5, 6, 14]):
    '''
    Get input, target and prediction from the file variables or from an existing tensor/array
    and create a pytorch.tensor

    Inputs: 
           vars_tensor = tensor or None. If None, the key "path" is needed to load the file
                        default = None 
           path = str, path where file is stored.
                  default: r'data\jagers_nn\bsi_dump.nc' 
           var = list, contains the indexes of the input, target, and prediction variables
                 contained in the input file.
                 default: [5, 6, 14] 
    
    Output:
           input_target_pred = tensor, contains input, target, and prediction
    '''
    if vars_tensor is None:
        vars_tensor = load_ncd_tensor(path=path, convert_tensor=True)

    input_target_pred = torch.stack([vars_tensor[var[0]], vars_tensor[var[1]], vars_tensor[var[2]]])
    return input_target_pred

def get_loss_metrics(input_target_pred=None, vars_tensor=None, path=r'data\jagers_nn\bsi_dump.nc', 
                     var=[5, 6, 14], water_threshold=0.6, return_metrics=False):
    '''
    Compute classification loss and metrics of the Neural network by Jagers (2003).

    Inputs:
           input_target_pred = array/tensor or None. If None, the key "vars_tensor" or "path" 
                               are needed
                               default: None
           vars_tensor = tensor or None. If None, the key "path" is needed to load the file
                        default = None 
           path = str, path where file is stored.
                  default: r'data\jagers_nn\bsi_dump.nc' 
           var = list, contains the indexes of the input, target, and prediction variables
                 contained in the input file.
                 default: [5, 6, 14]
           water_threshold = float, value for binarizing model prediction
                             default: 0.6 (larger than 0.5 after visual check for the best fit)
           return_metrics = bool, if True the function outputs the metric scores, otherwise 
                            just a print statement.
                            default: False
            
    Output: metrics scores (loss, accuracy, precision, recall, f1_score, csi_score) if return_metrics = True
    '''
    if input_target_pred is None:
        input_target_pred = get_inp_tar_pred(vars_tensor=vars_tensor, path=path, var=var)
    
    loss = nn.BCELoss()(input_target_pred[2], input_target_pred[1])
    accuracy, precision, recall, f1_score, csi_score = compute_metrics(input_target_pred[2], input_target_pred[1], water_threshold=water_threshold)
    
    print(f'Metrics using water threshold = {water_threshold}:\n\
BCE loss:          {loss:.3e}\n\
Accuracy:          {accuracy:.3f}\n\
Precision:         {precision:.3f}\n\
Recall:            {recall:.3f}\n\
F1 score:          {f1_score:.3f}\n\
CSI score:         {csi_score:.3f}')
    
    
    return loss, accuracy, precision, recall, f1_score, csi_score if return_metrics else None
    
def plot_NN(input_target_pred=None, vars_tensor=None, path=r'data\jagers_nn\bsi_dump.nc', 
            var=[5, 6, 14], water_threshold=0.6, save_img=False):
    '''
    Plot input, target, prediction, misclassification map and binary prediction from the Neural Network by Jagers (2003)

    Inputs:
           input_target_pred = array/tensor or None. If None, the key "vars_tensor" or "path" 
                               are needed
                               default: None
           vars_tensor = tensor or None. If None, the key "path" is needed to load the file
                        default = None 
           path = str, path where file is stored.
                  default: r'data\jagers_nn\bsi_dump.nc' 
           var = list, contains the indexes of the input, target, and prediction variables
                 contained in the input file.
                 default: [5, 6, 14]
           water_threshold = float, value for binarizing model prediction
                             default: 0.6 (larger than 0.5 after visual check for the best fit)

    Output: None
    '''    
    if input_target_pred is None:
        input_target_pred = get_inp_tar_pred(vars_tensor=vars_tensor, path=path, var=var)

    binary_output95 = (input_target_pred[2] >= water_threshold).float()
    misclass_map = binary_output95 - input_target_pred[1]

    shp = input_target_pred[0].shape
    x_ticks = np.arange(0, shp[1], 100)
    y_ticks = np.arange(0, shp[0], 100)  

    # convert x_ticks and y_ticks from pixels to meters
    x_tick_labels = [round(tick * 50/1000, 2) for tick in x_ticks]  
    y_tick_labels = [round(tick * 50/1000, 2) for tick in y_ticks]

    fig, axes = plt.subplots(1, 5, figsize=(18, 10), sharey=True)

    for i in range(len(input_target_pred)):
        if i < 2:
            axes[i].imshow(input_target_pred[i], cmap=grey_cmap, vmin=0, vmax=1)
        else:
            axes[i].imshow(input_target_pred[i], cmap='gray', vmin=0, vmax=1)
        
    for ax in axes:
        ax.set_xticks(x_ticks, fontsize=14)  
        ax.set_xticklabels(x_tick_labels)  
        ax.set_yticks(y_ticks, fontsize=14)  
        ax.set_yticklabels(y_tick_labels)  
        ax.set_xlabel('Width (km)')
    
    axes[0].set_ylabel('Length (km)', fontsize=14) 
    axes[0].set_title('Binary input 1994\n')
    axes[1].set_title('Binary target 1995\n')
    # axes[2].imshow(mlp_output95, cmap='gray')
    axes[2].set_title('Prediction 1995\n')
    axes[3].imshow(binary_output95, cmap=grey_cmap, vmin=0, vmax=1)
    axes[3].set_title(f'Binary prediction 1995\nwith threshold = {water_threshold}', fontsize=14)
    im = axes[4].imshow(misclass_map, cmap=diff_cmap, vmin=-1, vmax=1)
    axes[4].imshow(input_target_pred[1], cmap=grey_diff_cmap, vmin=0, vmax=1, alpha=0.2)
    axes[4].set_title('Misclassification map\n(binary prediction - input)')
    
    if save_img:        
        plt.savefig(rf'images\report\4_results\jagers_NN\input_pred_NN.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
        plt.close()

    return None

def pr_curve(prediction, target, save_img=False):
    '''
    Plot or return the Precision-Recall curve as well as the value of the AUC
    Compute optimal threshold for water classification by maximizing the F1-score
    
    Inputs:
           prediction = array/tensor, contains model prediction
           target = array/tensor, contains model target
           save_img = bool, sets whether the image is saved
                      default: False
    
    Output:
           best_thr = float, optimal threshold that maximizes the F1-score 
    '''
    target_flat, prediction_flat = np.array(target.flatten()), np.array(prediction.flatten())
    precision, recall, thresholds = precision_recall_curve(target_flat, prediction_flat, pos_label=1)
    ap = average_precision_score(target_flat, prediction_flat)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    
    # random classifier
    positive_ratio = np.sum(target_flat)/len(target_flat)
    positive_ratio = positive_ratio
    
    plt.figure()
    plt.plot(recall, precision, color='navy', lw=2.5, label=f'PR curve (AUC = {ap:.3f})')
    plt.fill_between(recall, precision,  color='palegoldenrod')
    plt.axhline(positive_ratio, color='red', lw=2.5, linestyle='--', label=f'Random classifier (AP = {positive_ratio:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall [-]', fontsize=14)
    plt.ylabel('Precision [-]', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(f'Precision Recall curve', fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    formatted_best_thr = f'{best_thr:.3f}'
    plt.annotate(f'Water threshold for\nmax F1-score: {formatted_best_thr}', xy=(0.63,0.72), fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="palegoldenrod"))

    if save_img:        
        plt.savefig(rf'images\report\4_results\jagers_NN\pr_curve_NN.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
        plt.close()
 
    return best_thr

def roc_curve(prediction, target, save_img=False):
    '''
    This function is used to plot or return the FPR, TPR and ROC area of a single sample, depending on the use.
    A mask is applied to the target to make sure that 'no-data' pixels are not included in the computations.

    Inputs:
           prediction = array/tensor, contains model prediction
           target = array/tensor, contains model target
           save_img = bool, sets whether the image is saved
                      default: False  
    
    Output:
           None 
    '''
    # need to import skl again otherwise kernel crashes - no clue why 
    import sklearn.metrics as skl

    target_flat = np.array(np.copy(target).ravel())
    prediction_flat = np.array(np.copy(prediction).ravel())
    
    # compute ROC curve and ROC area
    fpr, tpr, _ = skl.roc_curve(target_flat, prediction_flat, pos_label=1) 
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='navy', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.fill_between(fpr, tpr,  color='palegoldenrod')
    plt.plot([0, 1], [0, 1], color='red', lw=2.5, linestyle='--', label=f'Random classifier = 0.5')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate [-]', fontsize=14)
    plt.ylabel('True Positive Rate[-]', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(f'Receiver OPerating Characteristic curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)

    if save_img:        
        plt.savefig(rf'images\report\4_results\jagers_NN\roc_curve_NN.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
        plt.close()
    
    return None

def metrics_thresholds(prediction=None, target=None, vars_tensor=None, path=r'data\jagers_nn\bsi_dump.nc', 
                       var=[5, 6, 14], nonwater=0, water=1, save_img=False):
    '''
    Plot metrics evolution with varying threshold.

    Inputs:
           prediction = array/tensor or None, if None the keys "vars_tensor" or "path" are needed.
                        defafult: None
           target = array/tensoror None, if None the keys "vars_tensor" or "path" are needed.
                        defafult: None
           vars_tensor = tensor or None. If None, the key "path" is needed to load the file
                        default = None 
           path = str, path where file is stored.
                  default: r'data\jagers_nn\bsi_dump.nc' 
           var = list, contains the indexes of the input, target, and prediction variables
                 contained in the input file.
                 default: [5, 6, 14] 
           nonwater = int, non-water areas pixel value
           water = int, water areas pixel value
           save_img = bool, if True image is saved.
                      default: False

    Output: None
    '''
    if prediction is None and target is None:
        input_target_pred = get_inp_tar_pred(vars_tensor=vars_tensor, path=path, var=var)
        prediction, target = input_target_pred[2], input_target_pred[1]
    
    accuracies, precisions, recalls, f1_scores, csi_scores = [], [], [], [], []
    thresholds = np.arange(0,1,0.05)

    for threshold in thresholds:
        accuracy, precision, recall, f1_score, csi_score = compute_metrics(prediction, target, nonwater_value=nonwater, 
                                                                           water_value=water, water_threshold=threshold)
        accuracies.append(accuracy), precisions.append(precision)
        recalls.append(recall), f1_scores.append(f1_score), csi_scores.append(csi_score)
    
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    plt.figure(figsize=(10,6))
    
    plt.plot(thresholds, accuracies, color='mediumblue', linewidth=2.5, label='accuracy', ls=(5, (10, 5)), zorder=2)
    plt.plot(thresholds, precisions, color='crimson', linewidth=2.5, label='precision', ls='-.', zorder=2) 
    plt.plot(thresholds, recalls, color='darkgoldenrod', linewidth=2.5, label='recall', zorder=2)
    plt.plot(thresholds, f1_scores, color='black', linewidth=2.5, label='F1-score', ls=(0, (3, 1, 1, 1, 1, 1)), zorder=2)
    plt.plot(thresholds, csi_scores, color='seagreen', linewidth=2.5, label='CSI-score', ls=(0, (5, 1)), zorder=2) 
    plt.axvline(best_thr, color='navy', linewidth=2.5, ls='--', zorder=1)

    plt.xlabel('Thresholds [-]', fontsize=14)
    plt.ylabel('Metrics [-]', fontsize=14)
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.xticks(np.arange(0, 1.1, 0.10), fontsize=13)
    plt.yticks(np.arange(0, 1.1, 0.10), fontsize=13)

    plt.legend(ncol=1, loc='lower right', fontsize=12) # bbox_to_anchor=(0.05, 0.05),
    formatted_best_thr = f'{best_thr:.3f}'
    plt.annotate(f'Water threshold for max\nF1-score: {formatted_best_thr}', xy=(best_thr+0.02,0.1), fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="navy", facecolor="palegoldenrod"))
    
    if save_img:        
        plt.savefig(rf'images\report\4_results\jagers_NN\metrics_with_thresholds_jagersNN.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close()  # close the figure to free memory
    else:
        plt.show()
    return None

def plot_all_vars(dataset=None, path=r'data\jagers_nn\bsi_dump.nc'):
    '''
    Plot all variables within a netCDF file in subplots with 5 plots per row.

    Inputs:
           dataset = array/tensor or None. If None, the key "path" is needed.
                     default: None
           path = str, path where file is stored.
                  default: r'data\jagers_nn\bsi_dump.nc'

    Output: None
    '''
    if dataset is None:
        dataset = load_ncd_tensor(path=path, convert_tensor=False)

    # list of variable names and skip the first two (x and y coordinates)
    variable_names = list(dataset.variables.keys())[2:]  
    num_vars = len(variable_names)
    num_cols = 4  
    num_rows = int(np.ceil(num_vars / num_cols))  # calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(num_cols * 5, num_rows * 5)) 
    axes = axes.ravel() 

    for idx, var_name in enumerate(variable_names):
        variable_data = dataset.variables[var_name][:]

        # ensure the data is at least 2D for plotting
        if variable_data.ndim >= 2:
            # take the first slice if there are more than 2 dimensions
            if variable_data.ndim > 2:
                variable_data = variable_data[0]  

            # get long_name attribute, use var_name if it doesn't exist
            long_name = dataset.variables[var_name].getncattr('long_name') if 'long_name' in dataset.variables[var_name].ncattrs() else var_name

            # rotate and flip to ensure correct direction
            im = axes[idx].imshow(np.flip(np.rot90(variable_data, k=1), axis=0), origin='lower')
            axes[idx].set_title(f'{long_name}\n{var_name}', fontsize=10)  
            fig.colorbar(im, ax=axes[idx], orientation='vertical')
        else:
            axes[idx].axis('off')  # if variable can't be plotted turn off the axis

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    return None