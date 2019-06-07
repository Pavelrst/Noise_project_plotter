import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import numpy as np

FIRST_N_EPOCHS = 50 # how many epochs do you want to plot?
COMACT = False # Plot legend

def main():
    rootdir = 'C:\\Users\\Pavel\\Desktop\\Noise_project_plotter\\results'

    for subdirs, dirs, files in os.walk(rootdir):
        # for file in files:
        #     print(os.path.join(subdir, file))
        for dir in dirs:
            print(dir)
            for subdirs, dirs, files in os.walk(rootdir + '\\' + dir):
                for file in files:
                    file_dir = rootdir + '\\' + dir + '\\' + file
                    if file == 'checkpoint.pth.tar':
                        os.remove(file_dir)
                    elif file == 'model_best.pth.tar':
                        os.remove(file_dir)
                    elif file == 'results.csv':
                        data = pd.read_csv(file_dir)
                        epochs_list = data['epoch']
                        val_error1_list = data['val_error1']

                        if len(epochs_list) > FIRST_N_EPOCHS:
                            epochs_list = epochs_list[:FIRST_N_EPOCHS]
                            val_error1_list = val_error1_list[:FIRST_N_EPOCHS]

                        min_err = np.min(val_error1_list)
                        val_error5_list = data['val_error5']
                    elif file == 'params.txt':
                        with open(file_dir) as f:
                            first_line = f.readline()
                            label = create_label(first_line, COMACT)
                label = label + dir + ': error=' + str(np.round(100*min_err,1)) + '% , acc=' + str(np.round(100*(1-min_err),1)) + '%'
                plt.plot(epochs_list, val_error1_list, label=label)

    plt.xlabel('epochs')
    plt.ylabel('top1 error')
    plt.legend()
    plt.show()

def create_label(line, compact=False):
    if compact:
        delim = '_'
    else:
        delim = '\n'

    label=''
    if '--no-quantization' in line:
        label +='no-quant'+delim
    else:
        label += 'quantized'+delim

    if '--no-noise' in line:
        label +='no-noise'+delim
    else:
        label += 'noise-add'+delim

    if '--no-act-quantization' in line:
        label +='no-act-quant'+delim
    else:
        label +='act-quantized'+delim

    if '--no-quant-edges' in line:
        label += 'no-quant-edge'+delim
    else:
        label += 'edge-quant'+delim

    if '--noise_mask' in line:
        label += 'noise_mask:' + line.split('--noise_mask')[1].split('--')[0]+delim

    if '--noise_interval_mul' in line:
        label += 'noise_mul:' + line.split('--noise_interval_mul')[1].split('--')[0]+delim

    if '--with_clamping True' in line:
        label += 'with_clamping'+delim
    else:
        label += 'no_clamping'+delim
    return label

if __name__ == "__main__":
    main()

