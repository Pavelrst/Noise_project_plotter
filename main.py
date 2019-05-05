import matplotlib.pyplot as plt
import os
import csv
import pandas as pd


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
                        val_error5_list = data['val_error5']
                        label = dir
                        plt.plot(epochs_list, val_error1_list, label=label)

    plt.xlabel('epochs')
    plt.ylabel('top1 error')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
