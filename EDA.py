import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import IPython.display as display

# Followed tutorial from: https://www.kaggle.com/code/drcapa/esc-50-eda-pytorch/notebook
def load_data(path):
    return pd.read_csv(path)

# Helper functions for EDA
def plot_signal_and_spectogram(data_array, samplerate, filepath):
    fig, axs = plt.subplots(1, 2, figsize=(22, 5))
    fig.subplots_adjust(hspace = .1, wspace=.2)
    axs = axs.ravel()
    x = range(len(data_array))
    y = data_array
    axs[0].plot(x, y)
    axs[0].grid()
    axs[1].specgram(data_array,Fs=samplerate, mode='psd', scale='dB')
    axs[0].set_title('Signal')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Amplitude')
    axs[1].set_title('Spectogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    plt.grid()
    plt.savefig(filepath)
    plt.close()

def visualize(path_list, resultant_path):
    if not os.path.exists(resultant_path):
        os.mkdir(resultant_path)

    for path in path_list:
        filepath = resultant_path + '/' + path.split('/')[2].split('.')[0] + '.png'
        #print(filepath)
        data_array, samplerate = sf.read(path)
        plot_signal_and_spectogram(data_array, samplerate, filepath)
    
    # print('DONE')


    
def main():
    df = load_data('ESC-50/meta/esc50.csv')
    #print(df.columns)
    path_list = ['ESC-50/audio/' + audio for audio in list(df['filename'])]
    #print(path_list)
    visualize(path_list, 'vis')
    
if __name__ =='__main__':
    main()