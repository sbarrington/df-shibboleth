import sys
import random
import pandas as pd
import os
import pathlib
import yaml
import disvoice
import librosa 
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import diff

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set WD
try:
    os.chdir('cadence_modelling/')
except:
    print(f'WD: {os.getcwd()}')

# Import configs
with open('/home/ubuntu/configs/config.yaml', 'r') as file:
    inputs = yaml.safe_load(file)
    
################################################## MODEL TRAINING SCRIPT ##################################################

def run_cadence_feature_extraction_pipeline(data_input_path = '../../data/TIMIT-fake', silence_threshold = 0.005, low_pass_filter_cutoff = 10):
    
    ## PREPROCESS
    # Extract input files
    print('Extracting input files')
    all_wav_files, all_flags = extract_input_files(data_input_path)
    
    # Obtain sample rate
    sr = librosa.load(all_wav_files[0])[1]
    
    # Balance data
    print('Balancing data')
    rebalanced_wav_files, rebalanced_flags = balance_data(all_wav_files)
    
    # Normalise amplitudes
    print('Normalizing amplitudes')
    normalized_audios = normalize_audio_amplitudes(rebalanced_wav_files)
    
    # Truncate silences
    print('Truncating silences')
    truncated_audios, start_ids, end_ids = truncate_silences(normalized_audios, silence_threshold, window_size=100)
    
    ## FEATURE ENGINEERING
    # Extract pauses 
    print('Extracting pauses')
    r_pauses, f_pauses = run_all_files(truncated_audios, rebalanced_flags, get_silence, silence_threshold)

    # Extract pause spreads
    print('Extracting pause spreads')
    r_silence_spreads, f_silence_spreads = run_all_files(truncated_audios, rebalanced_flags, get_silence_spread, silence_threshold)

    # Extract amplitude and derivative
    print('Extracting amplitude features')
    r_amps, f_amps = run_all_files(truncated_audios, rebalanced_flags, get_amplitude, silence_threshold, sample_rate=sr, cutoff_frequency=low_pass_filter_cutoff)
    
    ## FEATURE CONSOLIDATION
    # Create dataframe 
    print('Creating dataframe')
    features = pd.DataFrame({'file': rebalanced_wav_files, 
                         'pause_ratio':[item['ratio_pause_voiced'] for item in r_pauses + f_pauses], 
                         'pause_mean':[item['mean_of_silences'] for item in r_silence_spreads + f_silence_spreads], 
                         'pause_std':[item['spread_of_silences'] for item in r_silence_spreads + f_silence_spreads],  
                         'n_pauses':[item['n_pauses'] for item in r_silence_spreads + f_silence_spreads], 
                         'amp_deriv':[item['abs_deriv_amplitude'] for item in r_amps + f_amps],
                         'amp_mean':[item['mean_amplitude'] for item in r_amps + f_amps], 
                         'fake':rebalanced_flags})
    
    print('Complete')

    return features


################################################## MODEL TESTING SCRIPT ##################################################

def run_cadence_test(data_input_path, flags = [1], silence_threshold = 0.005, low_pass_filter_cutoff = 10):
    
    ## PREPROCESS
    # Extract input files
    print('Extracting input files')
    all_wav_files, _ = extract_input_files(data_input_path)
    
    # Obtain sample rate
    sr = librosa.load(all_wav_files[0])[1]
    
    # Normalise amplitudes
    print('Normalizing amplitudes')
    normalized_audios = normalize_audio_amplitudes(all_wav_files)
    
    # Truncate silences
    print('Truncating silences')
    truncated_audios, start_ids, end_ids = truncate_silences(normalized_audios, silence_threshold, window_size=100)
    
    ## FEATURE ENGINEERING
    # Extract pauses 
    print('Extracting pauses')
    r_pauses, f_pauses = run_all_files(truncated_audios, flags, get_silence, silence_threshold)

    # Extract pause spreads
    print('Extracting pause spreads')
    r_silence_spreads, f_silence_spreads = run_all_files(truncated_audios, flags, get_silence_spread, silence_threshold)

    # Extract amplitude and derivative
    print('Extracting amplitude features')
    r_amps, f_amps = run_all_files(truncated_audios, flags, get_amplitude, silence_threshold, sample_rate=sr, cutoff_frequency=low_pass_filter_cutoff)
    
    ## FEATURE CONSOLIDATION
    # Create dataframe 
    print('Creating dataframe')
    features = pd.DataFrame({'file': all_wav_files, 
                         'pause_ratio':[item['ratio_pause_voiced'] for item in r_pauses + f_pauses], 
                         'pause_mean':[item['mean_of_silences'] for item in r_silence_spreads + f_silence_spreads], 
                         'pause_std':[item['spread_of_silences'] for item in r_silence_spreads + f_silence_spreads],  
                         'n_pauses':[item['n_pauses'] for item in r_silence_spreads + f_silence_spreads], 
                         'amp_deriv':[item['abs_deriv_amplitude'] for item in r_amps + f_amps],
                         'amp_mean':[item['mean_amplitude'] for item in r_amps + f_amps], 
                         'fake':flags})
    
    
    print('Complete')

    return features

test_features = run_cadence_test('test_biden')

################################################## FUNCTIONS ##################################################

# Begin with list of files; here we use an example template while we await the full class ouput
def extract_input_files(data_input_path):

    all_wav_files = pathlib.Path(data_input_path)
    all_wav_files = list(all_wav_files.rglob("*.wav")) + list(all_wav_files.rglob("*.WAV"))
    all_wav_files = [str(file) for file in all_wav_files]

    #real_resampled_wav_files = [file for file in all_wav_files if 'TIMIT converted' in file]
    #fake_resampled_wav_files = [file for file in all_wav_files if not 'TIMIT converted' in file]

    flags = [1 if 'TIMIT converted' in str(item) else 0 for item in all_wav_files]
    
    return all_wav_files, flags #real_resampled_wav_files, fake_resampled_wav_files, flags


# NEED TO CHECK THIS - does it do it by architecture?
def balance_data(all_wav_files):
    
    folders = set([all_wav_files[i].split('_')[-1].split('.')[0] for i in range(len(all_wav_files))])
    
    real_resampled_wav_files = [file for file in all_wav_files if 'TIMIT converted' in file]
    fake_resampled_wav_files = [file for file in all_wav_files if not 'TIMIT converted' in file]
    
    # Ensure we take the same number of each phrase for real and fake, downsample the fake files 
    balanced_real_resampled_wav_files = []
    balanced_fake_resampled_wav_files = []
    
    for folder in folders:
        real_examples = [file for file in real_resampled_wav_files if f'_{folder}.' in file]
        fake_examples = [file for file in fake_resampled_wav_files if f'_{folder}.' in file]

        if len(real_examples) > len(fake_examples):
            real_examples = random.sample(real_examples, len(fake_examples))
        else:
            fake_examples = random.sample(fake_examples, len(real_examples))

        [balanced_real_resampled_wav_files.append(file) for file in real_examples]
        [balanced_fake_resampled_wav_files.append(file) for file in fake_examples]
    
    rebalanced_wav_files = balanced_real_resampled_wav_files + balanced_fake_resampled_wav_files
    rebalanced_flags = [i for i in np.zeros(len(balanced_real_resampled_wav_files))] + [i for i in np.ones(len(balanced_fake_resampled_wav_files))] 
    
    return rebalanced_wav_files, rebalanced_flags

def normalize_audio_amplitudes(all_wav_files):
    normalized_audios = []
    
    for file in all_wav_files:
        sample = librosa.load(file)[0]
        max_abs = np.max(np.abs(sample))
        normalized_sample = sample/max_abs
        normalized_audios.append(normalized_sample)
        
    return normalized_audios

def truncate_silences(normalized_audios, silence_threshold, window_size=100, counter=0):
    truncated_audios = []
    start_ids = []
    end_ids = []
    
    for audio in normalized_audios:
        counter += 1
        if counter % 100 == 0:
            print(f'Truncating audio {counter}/{len(normalized_audios)} ({round(counter*100/len(normalized_audios))}%)')

        for j in range(len(audio)):
            roll_average = np.mean(np.abs(audio[j:j+window_size]))
            if roll_average > silence_threshold:
                truncation_id_start = j
                break

        for j in reversed(range(len(audio))):
            roll_average = np.mean(np.abs(audio[j-window_size:j]))
            if roll_average > silence_threshold:
                truncation_id_end = j-window_size
                break
        truncated_audios.append(audio[truncation_id_start:truncation_id_end])
        start_ids.append(truncation_id_start)
        end_ids.append(truncation_id_end)
    
    return truncated_audios, start_ids, end_ids

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_silence(audio, percent, sample_rate=None, cutoff_frequency=None):
    thresh = max(abs(audio))*percent
    
    moving_avg = moving_average(abs(audio), 100)

    silent = np.where(abs(moving_avg) < thresh)
    voiced = np.where(abs(moving_avg) >= thresh)
    
    pct_pause = len(silent[0])*100/(len(silent[0])+len(voiced[0]))
    pct_voiced = len(voiced[0])*100/(len(silent[0])+len(voiced[0]))
    ratio_pause_voiced = len(silent[0])/len(voiced[0]) 

    return {'pct_pause':pct_pause, 'pct_voiced': pct_voiced, 'ratio_pause_voiced': ratio_pause_voiced}

def get_silence_spread(audio, percent, sample_rate=None, cutoff_frequency=None):

    thresh = max(abs(audio))*percent
    
    moving_avg = moving_average(abs(audio), 100)

    silent_windows = np.where(moving_avg < thresh)
    moving_avg[silent_windows] = 0
    silence_count = 0
    silence_counts = []
    
    for i in range(len(moving_avg)-1):
        item = moving_avg[i]
        next_item = moving_avg[i+1]
        
        if item != 0 and next_item == 0:
            silence_count = 0
            
        elif item == 0 and next_item == 0:
            silence_count += 1
            
        elif item == 0 and next_item != 0:
            silence_counts.append(silence_count)
        
        else:
            continue  
    
    # Get spreads/means and normalise
    spread_of_silences = np.std(silence_counts)/len(moving_avg)
    mean_of_silences = np.mean(silence_counts)/len(moving_avg)
    n_pauses = len(silence_counts)
        
    return {'spread_of_silences':spread_of_silences, 'mean_of_silences':mean_of_silences, 'silence_counts':silence_counts, 'n_pauses':n_pauses}


def run_all_files(truncated_audios, flags, function, percent, sample_rate=None, cutoff_frequency=None):
    # Instantiate results - r=real, f=fake
    r_results = []
    f_results = []

    real_indices = np.array([int(i) for i in range(len(flags)) if flags[i] == 0])
    fake_indices = np.array([int(i) for i in range(len(flags)) if flags[i] != 0])
    
    real_examples = [truncated_audios[i] for i in real_indices]
    fake_examples = [truncated_audios[i] for i in fake_indices]

    for item in real_examples:
        r_result = function(item, percent, sample_rate, cutoff_frequency)
        r_results.append(r_result)

    for item in fake_examples:
        f_result = function(item, percent, sample_rate, cutoff_frequency)
        f_results.append(f_result)
    
    return r_results, f_results

def filter_signal(audio, sample_rate, cutoff_frequency):
    t = np.arange(len(audio)) / sample_rate 
    w = cutoff_frequency / (sample_rate / 2) 
    b, a = signal.butter(5, w, 'low')
    smoothed_signal = signal.filtfilt(b, a, audio)
    
    return smoothed_signal

def get_amplitude(audio, percent, sample_rate, cutoff_frequency):

    abs_audio = abs(audio)
    smoothed_signal = filter_signal(abs_audio, sample_rate, cutoff_frequency)
    
    deriv_amplitude = np.mean(diff(smoothed_signal))
    mean_amplitude = np.mean(smoothed_signal)
    
        
    return {'abs_deriv_amplitude':abs(deriv_amplitude), 'mean_amplitude':mean_amplitude}

################################################## RUN MAIN ##################################################

features = run_cadence_feature_extraction_pipeline()