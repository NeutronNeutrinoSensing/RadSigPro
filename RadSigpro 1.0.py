# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 05:24:07 2021
@author: bswellons
"""

import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import math
from os import path

def Raw_Pulse_Correction(file, baseline, pulses_per_csv = 100000, cfd_delay = 6, attenuation_fraction = 0.25):
## This code corrects the raw compass data files into more readable files, better units (ADC to mV), and creates a list of time data for each pulse.

    mpl.rc('font',family='Times New Roman')
    mpl.rc('font', size = 16)

    num_rows = sum(1 for line in open(file)) - 1     #This part finds out the number of rows of data in the file so that arrays can be pre-allocated of that number later.
    #num_rows = 30                 #This part allows you to manually limit the number of rows that will be pre-allocated, use this when only looking at a set number of rows.

    sample_start_index = 0
    timetag_index = 0
    sample_length = 0
    with open(file, newline='') as f:               #This part sets up the length of the samples, it should be 496 but this allows for if it isnt.
        csv_reader = csv.reader(f, delimiter=';')
        for counter,line in enumerate(csv_reader):
            if counter == 0:
                for counter2,i in enumerate(line):
                    if i == 'TIMETAG':
                        timetag_index = counter2
                    if i == 'SAMPLES':
                        sample_start_index = counter2
            if counter > 0:
                sample_length = len(line[sample_start_index:])
            if counter > 0:              #Don't change the counter limiter in this if statement, its form only checks the length of the first row of samples to save time.
                break

    ##These sections pre-allocate arrays to then later fill with data edited from the csv file, this is done to save computing time.
    if num_rows >= pulses_per_csv:
        pulses_corrected_first = np.zeros(sample_length)
        pulses_corrected = np.zeros(shape=(pulses_per_csv, sample_length))
        inverted_delayed_signal = np.zeros(sample_length)
        attenuated_signal = np.zeros(sample_length)
        times_of_pulses = np.zeros(shape=(pulses_per_csv, sample_length))
    else:
        pulses_corrected_first = np.zeros(sample_length)
        pulses_corrected = np.zeros(shape=(num_rows, sample_length))
        inverted_delayed_signal = np.zeros(sample_length)
        attenuated_signal = np.zeros(sample_length)
        times_of_pulses = np.zeros(shape=(num_rows, sample_length))

    #max_pulse_heights = np.zeros(num_rows)        #This would allow us to store the height of each pulse, its not necessary for tail to total.
    #max_height_times = np.zeros(num_rows)         #This would allow us to store the times of the height of each pulse, its not necessary for tail to total.

    raw_pulse = np.zeros(sample_length)

    ##This section takes the sample data and time tags in the csv file, manipulates them, and then puts them into arrays of each pulse and their corresponding times.
    for y in range(math.ceil(num_rows/pulses_per_csv)):

        start_pulse = y*pulses_per_csv
        final_pulse = start_pulse + pulses_per_csv
        if num_rows < final_pulse:
            final_pulse = num_rows

        with open(file, newline='') as f:
            csv_reader = csv.reader(f, delimiter=';')       #Opens the csv file and reads it (not the most effecient way but changing code would take too much time)
            for counter,line in enumerate(csv_reader):
                if counter > start_pulse and counter <= final_pulse:

                    '''#Creats plot of raw pulse.
                    if int(min(line[sample_start_index:])) <= 2000:

                        print(counter)
                        time = np.zeros(496)
                        for counter,i in enumerate(time):
                            time[counter] = counter*2

                        raw_pulse[:] = [float(p) for p in line[sample_start_index:]]

                        plt.plot(time,raw_pulse, color='b')
                        plt.xlabel('Time (ns)')
                        plt.ylabel('ADC Units')
                        plt.show()'''

                    #print(line[sample_start_index], line[sample_start_index+1], line[sample_start_index+2], line[sample_start_index+3], line[sample_start_index+4])
                    #Constant Baseline Method
                    #baseline = float(line[sample_start_index])
                    #pulses_corrected[counter-1] = [(-float(p)+baseline) for p in line[sample_start_index:]]

                    #Baseline Freeze at Pulse Start Method
                    pulses_corrected_first[:] = [(-float(p)+baseline)*0.1220703125 for p in line[sample_start_index:]]
                    for i,height in enumerate(pulses_corrected_first):
                        if height >= max(pulses_corrected_first):
                            pulses_corrected[counter-start_pulse-1] = [float(p-float(np.average(pulses_corrected_first[i-10:i-5]))) for p in pulses_corrected_first]
                            '''if pulses_corrected_first[i-5] >= 0.15*max(pulses_corrected_first):
                                plt.scatter(np.array(range(len(pulses_corrected_first))), pulses_corrected_first, s=3, color='r')
                                plt.scatter(i, pulses_corrected_first[i])
                                plt.scatter(i-5, pulses_corrected_first[i-5], s=3)
                                plt.show()
                                plt.plot(np.array(range(len(pulses_corrected_first))), pulses_corrected_first, color='r')
                                plt.show()'''
                            break

                    #print(f'Pulse {counter-1} Length: {len(line[sample_start_index:-cfd_delay])}')
                    inverted_delayed_signal[(cfd_delay//2):] = [(-float(p)+baseline) for p in line[sample_start_index:-cfd_delay//2]]
                    #inverted_delayed_signal[(cfd_delay//2):] = inverted_delayed_signal_first[cfd_delay::2]

                    attenuated_signal[:] = [attenuation_fraction*(float(p)-baseline) for p in line[sample_start_index:]]
                    #attenuated_signal[:] = attenuated_signal_first[::2]

                    shaped_signal = inverted_delayed_signal + attenuated_signal
                    shaped_signal_flipped = np.flip(shaped_signal)

                    index_max_height = 0
                    index_sbzc = 0
                    index_sazc = 0
                    for counter1,i in enumerate(shaped_signal_flipped):
                        if i >= max(shaped_signal_flipped):
                            index_max_height = counter1
                    for counter2,i in enumerate(shaped_signal_flipped[index_max_height:]):
                        if i <= 0:
                            index_sbzc = (sample_length - 1) - (counter2 + index_max_height)
                            index_sazc = index_sbzc + 1
                            #print(f'sample length = {sample_length}, Counter = {counter2}, max height index = {index_max_height}')
                            break
                    if shaped_signal[index_sbzc] == 0:
                        t_sbzc = float(line[timetag_index])/(10**3)
                    else:
                        #print(f'SAZC = {shaped_signal[index_sazc]}, index = {index_sazc}')
                        #print(f'SBZC = {shaped_signal[index_sbzc]}, index = {index_sbzc}')
                        t_fine = (-float(shaped_signal[index_sbzc])/(float(shaped_signal[index_sazc]) - float(shaped_signal[index_sbzc]))) * 2.0  #interpolation to find time from SBZC to ZC
                        t_sbzc = float(line[timetag_index])/(10**3) - t_fine

                    time_list = (np.arange(-index_sbzc+1, sample_length+1 - index_sbzc, dtype=np.float64)*2) + t_sbzc
                    times_of_pulses[counter-start_pulse-1] = time_list     #Creates array of time correspondiong to each pulse data point, starting from timetag bin location, and increasing/decreasing by 2ns around it.

            #Makes all negative values 0
            pulses_corrected = np.where(pulses_corrected < 0, 0, pulses_corrected)


        #Writes files with the processed data in it.
        '''f = open(f'{file[:-4]}_CorrectedPulses{y+1}.csv', 'w')
        f.write(",".join([f"time {k+1},pulse {k+1}" for k in range(start_pulse, final_pulse)])+ "\n")
        for j in range(sample_length):
            f.write(",".join([f"{times_of_pulses[k][j]},{pulses_corrected[k][j]}" for k in range(0, pulses_per_csv)]) + "\n")
        f.close()'''

    #Writes files with the processed data in it.
    for i in range(math.ceil(num_rows/pulses_per_csv)):
        start_pulse = i*pulses_per_csv
        final_pulse = start_pulse + pulses_per_csv
        if num_rows < final_pulse:
            final_pulse = num_rows
        f = open(f'{file[:-4]}_CorrectedPulses{i+1}.csv', 'w')
        f.write(",".join([f"time {k+1},pulse {k+1}" for k in range(start_pulse, final_pulse)])+ "\n")
        for j in range(sample_length):
            f.write(",".join([f"{times_of_pulses[k][j]},{pulses_corrected[k][j]}" for k in range(start_pulse, final_pulse)]) + "\n")
        f.close()



def PSD_Analysis(gamma_file, neutron_file = 'Not a File', min_height_allowed = 0, max_height_allowed = 1000, long_gate = 360, pregate = 50, short_gate = 70, cfd_delay = 6, attenuation_fraction = 0.25):
## This function makes a pulse shape discrimination plot of both the gamma and neutron corrected data on the same graph, with selected max and min pulse heights (pulse start and end shouldn't usually be changed).

    mpl.rc('font',family='Times New Roman')
    mpl.rc('font', size = 16)

    num_total_pulses = sum(1 for line in open(f'{gamma_file[:-21]}.csv')) - 1
    sample_length = (sum(1 for line in open(f'{gamma_file[:-5]}1.csv')) - 1)
    first_file = pd.read_csv(f'{gamma_file[:-5]}1.csv')
    pulses_per_csv = len(first_file.columns)//2

    time_data1 = np.zeros(shape=(num_total_pulses, sample_length))
    pulse_data1 = np.zeros(shape=(num_total_pulses, sample_length))
    time_data2 = np.zeros(shape=(num_total_pulses, sample_length))
    pulse_data2 = np.zeros(shape=(num_total_pulses, sample_length))

    num_pulses_inserted1 = 0
    for i in range(math.ceil(num_total_pulses/pulses_per_csv)):
        if path.exists(f'{gamma_file[:-5]}{i+1}.csv') == True:
            data = pd.read_csv(f'{gamma_file[:-5]}{i+1}.csv')
            num_pulses = len(data.columns)//2
            time_data1[num_pulses_inserted1:(num_pulses + num_pulses_inserted1)] = data[[f'time {j+1}' for j in range(num_pulses_inserted1, num_pulses + num_pulses_inserted1)]].to_numpy().T
            pulse_data1[num_pulses_inserted1:(num_pulses + num_pulses_inserted1)] = data[[f'pulse {j+1}' for j in range(num_pulses_inserted1, num_pulses + num_pulses_inserted1)]].to_numpy().T
            num_pulses_inserted1 += num_pulses
        else:
            break

    num_pulses_inserted2 = 0
    for i in range(math.ceil(num_total_pulses/pulses_per_csv)):
        if path.exists(f'{neutron_file[:-5]}{i+1}.csv') == True:
            data = pd.read_csv(f'{neutron_file[:-5]}{i+1}.csv')
            num_pulses = len(data.columns)//2
            time_data2[num_pulses_inserted2:(num_pulses + num_pulses_inserted2)] = data[[f'time {j+1}' for j in range(num_pulses_inserted2, num_pulses + num_pulses_inserted2)]].to_numpy().T
            pulse_data2[num_pulses_inserted2:(num_pulses + num_pulses_inserted2)] = data[[f'pulse {j+1}' for j in range(num_pulses_inserted2, num_pulses + num_pulses_inserted2)]].to_numpy().T
            num_pulses_inserted2 += num_pulses
        else:
            break


    total_integrals1 = np.zeros(num_total_pulses)
    tail_integrals1 = np.zeros(num_total_pulses)
    gamma_tail_to_total = np.zeros(num_total_pulses)
    total_integrals2 = np.zeros(num_total_pulses)
    tail_integrals2 = np.zeros(num_total_pulses)
    neutron_tail_to_total = np.zeros(num_total_pulses)

    inverted_delayed_signal1 = np.zeros(sample_length)
    attenuated_signal1 = np.zeros(sample_length)
    shaped_signal1 = np.zeros(sample_length)
    inverted_delayed_signal2 = np.zeros(sample_length)
    attenuated_signal2 = np.zeros(sample_length)
    shaped_signal2 = np.zeros(sample_length)

    index_of_discrimination1 = []
    index_of_discrimination2 = []

    gamma_negative = 0
    neutron_negative = 0
    total_number = 0
    ##This section does the intergration for tail and total and then calculates the ratio of the two.
    for counter, (pulse1,times1,pulse2,times2) in enumerate(zip(pulse_data1,time_data1,pulse_data2,time_data2)):

        max_height1 = max(pulse1)
        max_height2 = max(pulse2)

        inverted_delayed_signal1[(cfd_delay//2):] = pulse1[:-(cfd_delay//2)]
        attenuated_signal1 = -attenuation_fraction*pulse1
        shaped_signal1 = inverted_delayed_signal1 + attenuated_signal1
        shaped_signal_flipped1 = np.flip(shaped_signal1)
        inverted_delayed_signal2[(cfd_delay//2):] = pulse2[:-(cfd_delay//2)]
        attenuated_signal2 = -attenuation_fraction*pulse2
        shaped_signal2 = inverted_delayed_signal2 + attenuated_signal2
        shaped_signal_flipped2 = np.flip(shaped_signal2)

        index_max_height1 = 0
        index_sbzc1 = 0
        index_of_tail_start1 = 0
        index_max_height2 = 0
        index_sbzc2 = 0
        index_of_tail_start2 = 0
        for counter1,i in enumerate(shaped_signal_flipped1):
            if i >= max(shaped_signal_flipped1):
                index_max_height1 = counter1
        for counter2,i in enumerate(shaped_signal_flipped1[index_max_height1:]):
            if i <= 0:
                index_sbzc1 = (sample_length - 1) - (counter2 + index_max_height1)
                index_of_tail_start1 = index_sbzc1 - pregate//2 + short_gate//2
                break
        for counter3,i in enumerate(shaped_signal_flipped2):
            if i >= max(shaped_signal_flipped2):
                index_max_height2 = counter3
        for counter4,i in enumerate(shaped_signal_flipped2[index_max_height2:]):
            if i <= 0:
                index_sbzc2 = (sample_length - 1) - (counter4 + index_max_height2)
                index_of_tail_start2 = index_sbzc2 - pregate//2 + short_gate//2
                break



        #Method where integration starts at pregate.
        if max_height1 >= min_height_allowed and max_height1 <= max_height_allowed and (index_sbzc1 - pregate//2) > 0:   #Sets a max mV limit on the pulses counted towards the tail to total.
            total_number += 1
            tail_integrals1[counter] = np.trapz(pulse1[index_of_tail_start1 : index_sbzc1 - pregate//2 + (long_gate//2)], times1[index_of_tail_start1 : index_sbzc1 - pregate//2 + (long_gate//2)]) #Takes tail integral from tail start to after pulse body.
            total_integrals1[counter] = np.trapz(pulse1[index_sbzc1 - pregate//2 : index_sbzc1 - pregate//2 + (long_gate//2)], times1[index_sbzc1 - pregate//2 : index_sbzc1 - pregate//2 + (long_gate//2)])              #Takes the total intergral from pulse start to after pulse body.
            index_of_discrimination1.append(counter)
            if  tail_integrals1[counter]/total_integrals1[counter] < 0:
                gamma_negative += 1
                plt.plot(times1, pulse1)
                plt.axvline(x=times1[index_sbzc1 - pregate//2], label = 'Pulse Start', color='g')
                plt.axvline(x=times1[index_of_tail_start1], label = 'Tail Start', color='r')
                plt.axvline(x=times1[index_sbzc1 - pregate//2], label = 'Pulse End', color='y')
                plt.plot(times1, np.zeros(len(times1)), alpha=0.7, label = '0 mV Mark', color='cyan')
                plt.legend()
                plt.xlabel('Time (ns)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'Gamma Pulse {counter}')
                plt.show()
        if max_height1 >= min_height_allowed and max_height1 <= max_height_allowed and (index_sbzc1 - pregate//2) <= 0:   #Sets a max mV limit on the pulses counted towards the tail to total.
            total_number += 1
            tail_integrals1[counter] = np.trapz(pulse1[index_of_tail_start1 : index_sbzc1 - pregate//2 + (long_gate//2)], times1[index_of_tail_start1 : index_sbzc1 - pregate//2 + (long_gate//2)]) #Takes tail integral from tail start to after pulse body.
            total_integrals1[counter] = np.trapz(pulse1[: index_sbzc1 - pregate//2 + (long_gate//2)], times1[: index_sbzc1 - pregate//2 + (long_gate//2)])              #Takes the total intergral from pulse start to after pulse body.
            index_of_discrimination1.append(counter)
            if  tail_integrals1[counter]/total_integrals1[counter] < 0:
                gamma_negative += 1
                plt.plot(times1, pulse1)
                plt.axvline(x=times1[0], label = 'Pulse Start', color='g')
                plt.axvline(x=times1[index_of_tail_start1], label = 'Tail Start', color='r')
                plt.axvline(x=times1[index_sbzc1 - pregate//2], label = 'Pulse End', color='y')
                plt.plot(times1, np.zeros(len(times1)), alpha=0.7, label = '0 mV Mark', color='cyan')
                plt.legend()
                plt.xlabel('Time (ns)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'Gamma Pulse {counter}')
                plt.show()

        if max_height2 >= min_height_allowed and max_height2 <= max_height_allowed and (index_sbzc2 - pregate//2) > 0:
            tail_integrals2[counter] = np.trapz(pulse2[index_of_tail_start2 : index_sbzc2 - pregate//2 + (long_gate//2)], times2[index_of_tail_start2 : index_sbzc2 - pregate//2 + (long_gate//2)]) #Takes tail integral from tail start to after pulse body.
            total_integrals2[counter] = np.trapz(pulse2[index_sbzc2 - pregate//2 : index_sbzc2 - pregate//2 + (long_gate//2)], times2[index_sbzc2 - pregate//2 : index_sbzc2 - pregate//2 + (long_gate//2)])              #Takes the total intergral from pulse start to after pulse body.
            index_of_discrimination2.append(counter)
            if  tail_integrals2[counter]/total_integrals2[counter] < 0:
                neutron_negative += 1
                plt.plot(times2, pulse2)
                plt.axvline(x=times2[index_sbzc2 - pregate//2], label = 'Pulse Start', color='g')
                plt.axvline(x=times2[index_of_tail_start2], label = 'Tail Start', color='r')
                plt.axvline(x=times2[index_sbzc2 - pregate//2], label = 'Pulse End', color='y')
                plt.plot(times2, np.zeros(len(times2)), alpha=0.7, label = '0 mV Mark', color='cyan')
                plt.legend()
                plt.xlabel('Time (ns)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'Neutron Pulse {counter}')
                plt.show()
        if max_height2 >= min_height_allowed and max_height2 <= max_height_allowed and (index_sbzc2 - pregate//2) <= 0:
            tail_integrals2[counter] = np.trapz(pulse2[index_of_tail_start2 : index_sbzc2 - pregate//2 + (long_gate//2)], times2[index_of_tail_start2 : index_sbzc2 - pregate//2 + (long_gate//2)]) #Takes tail integral from tail start to after pulse body.
            total_integrals2[counter] = np.trapz(pulse2[: index_sbzc2 - pregate//2 + (long_gate//2)], times2[: index_sbzc2 - pregate//2 + (long_gate//2)])              #Takes the total intergral from pulse start to after pulse body.
            index_of_discrimination2.append(counter)
            if  tail_integrals2[counter]/total_integrals2[counter] < 0:
                neutron_negative += 1
                plt.plot(times2, pulse2)
                plt.axvline(x=times2[0], label = 'Pulse Start', color='g')
                plt.axvline(x=times2[index_of_tail_start2], label = 'Tail Start', color='r')
                plt.axvline(x=times2[index_sbzc2 - pregate//2], label = 'Pulse End', color='y')
                plt.plot(times2, np.zeros(len(times2)), alpha=0.7, label = '0 mV Mark', color='cyan')
                plt.legend()
                plt.xlabel('Time (ns)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'Neutron Pulse {counter}')
                plt.show()


        #Method where integration starts at 3 indexes before pulse height.
        '''pulse1_max_index = 0
        pulse2_max_index = 0
        for counter5,i in enumerate(pulse1):
            if i >= max(pulse1):
                pulse1_max_index = counter5
        for counter6,i in enumerate(pulse2):
            if i >= max(pulse2):
                pulse2_max_index = counter6

        if max_height1 >= min_height_allowed and max_height1 <= max_height_allowed:   #Sets a max mV limit on the pulses counted towards the tail to total.
            total_number += 1
            tail_integrals1[counter] = np.trapz(pulse1[index_of_tail_start1 : index_sbzc1 - pregate//2 + (long_gate//2)], times1[index_of_tail_start1 : index_sbzc1 - pregate//2 + (long_gate//2)]) #Takes tail integral from tail start to after pulse body.
            total_integrals1[counter] = np.trapz(pulse1[pulse1_max_index - 3 : index_sbzc1 - pregate//2 + (long_gate//2)], times1[pulse1_max_index - 3 : index_sbzc1 - pregate//2 + (long_gate//2)])              #Takes the total intergral from pulse start to after pulse body.
            index_of_discrimination1.append(counter)
            if  tail_integrals1[counter]/total_integrals1[counter] < 0:
                gamma_negative += 1
                plt.plot(times1, pulse1)
                plt.axvline(x=times1[index_sbzc1 - pregate//2], label = 'Pulse Start', color='g')
                plt.axvline(x=times1[index_of_tail_start1], label = 'Tail Start', color='r')
                plt.axvline(x=times1[index_sbzc1 - pregate//2], label = 'Pulse End', color='y')
                plt.plot(times1, np.zeros(len(times1)), alpha=0.7, label = '0 mV Mark', color='cyan')
                plt.legend()
                plt.xlabel('Time (ns)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'Gamma Pulse {counter}')
                plt.show()

        if max_height2 >= min_height_allowed and max_height2 <= max_height_allowed:
            tail_integrals2[counter] = np.trapz(pulse2[index_of_tail_start2 : index_sbzc2 - pregate//2 + (long_gate//2)], times2[index_of_tail_start2 : index_sbzc2 - pregate//2 + (long_gate//2)]) #Takes tail integral from tail start to after pulse body.
            total_integrals2[counter] = np.trapz(pulse2[pulse2_max_index - 3 : index_sbzc2 - pregate//2 + (long_gate//2)], times2[pulse2_max_index - 3 : index_sbzc2 - pregate//2 + (long_gate//2)])              #Takes the total intergral from pulse start to after pulse body.
            index_of_discrimination2.append(counter)
            if  tail_integrals2[counter]/total_integrals2[counter] < 0:
                neutron_negative += 1
                plt.plot(times2, pulse2)
                plt.axvline(x=times2[pulse2_max_index - 3], label = 'Pulse Start', color='g')
                plt.axvline(x=times2[index_of_tail_start2], label = 'Tail Start', color='r')
                plt.axvline(x=times2[index_sbzc2 - pregate//2], label = 'Pulse End', color='y')
                plt.plot(times2, np.zeros(len(times2)), alpha=0.7, label = '0 mV Mark', color='cyan')
                plt.legend()
                plt.xlabel('Time (ns)')
                plt.ylabel('Voltage (mV)')
                plt.title(f'Neutron Pulse {counter}')
                plt.show()
    print(total_integrals1)
    print(total_integrals2)'''


    for counter7, (tail1,total1,tail2,total2) in enumerate(zip(tail_integrals1,total_integrals1,tail_integrals2,total_integrals2)):    #Calculates the tail to total ratio
        if total1 == 0:
            gamma_tail_to_total[counter7] = 0
        else:
            gamma_tail_to_total[counter7] = tail1/total1
        if total2 == 0:
            neutron_tail_to_total[counter7] = 0
        else:
            neutron_tail_to_total[counter7] = tail2/total2
        #print(gamma_tail_to_total[counter])
        #print(neutron_tail_to_total[counter])
        #print(gamma_tail_to_total)

    #Creates a plot of gamma and neutron pulses on one plot.
    '''legend_counter = 0
    for counter, (pulse1,times1,pulse2,times2,g_psd,n_psd) in enumerate(zip(pulse_data1,time_data1,pulse_data2,time_data2,gamma_tail_to_total,neutron_tail_to_total)):
        legend_counter += 1
        height1 = max(pulse1)
        height2 = max(pulse2)
        if g_psd <= 0.17 and n_psd >= 0.25 and height1 >= 400 and height2 >= 400:
            time = np.zeros(100)
            for counter,i in enumerate(time):
                time[counter] = counter*2
            plt.plot(time, pulse1[:100]/height1, label = 'Gamma Pulse', color='b')
            plt.plot(time, pulse2[:100]/height2, label = 'Neutron Pulse', color='g',linestyle='--')
            plt.xlabel('Time (ns)')
            plt.ylabel('Ratio to Max Pulse Height')
            plt.legend()
            plt.show()'''

    #Used to compare with CoMPASS plots.
    spectrum_data = pd.read_csv('ToF11_BaselineHeld_G8138_and_N8125_PSDCut_CompassSpectrums.csv')
    #total_pulses_psd = []
    total_pulses_psd1 = []
    total_pulses_psd2 = []
    data_interval = 1/16383
    psd_values = np.arange(0, 1 + data_interval, data_interval)
    psd_data1 = spectrum_data['Gamma PSD'].to_numpy()
    psd_data2 = spectrum_data['Neutron PSD'].to_numpy()
    #psd_data1 = spectrum_data['LE - PSD'].to_numpy()
    #psd_data2 = spectrum_data['CFD - PSD'].to_numpy()
    for i in range(len(psd_data1)):
        how_many = psd_data1[i]
        if how_many > 0:
            total_pulses_psd1 += [psd_values[i] for j in range(how_many)]
    for i in range(len(psd_data2)):
        how_many = psd_data2[i]
        if how_many > 0:
            total_pulses_psd2 += [psd_values[i] for j in range(how_many)]

    '''pd.DataFrame(gamma_tail_to_total).to_csv('C:/Users/bswellons.AUTH/Documents/ben_gamma_PSD.csv')
    pd.DataFrame(neutron_tail_to_total).to_csv('C:/Users/bswellons.AUTH/Documents/ben_gamma_PSD.csv')'''
    if path.exists(f'{neutron_file[:-5]}{1}.csv'):

        #Plot without error bars
        plt.hist(gamma_tail_to_total, bins = 500, range = (0,1), label = 'Gamma Detector', color='blue', histtype='step')        #Plots histogram of tail to total ratio for each pulse.
        plt.hist(neutron_tail_to_total, bins = 500, range = (0,1), alpha=0.6, label = 'Neutron Detector', color='orange', histtype='step')
        plt.legend(loc='upper right',)
        plt.xlabel('Tail to Total Ratio')
        plt.ylabel('Number of Pulses (Counts)')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.show()
        #plt.hist(total_pulses_psd1, bins = 2000, range = (0,1), alpha=0.7, label = 'Compass PSD Plot', color='cyan')

        #Plot with error bars
        y, bin_edges = np.histogram(gamma_tail_to_total, bins= 200, range = (0, 1))
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 1.5, markeredgewidth = 0, markersize = 0.1, label = 'Gamma Detector', color='blue')
        '''interval = 0.005
        highest = 0
        highest_index = 0
        average_index = 0
        for counter,i in enumerate(y):
            if i >= highest:
                highest = i
                highest_index = counter
            if i >= float(np.mean(y)):
                average_index = counter
        print(f'Number of Pulses at Average Height: {np.mean(y)}, PSD of Average: {average_index*interval}')
        print(f'Number of Pulses at Highest: {highest}, PSD of Highest: {highest_index*interval}')'''
        y, bin_edges = np.histogram(neutron_tail_to_total, bins= 200, range = (0, 1))
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 2, markeredgewidth = 0, markersize = 0.1, label = 'Neutron Detector', color='red', ls = ':')
        '''interval = 0.005
        highest = 0
        highest_index = 0
        average_index = 0
        for counter,i in enumerate(y):
            if i >= highest:
                highest = i
                highest_index = counter
            if i >= float(np.mean(y)):
                average_index = counter
        print(f'Number of Pulses at Average Height: {np.mean(y)}, PSD of Average: {average_index*interval}')
        print(f'Number of Pulses at Highest: {highest}, PSD of Highest: {highest_index*interval}')'''
        plt.ylim(0,500)
        plt.xlim(0, 0.6)
        plt.legend(loc='upper right')
        plt.xlabel('Tail to Total Ratio')
        plt.ylabel('Number of Pulses (Counts)')
        plt.xticks(np.arange(0, 0.7, 0.1))
        plt.show()

    else:
        if f'{gamma_file[0]}' == 'G':
            #plt.hist(gamma_tail_to_total, bins = 500, range = (0,1), label = 'My PSD Plot', color='red')             #Plots histogram of tail to total ratio for each pulse.
            y, bin_edges = np.histogram(total_pulses_psd1, bins= 200, range = (0, 1))
            bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
            plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 1.5, markeredgewidth = 0, markersize = 0.1, label = 'CoMPASS', color='blue')
            y, bin_edges = np.histogram(gamma_tail_to_total, bins= 200, range = (0, 1))
            bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
            plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 2, markeredgewidth = 0, markersize = 0.1, label = 'RadSigPro', color='red', ls = ':')
            #plt.hist(total_pulses_psd1, bins = 500, range = (0,1), alpha=0.7, label = 'Compass PSD Plot', color='blue')
            #plt.hist(total_pulses_psd1, bins = 500, range = (0,1),  label = 'Leading Edge PSD')
            #plt.hist(total_pulses_psd2, bins = 500, range = (0,1), alpha=0.6, label = 'CFD PSD')
            plt.legend(loc='upper right')
            plt.xlabel('Tail to Total Ratio')
            plt.ylabel('Number of Pulses (Counts)')
            plt.xlim(0, 0.6)
            plt.xticks(np.arange(0, 0.7, 0.1))
            #plt.yscale('log')
            plt.show()
        if f'{gamma_file[0]}' == 'N':
            #plt.hist(total_pulses_psd2,, bins = 500, range = (0,1), label = 'My PSD Plot', color='red')             #Plots histogram of tail to total ratio for each pulse.
            y, bin_edges = np.histogram(total_pulses_psd2, bins= 200, range = (0, 1))
            bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
            plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 1.5, markeredgewidth = 0, markersize = 0.1, label = 'CoMPASS', color='blue')
            y, bin_edges = np.histogram(gamma_tail_to_total, bins= 200, range = (0, 1))
            bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
            plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 2, markeredgewidth = 0, markersize = 0.1, label = 'RadSigPro', color='red', ls = ':')
            #plt.hist(total_pulses_psd2, bins = 500, range = (0,1), alpha=0.7, label = 'Compass PSD Plot', color='blue')
            #plt.hist(total_pulses_psd1, bins = 500, range = (0,1),  label = 'Leading Edge PSD')
            #plt.hist(total_pulses_psd2, bins = 500, range = (0,1), alpha=0.6, label = 'CFD PSD')
            plt.legend(loc='upper right')
            plt.xlabel('Tail to Total Ratio')
            plt.ylabel('Number of Pulses (Counts)')
            plt.xlim(0, 0.6)
            plt.xticks(np.arange(0, 0.7, 0.1))
            #plt.yscale('log')
            plt.show()

    if path.exists(f'{gamma_file[:-21]}_PSD_Data{1}.csv'):
        x = 1
    else:
        for i in range(math.ceil(len(index_of_discrimination1)/50000)):
            start_pulse = i*50000
            final_pulse = start_pulse + 50000
            if len(index_of_discrimination1) < final_pulse:
                final_pulse = len(index_of_discrimination1)
            f = open(f'{gamma_file[:-21]}_PSD_Data{i+1}.csv', 'w')
            f.write(",".join([f"time {index_of_discrimination1[k]+1},pulse {index_of_discrimination1[k]+1},tail/total {index_of_discrimination1[k]+1}" for k in range(start_pulse, final_pulse)])+ "\n")
            for j in range(sample_length):
                f.write(",".join([f"{time_data1[index_of_discrimination2[k]][j]},{pulse_data1[index_of_discrimination1[k]][j]},{gamma_tail_to_total[index_of_discrimination1[k]]}" for k in range(start_pulse, final_pulse)]) + "\n")
            f.close()

    if path.exists(f'{neutron_file[:-21]}_PSD_Data{1}.csv'):
        x = 1
    else:
        if path.exists(f'{neutron_file[:-5]}{1}.csv'):
            for i in range(math.ceil(len(index_of_discrimination2)/50000)):
                start_pulse = i*50000
                final_pulse = start_pulse + 50000
                if len(index_of_discrimination2) < final_pulse:
                    final_pulse = len(index_of_discrimination2)
                f = open(f'{neutron_file[:-21]}_PSD_Data{i+1}.csv', 'w')
                f.write(",".join([f"time {index_of_discrimination2[k]+1},pulse {index_of_discrimination2[k]+1},tail/total {index_of_discrimination2[k]+1}" for k in range(start_pulse, final_pulse)])+ "\n")
                for j in range(sample_length):
                    f.write(",".join([f"{time_data2[index_of_discrimination2[k]][j]},{pulse_data2[index_of_discrimination2[k]][j]},{neutron_tail_to_total[index_of_discrimination2[k]]}" for k in range(start_pulse, final_pulse)]) + "\n")
                f.close()

    print(f'Number of Negative Gamma PSD Pulses = {gamma_negative}')
    print(f'Number of Negative Neutron PSD Pulses = {neutron_negative}')
    print(f'Total Number of Pulses = {total_number}')


def PHD_Plot(file, adc_axis_max, baseline):
## This function makes a pulse heigh distribution plot of a raw data file (not a file already run through the pulse correction function)

    mpl.rc('font',family='Times New Roman')
    mpl.rc('font', size = 16)

    num_rows = sum(1 for line in open(file)) - 1

    pulse_heights_ADC = np.zeros(num_rows)
    pulse_heights_mV = np.zeros(num_rows)

    sample_start_index = 0
    with open(file, newline='') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for counter,line in enumerate(csv_reader):
            if counter == 0:
                for counter2,i in enumerate(line):
                    if i == 'SAMPLES':
                        sample_start_index = counter2
            if counter > 0:
                #baseline = max([float(p) for p in line[sample_start_index:len(line)]])
                pulse_heights_ADC[counter-1] = (-float(min(line[sample_start_index:len(line)]))+baseline)
                pulse_heights_mV[counter-1] = (-float(min(line[sample_start_index:len(line)]))+baseline)*0.1220703125

     #Compass comparrison
    spectrum_data = pd.read_csv('ToF11_BaselineHeld_G8138_and_N8125_PSDCut_CompassSpectrums.csv')
    total_pulses_energy = []
    #total_pulses_energy1 = []
    #total_pulses_energy2 = []

    energy_channels = np.arange(0, 16384, 1, dtype=np.float64)
    energy_data = spectrum_data['Gamma Energy'].to_numpy()
    #energy_data1 = spectrum_data['LE - PHD'].to_numpy()
    #energy_data2 = spectrum_data['CFD - PHD'].to_numpy()

    for i in range(len(energy_data)-1):
        how_many = int(energy_data[i])
        if how_many > 0:
            total_pulses_energy += [int(energy_channels[i]) for j in range(how_many)]
    '''for i in range(len(energy_data1)-1):
        how_many = float(energy_data1[i])
        if how_many > 0:
            total_pulses_energy1 += [float(energy_channels[i]) for j in range(how_many)]
    for i in range(len(energy_data2)-1):
        how_many = float(energy_data2[i])
        if how_many > 0:
            total_pulses_energy2 += [float(energy_channels[i]) for j in range(how_many)]'''

    #Plot without error bars
    plt.hist(pulse_heights_ADC, bins = int(adc_axis_max//11), range = (0, adc_axis_max), label = 'My PHD Plot', color='red')
    #plt.hist(pulse_heights_ADC, bins = int(adc_axis_max//11), range = (0, adc_axis_max), color='red', histtype='step')
    #plt.hist(total_pulses_energy, bins = int(adc_axis_max//11), range = (0, adc_axis_max), alpha=0.6, label = 'Compass PHD Plot', color='y')
    #plt.hist(total_pulses_energy1, bins = float(adc_axis_max//11), range = (0, adc_axis_max), label = 'Leading Edge PHD')
    #plt.hist(total_pulses_energy2, bins = float(adc_axis_max//11), range = (0, adc_axis_max), alpha=0.6, label = 'CFD PHD')
    plt.legend(loc='upper right')
    plt.xlabel('ADC Channel')
    plt.ylabel('Number of Pulses (Counts)')
    #plt.ylim((0, 20000))
    #plt.xticks(np.arange(0, adc_axis_max+1, 500))
    plt.show()
    plt.hist(pulse_heights_mV, bins = int(adc_axis_max*0.1220703125//1.15), range = (0, int(adc_axis_max*0.1220703125)))
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Number of Pulses (Counts)')
    #plt.ylim((0, 20000))
    plt.show()

    print(len(total_pulses_energy), len(pulse_heights_ADC))
    print(np.amax(total_pulses_energy), np.amax(pulse_heights_ADC))
    total_energies2 = np.zeros(len(total_pulses_energy))
    for bigg,i in enumerate(total_pulses_energy):
        total_energies2[bigg] = i * float(np.amax(pulse_heights_ADC)/np.amax(total_pulses_energy))

    #Plot with error bars
    '''y, bin_edges = np.histogram(pulse_heights_mV, bins = int(adc_axis_max*0.1220703125//11), range = (0, int(adc_axis_max*0.1220703125)))
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 2, linewidth = 2, markeredgewidth = 0, markersize = 0.1, color='red')'''
    y, bin_edges = np.histogram(total_energies2, bins = int(200), range = (0, 8192))
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 2, linewidth = 2, markeredgewidth = 0, markersize = 0.1, label = 'CoMPASS', color='blue')
    y, bin_edges = np.histogram(pulse_heights_ADC, bins = int(200), range = (0, 8192))
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 2, linewidth = 2, markeredgewidth = 0, markersize = 0.1, label = 'RadSigPro', color='red', ls = ':')
    #plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlim((0,8200))
    #plt.xlabel('Voltage (mV)')
    plt.xlabel('ADC Channel')
    plt.ylabel('Number of Pulses (Counts)')
    plt.show()

    '''my_hist, bin_edges = np.histogram(pulse_heights_ADC, bins = 16383, range = (0, 16383))

    f = open(f'test.csv', 'w')
    for i in my_hist:
        f.write(f'{i}' + "\n")
    f.close()'''


def ToF_Analysis(gamma_file, neutron_file, time_axis_min, time_axis_max, long_gate = 360, pregate = 50, short_gate = 70, cfd_delay = 6, attenuation_fraction = 0.25):
## This function makes a time of flight plot of the neutron times - gamma times, where you select the minimum and maximum points on the x-axis.

    mpl.rc('font',family='Times New Roman')
    mpl.rc('font', size = 16)

    num_total_pulses = sum(1 for line in open(f'{gamma_file[:-21]}.csv')) - 1
    sample_length = (sum(1 for line in open(f'{gamma_file[:-5]}1.csv')) - 1)
    first_file = pd.read_csv(f'{gamma_file[:-5]}1.csv')
    pulses_per_csv = len(first_file.columns)//2

    time_data1 = np.zeros(shape=(num_total_pulses, sample_length))
    pulse_data1 = np.zeros(shape=(num_total_pulses, sample_length))
    time_data2 = np.zeros(shape=(num_total_pulses, sample_length))
    pulse_data2 = np.zeros(shape=(num_total_pulses, sample_length))
    gamma_psd = 0
    neutron_psd = 0

    num_pulses_inserted1 = 0
    for i in range(math.ceil(num_total_pulses/pulses_per_csv)):
        if path.exists(f'{gamma_file[:-5]}{i+1}.csv'):
            data = pd.read_csv(f'{gamma_file[:-5]}{i+1}.csv')
            num_pulses = len(data.columns)//2
            time_data1[num_pulses_inserted1:(num_pulses + num_pulses_inserted1)] = data[[f'time {j+1}' for j in range(num_pulses_inserted1, num_pulses + num_pulses_inserted1)]].to_numpy().T
            pulse_data1[num_pulses_inserted1:(num_pulses + num_pulses_inserted1)] = data[[f'pulse {j+1}' for j in range(num_pulses_inserted1, num_pulses + num_pulses_inserted1)]].to_numpy().T
            num_pulses_inserted1 += num_pulses
        else:
            break
    num_pulses_inserted2 = 0
    for i in range(math.ceil(num_total_pulses/pulses_per_csv)):
        if path.exists(f'{neutron_file[:-5]}{i+1}.csv'):
            data = pd.read_csv(f'{neutron_file[:-5]}{i+1}.csv')
            num_pulses = len(data.columns)//2
            time_data2[num_pulses_inserted2:(num_pulses + num_pulses_inserted2)] = data[[f'time {j+1}' for j in range(num_pulses_inserted2, num_pulses + num_pulses_inserted2)]].to_numpy().T
            pulse_data2[num_pulses_inserted2:(num_pulses + num_pulses_inserted2)] = data[[f'pulse {j+1}' for j in range(num_pulses_inserted2, num_pulses + num_pulses_inserted2)]].to_numpy().T
            num_pulses_inserted2 += num_pulses
        else:
            break


    num_discriminated_gammas = 0
    num_discriminated_neutrons = 0
    if path.exists(f'{gamma_file[:-21]}_PSD_Data{1}.csv') and path.exists(f'{neutron_file[:-21]}_PSD_Data{1}.csv'):
        for i in range(50):
            if path.exists(f'{gamma_file[:-21]}_PSD_Data{i+1}.csv'):
                gamma_data = pd.read_csv(f'{gamma_file[:-21]}_PSD_Data{i+1}.csv')
                num_discriminated_gammas += len(gamma_data.columns)//3
            if path.exists(f'{neutron_file[:-21]}_PSD_Data{i+1}.csv'):
                neutron_data = pd.read_csv(f'{neutron_file[:-21]}_PSD_Data{i+1}.csv')
                num_discriminated_neutrons += len(neutron_data.columns)//3

    gamma_time_data = np.zeros(shape=(num_discriminated_gammas, sample_length))
    gamma_pulse_data = np.zeros(shape=(num_discriminated_gammas, sample_length))
    gamma_psd = np.zeros(num_discriminated_gammas)
    neutron_time_data = np.zeros(shape=(num_discriminated_neutrons, sample_length))
    neutron_pulse_data = np.zeros(shape=(num_discriminated_neutrons, sample_length))
    neutron_psd = np.zeros(num_discriminated_neutrons)

    num_pulses_inserted3 = 0
    if path.exists(f'{gamma_file[:-21]}_PSD_Data{1}.csv') and path.exists(f'{neutron_file[:-21]}_PSD_Data{1}.csv'):
        for i in range(math.ceil(num_discriminated_gammas/50000)):
            if path.exists(f'{gamma_file[:-21]}_PSD_Data{i+1}.csv'):
                gamma_data = pd.read_csv(f'{gamma_file[:-21]}_PSD_Data{i+1}.csv')
                num_pulses = len(gamma_data.columns)//3
                gamma_time_data[num_pulses_inserted3:(num_pulses + num_pulses_inserted3)] = gamma_data[[f'time {j+1}' for j in range(num_pulses_inserted3, num_pulses + num_pulses_inserted3)]].to_numpy().T
                gamma_pulse_data[num_pulses_inserted3:(num_pulses + num_pulses_inserted3)] = gamma_data[[f'pulse {j+1}' for j in range(num_pulses_inserted3, num_pulses + num_pulses_inserted3)]].to_numpy().T
                gamma_psd[num_pulses_inserted3:(num_pulses + num_pulses_inserted3)] = gamma_data[[f'tail/total {j+1}' for j in range(num_pulses_inserted3, num_pulses + num_pulses_inserted3)]].loc[0].to_numpy().T
                num_pulses_inserted3 += num_pulses

    num_pulses_inserted4 = 0
    if path.exists(f'{gamma_file[:-21]}_PSD_Data{1}.csv') and path.exists(f'{neutron_file[:-21]}_PSD_Data{1}.csv'):
        for i in range(math.ceil(num_discriminated_neutrons/50000)):
            if path.exists(f'{neutron_file[:-21]}_PSD_Data{i+1}.csv'):
                neutron_data = pd.read_csv(f'{neutron_file[:-21]}_PSD_Data{i+1}.csv')
                num_pulses = len(neutron_data.columns)//3
                neutron_time_data[num_pulses_inserted4:(num_pulses + num_pulses_inserted4)] = neutron_data[[f'time {j+1}' for j in range(num_pulses_inserted4, num_pulses + num_pulses_inserted4)]].to_numpy().T
                neutron_pulse_data[num_pulses_inserted4:(num_pulses + num_pulses_inserted4)] = neutron_data[[f'pulse {j+1}' for j in range(num_pulses_inserted4, num_pulses + num_pulses_inserted4)]].to_numpy().T
                neutron_psd[num_pulses_inserted4:(num_pulses + num_pulses_inserted4)] = neutron_data[[f'tail/total {j+1}' for j in range(num_pulses_inserted4, num_pulses + num_pulses_inserted4)]].loc[0].to_numpy().T
                num_pulses_inserted4 += num_pulses


    gamma_timetags = np.zeros(num_total_pulses)
    neutron_timetags = np.zeros(num_total_pulses)
    times_of_flight = np.zeros(num_total_pulses)
    gamma_timetags_psd = np.zeros(num_discriminated_gammas)
    neutron_timetags_psd = np.zeros(num_total_pulses)
    if num_discriminated_gammas >= num_discriminated_neutrons:
        times_of_flight_psd = np.zeros(num_discriminated_neutrons)
    else:
        times_of_flight_psd = np.zeros(num_discriminated_gammas)

    inverted_delayed_signal1 = np.zeros(sample_length)
    attenuated_signal1 = np.zeros(sample_length)
    shaped_signal1 = np.zeros(sample_length)
    inverted_delayed_signal2 = np.zeros(sample_length)
    attenuated_signal2 = np.zeros(sample_length)
    shaped_signal2 = np.zeros(sample_length)


    for counter,(pulse1,time1,pulse2,time2) in enumerate(zip(pulse_data1, time_data1, pulse_data2, time_data2)):

        inverted_delayed_signal1[(cfd_delay//2):] = pulse1[:-(cfd_delay//2)]
        attenuated_signal1 = -attenuation_fraction*pulse1
        shaped_signal1 = inverted_delayed_signal1 + attenuated_signal1
        shaped_signal_flipped1 = np.flip(shaped_signal1)
        inverted_delayed_signal2[(cfd_delay//2):] = pulse2[:-(cfd_delay//2)]
        attenuated_signal2 = -attenuation_fraction*pulse2
        shaped_signal2 = inverted_delayed_signal2 + attenuated_signal2
        shaped_signal_flipped2 = np.flip(shaped_signal2)

        for counter1,i in enumerate(shaped_signal_flipped1):
            if i >= max(shaped_signal_flipped1):
                index_max_height1 = counter1
        for counter2,i in enumerate(shaped_signal_flipped1[index_max_height1:]):
            if i <= 0:
                index_sbzc1 = (sample_length - 1) - (counter2 + index_max_height1)
                index_sazc1 = index_sbzc1 + 1
                t_fine1 = (-float(shaped_signal1[index_sbzc1])/(float(shaped_signal1[index_sazc1]) - float(shaped_signal1[index_sbzc1]))) * 2.0
                timetag1 = float(time1[index_sbzc1]) + float(t_fine1)
                break
        for counter3,i in enumerate(shaped_signal_flipped2):
            if i >= max(shaped_signal_flipped2):
                index_max_height2 = counter3
        for counter4,i in enumerate(shaped_signal_flipped2[index_max_height2:]):
            if i <= 0:
                index_sbzc2 = (sample_length - 1) - (counter4 + index_max_height2)
                index_sazc2 = index_sbzc2 + 1
                t_fine2 = (-float(shaped_signal2[index_sbzc2])/(float(shaped_signal2[index_sazc2]) - float(shaped_signal2[index_sbzc2]))) * 2.0
                timetag2 = float(time2[index_sbzc2]) + float(t_fine2)
                break

        gamma_timetags[counter] = timetag1
        neutron_timetags[counter] = timetag2

    for counter, (gamma_timetag,neutron_timetag) in enumerate(zip(gamma_timetags,neutron_timetags)):
        times_of_flight[counter] = neutron_timetag - gamma_timetag


    if path.exists(f'{gamma_file[:-21]}_PSD_Data{1}.csv') and path.exists(f'{neutron_file[:-21]}_PSD_Data{1}.csv'):
        for counter,(pulse1,time1,pulse2,time2) in enumerate(zip(gamma_pulse_data, gamma_time_data, neutron_pulse_data, neutron_time_data)):

            inverted_delayed_signal1[(cfd_delay//2):] = pulse1[:-(cfd_delay//2)]
            attenuated_signal1 = -attenuation_fraction*pulse1
            shaped_signal1 = inverted_delayed_signal1 + attenuated_signal1
            shaped_signal_flipped1 = np.flip(shaped_signal1)
            inverted_delayed_signal2[(cfd_delay//2):] = pulse2[:-(cfd_delay//2)]
            attenuated_signal2 = -attenuation_fraction*pulse2
            shaped_signal2 = inverted_delayed_signal2 + attenuated_signal2
            shaped_signal_flipped2 = np.flip(shaped_signal2)

            for counter1,i in enumerate(shaped_signal_flipped1):
                if i >= max(shaped_signal_flipped1):
                    index_max_height1 = counter1
            for counter2,i in enumerate(shaped_signal_flipped1[index_max_height1:]):
                if i <= 0:
                    index_sbzc1 = (sample_length - 1) - (counter2 + index_max_height1)
                    index_sazc1 = index_sbzc1 + 1
                    t_fine1 = (-float(shaped_signal1[index_sbzc1])/(float(shaped_signal1[index_sazc1]) - float(shaped_signal1[index_sbzc1]))) * 2.0
                    timetag1 = float(time1[index_sbzc1]) + float(t_fine1)
                    break
            for counter3,i in enumerate(shaped_signal_flipped2):
                if i >= max(shaped_signal_flipped2):
                    index_max_height2 = counter3
            for counter4,i in enumerate(shaped_signal_flipped2[index_max_height2:]):
                if i <= 0:
                    index_sbzc2 = (sample_length - 1) - (counter4 + index_max_height2)
                    index_sazc2 = index_sbzc2 + 1
                    t_fine2 = (-float(shaped_signal2[index_sbzc2])/(float(shaped_signal2[index_sazc2]) - float(shaped_signal2[index_sbzc2]))) * 2.0
                    timetag2 = float(time2[index_sbzc2]) + float(t_fine2)
                    break

            gamma_timetags_psd[counter] = timetag1
            neutron_timetags_psd[counter] = timetag2

    for counter, (gamma_timetag,neutron_timetag) in enumerate(zip(gamma_timetags_psd,neutron_timetags_psd)):
        times_of_flight_psd[counter] = neutron_timetag - gamma_timetag


    #plt.hist(times_of_flight, bins = int((abs(time_axis_min)+abs(time_axis_max))*2), range = (int(time_axis_min), int(time_axis_max)))
    plt.hist(times_of_flight, bins = int((abs(time_axis_min)+abs(time_axis_max))*2), range = (int(time_axis_min), int(time_axis_max)), histtype='step')
    #plt.yscale('log')
    plt.xlabel('Time of Flight (ns)')
    plt.title('Total Time of Flight')
    plt.ylabel('Number of Pulses (Counts)')
    plt.show()

    #Plotting PSD histograms after a TOF cut, includes error bars
    tof_cut_gamma_psd = []
    tof_cut_neutron_psd = []
    for counter,(g_psd, n_psd, tof) in enumerate(zip(gamma_psd, neutron_psd, times_of_flight_psd)):
        if tof >= 10 and tof <= 40:
            tof_cut_gamma_psd.append(g_psd)
            tof_cut_neutron_psd.append(n_psd)
    y, bin_edges = np.histogram(tof_cut_gamma_psd, bins= 200, range = (0, 1))
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 1.5, markeredgewidth = 0, markersize = 0.1, label = 'Gamma Detector', color='blue')
    y, bin_edges = np.histogram(tof_cut_neutron_psd, bins= 200, range = (0, 1))
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 2, linewidth = 1.5, markeredgewidth = 0, markersize = 0.1, label = 'Neutron Detector', color='red', ls = ':')
    plt.legend(loc='upper right')
    plt.xlabel('Tail to Total Ratio')
    plt.ylabel('Number of Pulses (Counts)')
    plt.ylim(0,475)
    plt.xlim(0, 0.6)
    plt.xticks(np.arange(0, 0.7, 0.1))
    plt.show()

    #Plot with error bars
    y, bin_edges = np.histogram(times_of_flight, bins= int((abs(time_axis_min)+abs(time_axis_max))), range = (int(time_axis_min), int(time_axis_max)))
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    plt.errorbar(bin_centers, y, yerr = y**0.5, capsize = 2, marker = '.', drawstyle = 'steps-mid', ecolor = 'dimgrey', elinewidth = 1.5, linewidth = 1.5, markeredgewidth = 0, markersize = 0.1, color = 'red')
    #print(y)
    plt.xlabel('Time Cross-Correlation (ns)')
    #plt.title('Total Time of Flight')
    plt.ylabel('Number of Pulses (Counts)')
    plt.ylim(0,300)
    plt.show()

    if path.exists(f'{gamma_file[:-21]}_PSD_Data{1}.csv') and path.exists(f'{neutron_file[:-21]}_PSD_Data{1}.csv'):
        plt.hist(times_of_flight_psd, bins = int((abs(time_axis_min)+abs(time_axis_max))*2), range = (int(time_axis_min), int(time_axis_max)))
        plt.title('Time of Flight from PSD Restriction')
        #plt.yscale('log')
        plt.xlabel('Time of Flight (ns)')
        plt.ylabel('Number of Pulses (Counts)')
        plt.show()

    if path.exists(f'{gamma_file[:-21]}_PSD_Data{1}.csv') or path.exists(f'{neutron_file[:-21]}_PSD_Data{1}.csv'):
        f = open(f'{gamma_file[6:-21]}_ToF_&_PSD.csv', 'w')
        f.write(",".join(['pulse,time of flight,gamma psd,neutron psd'])+ "\n")
        for i in range(len(times_of_flight)):
            if times_of_flight[i] <= 40 and times_of_flight[i] >= 10:
                f.write(",".join([f"{i},{times_of_flight_psd[i]},{gamma_psd[i]},{neutron_psd[i]}"]) + "\n")
        f.close()



def ToF_Comparison(gamma_file, neutron_file, max_time_difference, min_time_difference = 2, show_times = 'No', max_height_cutoff = 10000, min_height_cutoff = 0, long_gate = 360, pregate = 50, short_gate = 70, cfd_delay = 6, attenuation_fraction = 0.25):
## This function compares gamma and neutron pulse pairs, then plots and records the pairs that fit the specified constraints.

    mpl.rc('font',family='Times New Roman')
    mpl.rc('font', size = 16)

    num_total_pulses = sum(1 for line in open(f'{gamma_file[:-21]}.csv')) - 1
    sample_length = (sum(1 for line in open(f'{gamma_file[:-5]}1.csv')) - 1)
    first_file = pd.read_csv(f'{gamma_file[:-5]}1.csv')
    pulses_per_csv = len(first_file.columns)//2

    time_data1 = np.zeros(shape=(num_total_pulses, sample_length))
    pulse_data1 = np.zeros(shape=(num_total_pulses, sample_length))
    time_data2 = np.zeros(shape=(num_total_pulses, sample_length))
    pulse_data2 = np.zeros(shape=(num_total_pulses, sample_length))

    num_pulses_inserted1 = 0
    for i in range(math.ceil(num_total_pulses/pulses_per_csv)):
        if path.exists(f'{gamma_file[:-5]}{i+1}.csv') == True:
            data = pd.read_csv(f'{gamma_file[:-5]}{i+1}.csv')
            num_pulses = len(data.columns)//2
            time_data1[num_pulses_inserted1:(num_pulses + num_pulses_inserted1)] = data[[f'time {j+1}' for j in range(num_pulses_inserted1, num_pulses + num_pulses_inserted1)]].to_numpy().T
            pulse_data1[num_pulses_inserted1:(num_pulses + num_pulses_inserted1)] = data[[f'pulse {j+1}' for j in range(num_pulses_inserted1, num_pulses + num_pulses_inserted1)]].to_numpy().T
            num_pulses_inserted1 += num_pulses
        else:
            break
    num_pulses_inserted2 = 0
    for i in range(math.ceil(num_total_pulses/pulses_per_csv)):
        if path.exists(f'{neutron_file[:-5]}{i+1}.csv') == True:
            data = pd.read_csv(f'{neutron_file[:-5]}{i+1}.csv')
            num_pulses = len(data.columns)//2
            time_data2[num_pulses_inserted2:(num_pulses + num_pulses_inserted2)] = data[[f'time {j+1}' for j in range(num_pulses_inserted2, num_pulses + num_pulses_inserted2)]].to_numpy().T
            pulse_data2[num_pulses_inserted2:(num_pulses + num_pulses_inserted2)] = data[[f'pulse {j+1}' for j in range(num_pulses_inserted2, num_pulses + num_pulses_inserted2)]].to_numpy().T
            num_pulses_inserted2 += num_pulses
        else:
            break

    inverted_delayed_signal1 = np.zeros(sample_length)
    attenuated_signal1 = np.zeros(sample_length)
    shaped_signal1 = np.zeros(sample_length)
    inverted_delayed_signal2 = np.zeros(sample_length)
    attenuated_signal2 = np.zeros(sample_length)
    shaped_signal2 = np.zeros(sample_length)

    index_of_discrimination = []
    for i,(pulse1,time1,pulse2,time2) in enumerate(zip(pulse_data1, time_data1, pulse_data2, time_data2)):

        max_height1 = max(pulse1)
        max_height2 = max(pulse2)

        inverted_delayed_signal1[(cfd_delay//2):] = pulse1[:-(cfd_delay//2)]
        attenuated_signal1 = -attenuation_fraction*pulse1
        shaped_signal1 = inverted_delayed_signal1 + attenuated_signal1
        shaped_signal_flipped1 = np.flip(shaped_signal1)
        inverted_delayed_signal2[(cfd_delay//2):] = pulse2[:-(cfd_delay//2)]
        attenuated_signal2 = -attenuation_fraction*pulse2
        shaped_signal2 = inverted_delayed_signal2 + attenuated_signal2
        shaped_signal_flipped2 = np.flip(shaped_signal2)

        for counter1,i in enumerate(shaped_signal_flipped1):
            if i >= max(shaped_signal_flipped1):
                index_max_height1 = counter1
        for counter2,i in enumerate(shaped_signal_flipped1[index_max_height1:]):
            if i <= 0:
                index_sbzc1 = (sample_length - 1) - (counter2 + index_max_height1)
                index_sazc1 = index_sbzc1 + 1
                t_fine1 = (-float(shaped_signal1[index_sbzc1])/(float(shaped_signal1[index_sazc1]) - float(shaped_signal1[index_sbzc1]))) * 2.0
                timetag1 = index_sbzc1 + t_fine1
                break
        for counter3,i in enumerate(shaped_signal_flipped2):
            if i >= max(shaped_signal_flipped2):
                index_max_height2 = counter3
        for counter4,i in enumerate(shaped_signal_flipped2[index_max_height2:]):
            if i <= 0:
                index_sbzc2 = (sample_length - 1) - (counter4 + index_max_height2)
                index_sazc2 = index_sbzc2 + 1
                t_fine2 = (-float(shaped_signal2[index_sbzc2])/(float(shaped_signal2[index_sazc2]) - float(shaped_signal2[index_sbzc2]))) * 2.0
                timetag2 = index_sbzc2 + t_fine2
                break

        if ((timetag1 + max_time_difference) >= timetag2) and ((timetag1 + min_time_difference) <= timetag2) and (max_height1 <= max_height_cutoff and max_height2 <= max_height_cutoff) and (max_height1 >= min_height_cutoff and max_height2 >= min_height_cutoff):
            plt.plot(time1, pulse1, label = 'Gamma Pulse', color='b')
            plt.plot(time2, pulse2, label = 'Neutron Pulse', color='g',linestyle='--')
            plt.xlabel('Time (ns)')
            plt.ylabel('Voltage (mV)')
            plt.legend()
            plt.show()

            if (show_times == 'Show Times') or (show_times == 'show times') or (show_times == 'Show times') or (show_times == 'Show') or (show_times == 'show') or (show_times == 'Yes') or (show_times == 'yes'):
                print(f'Gamma {i} Arrival Time: {timetag1} ns, Neutron {i} Arrival Time: {timetag2} ns')

            index_of_discrimination.append(i)
        else:
            continue


    f = open(f'{gamma_file[:-21]}_DiscriminatedPulses_MaxTime{max_time_difference}_MinTime{min_time_difference}_MaxHeight{max_height_cutoff}_MinHeight{min_height_cutoff}.csv', 'w')
    f.write(",".join([f"time {k},pulse {k}" for k in index_of_discrimination])+ "\n")
    for j in range(sample_length):
        f.write(",".join([f"{time_data1[k][j]},{pulse_data1[k][j]}" for k in index_of_discrimination]) + "\n")
    f.close()

    f = open(f'{neutron_file[:-21]}_DiscriminatedPulses_MaxTime{max_time_difference}_MinTime{min_time_difference}_MaxHeight{max_height_cutoff}_MinHeight{min_height_cutoff}.csv', 'w')
    f.write(",".join([f"time {k},pulse {k}" for k in index_of_discrimination])+ "\n")
    for j in range(sample_length):
        f.write(",".join([f"{time_data2[k][j]},{pulse_data2[k][j]}" for k in index_of_discrimination]) + "\n")
    f.close()


#Raw_Pulse_Correction('Detector1_Gammas_5Cs137.csv', 0)
#Raw_Pulse_Correction('Gamma_ToF11_BaselineHeld_G8138_and_N8125_PSDCut.csv', 8138)
#Raw_Pulse_Correction('Neutron_ToF11_BaselineHeld_G8138_and_N8125_PSDCut.csv', 8125)
#PSD_Analysis('Gamma_ToF11_BaselineHeld_G8138_and_N8125_PSDCut_CorrectedPulses1.csv')
#PSD_Analysis('Neutron_ToF11_BaselineHeld_G8138_and_N8125_PSDCut_CorrectedPulses1.csv')
PHD_Plot('Gamma_ToF11_BaselineHeld_G8138_and_N8125_PSDCut.csv', 8192, 8138)
#PHD_Plot('Neutron_ToF11_BaselineHeld_G8138_and_N8125_PSDCut.csv', 8192, 8125)
#ToF_Analysis('Gamma_EC_ToF_G8142_N8127_CorrectedPulses1.csv', 'Neutron_EC_ToF_G8142_N8127_CorrectedPulses1.csv', -10, 55)
#ToF_Comparison('Gamma_WeakCf252_PSDCut2_CorrectedPulses1.csv', 'Neutron_WeakCf252_PSDCut2_CorrectedPulses1.csv', 'No', 20, 10)
