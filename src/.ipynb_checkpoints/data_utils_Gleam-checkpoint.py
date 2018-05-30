import csv
import numpy as np
from scipy.interpolate import interp1d


def load_data(path, subjects):
    
    data = []
    sensors_name = ['MPL Magnetic Field',
                    'MPL Rotation Vector',
                    'MPL Linear Acceleration',
                    'MPL Gravity','MPL Gyroscope',
                    'MPL Accelerometer',
                    'LTR-506ALS Light sensor']
    for i in subjects: 
        sub_path = path + '/{}/{}_sensorData.csv'.format(i,i)
        
        unix_time = read_column(sub_path, 0)
        sensor_column = read_column(sub_path, 1)
        value1 = read_column(sub_path, 2)
        value2 = read_column(sub_path, 3)
        value3 = read_column(sub_path, 4)
        
        sensors = {}
        for sensor in sensors_name:
            
            index = index_of_sensor(sensor_column, sensor)
           
            time = []
            x_axis = []
            y_axis = []
            z_axis = []
            for idx in index:
                
                x_axis.append(value1[idx])
                time.append(unix_time[idx])
                
                if value2[idx]!='':
                    y_axis.append(value2[idx])
                if value3[idx]!='':
                    z_axis.append(value3[idx])
         
            sensor_data = {}
            sensor_data ['time'] = np.asarray(time, dtype = "int64")
            sensor_data ['x_axis'] = np.asarray(x_axis, dtype = "float32")
            sensor_data ['y_axis'] = np.asarray(y_axis, dtype = "float32")
            sensor_data ['z_axis'] = np.asarray(z_axis, dtype = "float32")
            
            sensors[sensor] = sensor_data
            
        data.append(sensors)
    return data
     
    
def slide_window(data, window_size, stride, num_dim_expand=0):
    """
    Inputs:
            data: list of numpy array 2dim, includes time series 
            window_size: int, Time series will be devided to the window_size length if they are too long
            stride: int, Time series will be diveded to different ts with this time series 
            num_dim_expand:  
            
    Outputs:
            timeseries_output: list of numpy.array, times series with window_size of 'window_size'
            info: A list which contains number of real information(non padding) for each time series
    """
    
    #Initializing the output variables 
    info = []
    timeseries_output = []
    
    for item in range(len(data)):
        one_timeseries = data[item]    #this is numpy.ndarray 2dim (signal length,dim)
        
        for _ in range (num_dim_expand):
            one_timeseries = np.expand_dims (one_timeseries, -1)
            
        start = 0
        end = window_size
        len_one_timeseries = one_timeseries.shape[0]
        
        while end <= len_one_timeseries:
            
            timeseries_output.append(one_timeseries[start: end])
            info.append(window_size)
            start += stride
            end = start + window_size
            
        if start < len_one_timeseries:
            
            temp = one_timeseries[start: len_one_timeseries]
            temp_pad = np.concatenate ((temp, 
                    np.zeros((start + window_size - len_one_timeseries,) + temp.shape[1:],
                    dtype=temp.dtype)), axis=0)
            
            timeseries_output.append(temp_pad)
            info.append(len_one_timeseries - start)
            
            
    return (info, np.stack(timeseries_output))     
        
     
def pre_process(data, down_factor):
    
    
    info = []        # includes tuples of minimum linux time and downfactor
    list_data = []   #list of numpy.array of subjects
    num_subject = len(data)
    sensors_name = ['MPL Magnetic Field',
                    'MPL Rotation Vector',
                    'MPL Linear Acceleration',
                    'MPL Gravity',
                    'MPL Gyroscope',
                    'MPL Accelerometer',
                    'LTR-506ALS Light sensor']
    
    for i in range (num_subject):
        numpyarray = []
        sensors = data[i]
        min_max = find_min_max(sensors)
        
        for sensor in sensors_name:
            if sensor != 'LTR-506ALS Light sensor':
                interpfunc = interp1d(data[i][sensor]['time'],
                                      data[i][sensor]['x_axis'],
                                      bounds_error = False , 
                                      fill_value = 'extrapolate')
                numpyarray.append(interpfunc(np.arange(min_max[0], min_max[1], down_factor)))
                
                interpfunc = interp1d(data[i][sensor]['time'],
                                      data[i][sensor]['y_axis'],
                                      bounds_error = False ,
                                      fill_value = 'extrapolate')
                numpyarray.append(interpfunc(np.arange(min_max[0], min_max[1], down_factor)))
                
                interpfunc = interp1d(data[i][sensor]['time'],
                                      data[i][sensor]['z_axis'],
                                      bounds_error = False ,
                                      fill_value = 'extrapolate')
                numpyarray.append(interpfunc(np.arange(min_max[0], min_max[1], down_factor)))
                
            else:
                interpfunc = interp1d(data[i][sensor]['time'],
                                      data[i][sensor]['x_axis'],
                                      bounds_error = False ,
                                      fill_value = 'extrapolate')
                numpyarray.append(interpfunc(np.arange(min_max[0], min_max[1], down_factor)))
     
        list_data.append(np.transpose(np.array(numpyarray),(1,0)))
                       
        info.append({'min': min_max[0],
                     'time_shape': np.array(numpyarray).shape[1],
                     'down_factor': down_factor})
     
    return (list_data, info)


def load_groundtruth(path, subjects, info):
       
    gt = []
    for i in range(len(subjects)):
        down_factor = info[i]['down_factor']           
        
        if subjects[i] < 9:
            subject = int('110'+str(subjects[i]+1))
        else:
            subject = int('11'+str(subjects[i]+1))
        
        sub_path = path + '/{}/{}_annotate.csv'.format(subject, subject)
        gt_time = read_column(sub_path, 0)
        gt_act = read_column(sub_path, 1)
        
        temp = np.array(gt_value, dtype= 'int64')
        temp = np.unique(temp)
        temp = (temp - min_) / down_factor
        
        gt.append(temp)
        
    return gt
        
    
def load_groundtruth2(path, subjects, info, uniq_activities):
   # use this for activity recognition models

    gt = []
    for i in range(len(subjects)):
        min_ = info[i]['min']
        time_shape = info[i]['time_shape']
        down_factor = info[i]['down_factor']           
        matrix = np.zeros((time_shape, uniq_activities.shape[0]))
        
        if subjects[i] < 9:
            subject = int('110'+str(subjects[i]+1))
        else:
            subject = int('11'+str(subjects[i]+1))
            
        sub_path = path + '/{}/{}_annotate.csv'.format(subject, subject)
        time_of_change = np.asarray(read_column(sub_path, 0)).astype('int64')
        activities= read_column(sub_path, 1)         
        start_end = read_column(sub_path, 2)
        
        j=0
        while j<len(activities)-1:
            if (activities[j] == activities[j+1]) and (start_end[j] == 'start' and start_end[j+1] == 'stop'):
                
                #print (time_of_change[j]- min_)/down_factor, (time_of_change[j+1]- min_)/down_factor, time_shape
                
                start_time = int(np.true_divide((time_of_change[j] - min_), down_factor))
                end_time = int(np.true_divide((time_of_change[j+1] - min_), down_factor))
                activity_idx = np.where(uniq_activities == activities[j])[0]
                
                matrix[start_time: end_time,activity_idx] = 1
            else:
                #raise "Error in parsing ground truth"
                print "Error in parsing ground truth"
                #print activities[j],activities[j+1]
                #print start_end[j], start_end[j+1]
            j += 2    
        gt.append(matrix)
    return gt
            
            
def get_all_activities(path, subjects):
    
    activities = []
    for i in range(len(subjects)):
        if subjects[i] < 9:
            subject = int('110'+str(subjects[i]+1))
        else:
            subject = int('11'+str(subjects[i]+1))
            
            
        sub_path = path + '/{}/{}_annotate.csv'.format(subject, subject)
        activities += read_column(sub_path, 1)         
    
    #remove activity 'w' whcih has been mistakenly written instead of 'walk'
    activities = np.asarray(activities)
    idx_not_w = np.where(activities != 'w')
    activities = activities[idx_not_w]
    return np.unique(activities)
            
def pre_process_ground_truth(gt_means, length, sigma):
    '''
    gt_means: list on numpy.array 2d
    length: list of int, len of different subject
    
    '''
    gtruth = []
    for i in range(len(gt_means)):
        
        gt = np.zeros((length[i],), dtype = 'float32')
        for j in range(gt_means[i].shape[0]):
            x = np.arange(length[i])
            gauss = np.exp(-0.5 * (x - gt_means[i][j])**2 / sigma**2)
            if gauss.max() > 0:
                gauss = gauss / gauss.max()
            gt = gt + gauss
        gt [gt>1.] = 1.
        gtruth.append(gt)
        
    return gtruth
    
    
# find minimum and maximum time among time_series of one subject
def find_min_max(sensors):
    
    sensors_name = ['MPL Magnetic Field',
                    'MPL Rotation Vector',
                    'MPL Linear Acceleration',
                    'MPL Gravity','MPL Gyroscope',
                    'MPL Accelerometer',
                    'LTR-506ALS Light sensor']
    
    min_value = sensors['MPL Magnetic Field']['time'][0]
    #print 'basemin: ' + str(min_value)
    max_value = 0
    
    for sensor in sensors_name:
        min_index = 0
        max_index = sensors[sensor]['time'].shape[0]-1
            
        if min_value > sensors[sensor]['time'][min_index]:
            min_value = sensors[sensor]['time'][min_index]
        
        if max_value < sensors[sensor]['time'][max_index]:
            max_value = sensors[sensor]['time'][max_index]
    
        
    return (min_value, max_value)    
    
# read and return c_th column of file_name
def read_column(path, c):
    
    ifile = open (path, "Ur")
    reader = csv.reader(ifile)
    
    first_row = True
    cth_column = []
    
    for row in reader:
        if first_row == True:
             first_row = False
                
        else:
            cth_column.append(row[c])
                  
    ifile.close()
    return cth_column
    
 
     
def index_of_sensor(sensor_column, sensor_name):
    
    index = []
    
    for i in range(len(sensor_column)):
        
        if sensor_column[i] == sensor_name:
            index.append(i)
            
    return index
    