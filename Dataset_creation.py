
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import tensorflow as tf    
tf.compat.v1.disable_v2_behavior() 
import random


# Load Data


# Load the df_param_time_series DataFrame
with open(f"./cleaned_data/all/df_param_all.pkl", 'rb') as file:
    df_param_all = pickle.load(file)

with open("./cleaned_data/all/all_index_list.pkl", 'rb') as file:
    all_index_list = pickle.load(file)

with open("./cleaned_data/all/df_i2M2_all.pkl", 'rb') as file:
    df_i2M2 = pickle.load(file)
    


## Préparation des outils de création de dataset


# On enlève les index avec "Fraction inconnue de l'eau"
for index in all_index_list:
    if "Fraction inconnue de l'eau" in index:
        all_index_list.remove(index)

# On enlève les index dont l'i2M2 est constant
list_station_i2M2_constant = []

# Iterate over each row in the DataFrame
for index, row in df_i2M2.iterrows():
    i2m2_list = row['i2M2']
    # Check if the i2M2 list is constant
    if len(i2m2_list) > 0 and all(val == i2m2_list[0] for val in i2m2_list):
        list_station_i2M2_constant.append(index)


for index in all_index_list:
    if index[1] in list_station_i2M2_constant:
        all_index_list.remove(index)

# Definition d'un dictionnaire d'index du df_param_all par clé de station présente dans i2M2
dict_station_to_list_index_param = {}

for index in all_index_list:
    # Retrieve the station name for the current index
    station = index[1]
    
    # If the station is already in the dictionary, append the index to its list
    if station in dict_station_to_list_index_param:
        dict_station_to_list_index_param[station].append(index)
    else:
        # Otherwise, initialize a new list with the current index
        dict_station_to_list_index_param[station] = [index]

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/index_dict_by_station.pkl', 'wb') as file:
    pickle.dump(dict_station_to_list_index_param, file)


# Definition d'un dictionnaire d'index du df_param_all par clé de paramètres 
dict_param_to_list_index_param = {}

for index in all_index_list:
    # Retrieve the parameter name for the current index
    param = index[3]
    
    # If the parameter is already in the dictionary, append the index to its list
    if param in dict_param_to_list_index_param:
        dict_param_to_list_index_param[param].append(index)
    else:
        # Otherwise, initialize a new list with the current index
        dict_param_to_list_index_param[param] = [index]

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/index_dict_by_parameter.pkl', 'wb') as file:
    pickle.dump(dict_station_to_list_index_param, file)




dict_HER_to_param_time_series = {}

for index in all_index_list:
    # Retrieve the HER name for the current index
    HER = index[0 ]

    # If the HER is already in the dictionary, append the index to its list
    if HER in dict_HER_to_param_time_series:
        dict_HER_to_param_time_series[HER].append(index)
    else:
        # Otherwise, initialize a new list with the current index
        dict_HER_to_param_time_series[HER] = [index]

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/index_dict_by_HER.pkl', 'wb') as file:
    pickle.dump(dict_HER_to_param_time_series, file)


HER_to_HER_id_dict = {HER: random.uniform(-0.5, 2.5) for HER in dict_HER_to_param_time_series}

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/HER_to_HER_id_dict.pkl', 'wb') as file:
    pickle.dump(HER_to_HER_id_dict, file)

HER_id_to_HER_dict = {HER_id: HER for HER, HER_id in HER_to_HER_id_dict.items()}

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/HER_id_to_HER_dict.pkl', 'wb') as file:
    pickle.dump(HER_id_to_HER_dict, file)

param_name = ['Nitrates', 'Ammonium', 'Phosphore total']
values_nitrates = []
values_ammonium = []
values_phosphore_total = []

for index in all_index_list:
    param = df_param_all.loc[index, 'Value'][0]

    if index[3] == param_name[0]:  # 'Nitrates'
        values_nitrates.extend([float(value) for value in param])

    elif index[3] == param_name[1]:  # 'Ammonium'
        values_ammonium.extend([float(value) for value in param])

    elif index[3] == param_name[2]:  # 'Phosphore total'
        values_phosphore_total.extend([float(value) for value in param])

median_nitrates = np.median(values_nitrates)
median_ammonium = np.median(values_ammonium)
median_phosphore_total = np.median(values_phosphore_total)

IQR_nitrates = np.subtract(*np.percentile(values_nitrates, [75, 25]))
IQR_ammonium = np.subtract(*np.percentile(values_ammonium, [75, 25]))
IQR_phosphore_total = np.subtract(*np.percentile(values_phosphore_total, [75, 25]))

median_dict_param = {'Nitrates':median_nitrates, 'Ammonium':median_ammonium, 'Phosphore total':median_phosphore_total, 'date_diff':27}
IQR_dict_param = {'Nitrates':IQR_nitrates, 'Ammonium':IQR_ammonium, 'Phosphore total':IQR_phosphore_total, 'date_diff':49}

#save it in a pickle file
with open('./LSTM_Dataset_&_tools/median_dict_param.pkl', 'wb') as file:
    pickle.dump(median_dict_param, file)

#save it in a pickle file
with open('./LSTM_Dataset_&_tools/IQR_dict_param.pkl', 'wb') as file:
    pickle.dump(IQR_dict_param, file)


feature_name_Amonnium_Eau_brute = [
    "Ammonium_Eau_brute_min",
    "Ammonium_Eau_brute_max",
    "Ammonium_Eau_brute_moy",
    "Ammonium_Eau_brute_var",
    "Ammonium_Eau_brute_diff_date"
]

feature_name_Amonnium_Pa_filtre_centrifugee= [
    "Ammonium_Phase_aqueuse_de_l_eau_filtrée_centrifugée_min",
    "Ammonium_Phase_aqueuse_de_l_eau_filtrée_centrifugée_max",
    "Ammonium_Phase_aqueuse_de_l_eau_filtrée_centrifugée_moy",
    "Ammonium_Phase_aqueuse_de_l_eau_filtrée_centrifugée_var",
    "Ammonium_Phase_aqueuse_de_l_eau_filtrée_centrifugée_diff_date"
]

# !!! ici ne pas changer l'ordre d'ajout des elements !!!
list_of_list_Ammonium_feature_name = [feature_name_Amonnium_Eau_brute, feature_name_Amonnium_Pa_filtre_centrifugee]

feature_name_Nitrates_Eau_brute = [
    "Nitrates_Eau_brute_min",
    "Nitrates_Eau_brute_max",
    "Nitrates_Eau_brute_moy",
    "Nitrates_Eau_brute_var",
    "Nitrates_Eau_brute_diff_date"
]

feature_name_Nitrates_Pa_filtre_centrifugee = [
    "Nitrates_Phase_aqueuse_de_l_eau_filtrée_centrifugée_min",
    "Nitrates_Phase_aqueuse_de_l_eau_filtrée_centrifugée_max",
    "Nitrates_Phase_aqueuse_de_l_eau_filtrée_centrifugée_moy",
    "Nitrates_Phase_aqueuse_de_l_eau_filtrée_centrifugée_var",
    "Nitrates_Phase_aqueuse_de_l_eau_filtrée_centrifugée_diff_date"
]

# !!! ici ne pas changer l'ordre d'ajout des elements !!!
list_of_list_Nitrates_feature_name = [feature_name_Nitrates_Eau_brute, feature_name_Nitrates_Pa_filtre_centrifugee]


feature_name_Phosphore_total_Eau_brute = [
    "Phosphore_total_Eau_brute_min",
    "Phosphore_total_Eau_brute_max",
    "Phosphore_total_Eau_brute_moy",
    "Phosphore_total_Eau_brute_var",
    "Phosphore_total_Eau_brute_diff_date"
]

feature_name_Phosphore_total_Pa_filtre_centrifugee = [
    "Phosphore_total_Phase_aqueuse_de_l_eau_filtrée_centrifugée_min",
    "Phosphore_total_Phase_aqueuse_de_l_eau_filtrée_centrifugée_max",
    "Phosphore_total_Phase_aqueuse_de_l_eau_filtrée_centrifugée_moy",
    "Phosphore_total_Phase_aqueuse_de_l_eau_filtrée_centrifugée_var",
    "Phosphore_total_Phase_aqueuse_de_l_eau_filtrée_centrifugée_diff_date"
]

feature_name_Phosphore_total_M_E_S_brutes = [
    "Phosphore_total_M_E_S_brutes_min",
    "Phosphore_total_M_E_S_brutes_max",
    "Phosphore_total_M_E_S_brutes_moy",
    "Phosphore_total_M_E_S_brutes_var",
    "Phosphore_total_M_E_S_brutes_diff_date"
]

feature_name_Phosphore_total_Particule_2_mm_de_sédiments = [
    "Phosphore_total_Particule_2_mm_de_sédiments_min",
    "Phosphore_total_Particule_2_mm_de_sédiments_max",
    "Phosphore_total_Particule_2_mm_de_sédiments_moy",
    "Phosphore_total_Particule_2_mm_de_sédiments_var",
    "Phosphore_total_Particule_2_mm_de_sédiments_diff_date"
]

feature_name_Phosphore_total_Matière_sèche_de_particules_2_mm = [
    "Phosphore_total_Matière_sèche_de_particules_2_mm_min",
    "Phosphore_total_Matière_sèche_de_particules_2_mm_max",
    "Phosphore_total_Matière_sèche_de_particules_2_mm_moy",
    "Phosphore_total_Matière_sèche_de_particules_2_mm_var",
    "Phosphore_total_Matière_sèche_de_particules_2_mm_diff_date"
]

# !!! ici ne pas changer l'ordre d'ajout des elements !!!
list_of_list_Phos_feature_name = [feature_name_Phosphore_total_Eau_brute, feature_name_Phosphore_total_Pa_filtre_centrifugee, feature_name_Phosphore_total_M_E_S_brutes, feature_name_Phosphore_total_Particule_2_mm_de_sédiments, feature_name_Phosphore_total_Matière_sèche_de_particules_2_mm]

feature_name_Eau_brute = feature_name_Amonnium_Eau_brute + feature_name_Nitrates_Eau_brute + feature_name_Phosphore_total_Eau_brute
feature_name_pa_filtre_centrifugee = feature_name_Amonnium_Pa_filtre_centrifugee + feature_name_Nitrates_Pa_filtre_centrifugee + feature_name_Phosphore_total_Pa_filtre_centrifugee


# !!! ici ne pas échanger l'ordre des lignes !!!
list_fraction = [
    "Eau brute", 
    "Phase aqueuse de l'eau (filtrée, centrifugée...)", 
    'M.E.S. brutes',
    'Particule < 2 mm de sédiments',
    'Matière sèche de particules < 2 mm'  
]



## Pour accéder aux noms du bon nuplet de features features, et leur fraction 


def create_feature_name_combinations(ammonium_list, nitrates_list, phosphorus_list, list_fraction_names):
    feature_combinations = []
    for i, nitrates_features in enumerate(ammonium_list):
        for j,  ammonium_features in enumerate(nitrates_list):
            for k, phosphorus_features in enumerate(phosphorus_list):
                combination = ammonium_features + nitrates_features + phosphorus_features
                fractions = [list_fraction_names[i], list_fraction_names[j], list_fraction_names[k]]
                feature_combinations.append([combination, fractions])

    return feature_combinations

# Create the combinations
feature_combinations = create_feature_name_combinations(list_of_list_Ammonium_feature_name, list_of_list_Nitrates_feature_name, list_of_list_Phos_feature_name, list_fraction)



# ## Define Hyperparameters and global variables


# parameters

list_fraction_param = ["Phase aqueuse de l'eau (filtrée, centrifugée...)", "Phase aqueuse de l'eau (filtrée, centrifugée...)", "Eau brute"]

target_combination = None
for combination, fractions in feature_combinations:
    if fractions == list_fraction_param:
        target_combination = combination
        break

features_name = target_combination + ['her']

# Save features_name
with open('./LSTM_Dataset_&_tools/features_name.pkl', 'wb') as f:
    pickle.dump(features_name, f)

# Save list_fraction_param
with open('./LSTM_Dataset_&_tools/list_fraction_param.pkl', 'wb') as f:
    pickle.dump(list_fraction_param, f)



# Création de Dataset et gestion du scaling


# On définie le max de chaque paramètre pour une future normalisation
param_name = ['Nitrates', 'Ammonium', 'Phosphore total']

max_nitrates = 0
max_ammonium = 0
max_phosphore_total = 0


for index in all_index_list:
    param = df_param_all.loc[index, 'Value'][0]

    if index[3] == param_name[0]:  # Assuming param_name[0] is 'Nitrates'
        for value in param:
            value = float(value)
            if value > max_nitrates:
                max_nitrates = value

    elif index[3] == param_name[1]:  # Assuming param_name[1] is 'Ammonium'
        for value in param:
            value = float(value)
            if value > max_ammonium:
                max_ammonium = value

    elif index[3] == param_name[2]:  # Assuming param_name[2] is 'Phosphore total'
        for value in param:
            value = float(value)
            if value > max_phosphore_total:
                max_phosphore_total = value

max_param_dict = {'Nitrates':max_nitrates, 'Ammonium':max_ammonium, 'Phosphore total':max_phosphore_total, 'date_diff':5005}

# Save max_param_dict
with open('./LSTM_Dataset_&_tools/max_param_dict.pkl', 'wb') as f:
    pickle.dump(max_param_dict, f)


## Pour le scaling on va juste ajuster les valeurs tels que le maximum des paramètres se retrouvent à 1


# Creates the dataset (and scales the data)
def create_dataset(df_param, df_i2M2, list_fraction_param_arg):
    X_train_list = []
    
    Y_train_list = []
    Y_train_scalers = []

    station_list_for_dataset = []
    index_list_for_dataset = []

    # For each station on a une courbe i2M2
    for station in tqdm(dict_station_to_list_index_param, desc="Processing stations"):
        station_list_for_dataset.append(station)

        scaler_y = MinMaxScaler(feature_range=(0, 1))

        i2M2_curve = np.array(df_i2M2.loc[station]['i2M2'])
        i2M2_reshaped = i2M2_curve.reshape(-1, 1)
        i2M2_curve_scaled = scaler_y.fit_transform(i2M2_reshaped)
        i2M2_curve_scaled = np.array(i2M2_curve_scaled, dtype=float)*100

        Y_train_list.append(i2M2_curve_scaled)
        Y_train_scalers.append(scaler_y)

        # Define the order of parameters
        param_order = ['Nitrates', 'Ammonium', 'Phosphore total']

        i2M2_station_dates = df_i2M2.loc[station]['Date']


        X_train_station = []
        
        # Iterating over the dates
        for date in i2M2_station_dates:
            input_X = []

            # On initialise her à None
            her = None
            # Convert the date string to a datetime object if it's not already
            date_datetime = pd.to_datetime(date)

            # Define the date one year earlier
            one_year_earlier = date_datetime - timedelta(days=365)

            # Pour tous les paramètres physico-chimiques avec la bonne fraction
            # Check and add data for each parameter
            param_idc = 0
            for param in param_order:

                current_index = [index for index in dict_station_to_list_index_param[station] if index[3] == param and index[2]== list_fraction_param_arg[param_idc]]
                param_idc += 1  
                
                if len(current_index) > 1:
                    print(f"\033[91mError: Multiple data entries found for parameter {param}, fraction {list_fraction_param_arg[param_idc]} in the station {station}.\033[0m")
                    print(current_index)
                    return
                

                # Si il y a des données sur ce paramètre dans cette station

                # initialiser her à None au cas où current_index est vide
                her = None
                if current_index:
                    index = current_index[0]

                    index_list_for_dataset.append(index)

                    param_time_series_values = np.array(df_param_all.loc[index, 'Value'][0])
                    # param_time_series_values = df_param_all.loc[index, 'Value'][0]

                    param_time_series_dates = df_param_all.loc[index, 'Value'][1]

                    if param == 'Nitrates':
                        param_time_series_values_scaled = (np.array(param_time_series_values, dtype=float) - median_dict_param['Nitrates']) / IQR_nitrates
                    elif param == 'Ammonium':
                        param_time_series_values_scaled = (np.array(param_time_series_values, dtype=float) - median_dict_param['Ammonium']) / IQR_ammonium
                    elif param == 'Phosphore total':
                        param_time_series_values_scaled = (np.array(param_time_series_values, dtype=float) - median_dict_param['Phosphore total']) / IQR_phosphore_total

                    # Filter the values to keep only those within one year before the given date
                    filtered_values = [value for value, d in zip(param_time_series_values_scaled, param_time_series_dates) 
                            if one_year_earlier <= pd.to_datetime(d) <= date_datetime]
                        
                    if len(filtered_values) != 0:
                        min_value = np.min(filtered_values).item()
                        max_value = np.max(filtered_values).item()
                        moy_value = np.mean(filtered_values).item()
                    else:
                        min_value = -10
                        max_value = -10
                        moy_value = -10
                        

                    # Variations
                    sum_of_variations = 0
                    for i in range(1, len(filtered_values)):
                        sum_of_variations += filtered_values[i] - filtered_values[i-1]
                    
                    # Date difference
                    # Convert param_time_series_dates to datetime objects
                    param_dates_datetime = [pd.to_datetime(d) for d in param_time_series_dates]

                    # Filter dates to find those before the given date
                    dates_before_given = [d for d in param_dates_datetime if d < date]

                    # Find the closest date before the given date
                    if dates_before_given:
                        closest_date_before = max(dates_before_given)
                        date_diff = date - closest_date_before
                    else:
                        date_diff = -10

                    her = index[0]

                    # convert date_diff in int (number of days)
                    if date_diff != -10:
                        date_diff = date_diff.days

                        # On normalise date_diff par son maxglobal
                        date_diff = (date_diff - median_dict_param['date_diff']) / IQR_dict_param['date_diff'] 

                    # input_X.extend([date_diff, min_value, max_value, sum_of_variations, moy_value])
                    input_X.append(min_value)
                    input_X.append(max_value)
                    input_X.append(moy_value)
                    input_X.append(sum_of_variations)
                    input_X.append(date_diff)
                    
                else:
                    # Append None values if parameter data is not available
                    input_X.extend([-10] * 5)  # 5 None values for date_diff, min, max, variations, and moy
            if her is not None:
                input_X.append(HER_to_HER_id_dict[her])
            else:
                input_X.append(-10 )
            X_train_station.append(input_X)

        X_train_list.append(X_train_station)
        

    return X_train_list, Y_train_list, Y_train_scalers, station_list_for_dataset, index_list_for_dataset
    
    


# Ça met 50 secondes environ
X_train_list, Y_train, Y_train_scalers, station_list_for_dataset, index_list_for_dataset= create_dataset(df_param_all, df_i2M2, list_fraction_param)

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/X_train_list.pkl', 'wb') as file:
    pickle.dump(X_train_list, file)

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/Y_train.pkl', 'wb') as file:
    pickle.dump(Y_train, file)

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/Y_train_scalers.pkl', 'wb') as file:
    pickle.dump(Y_train_scalers, file)

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/station_list_for_dataset.pkl', 'wb') as file:
    pickle.dump(station_list_for_dataset, file)

# save it in a pickle file
with open('./LSTM_Dataset_&_tools/index_list_for_dataset.pkl', 'wb') as file:
    pickle.dump(index_list_for_dataset, file)




