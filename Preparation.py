
# 1) Chargementdes données


import pandas as pd
import numpy as np
import plotly.graph_objs as go
import seaborn as sns
from IPython.display import display
import matplotlib as plt
import matplotlib.pyplot as plt_pp
import matplotlib.dates as mdates
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import json
import os
import warnings
import pickle


    



### Definition des path


Analyse_2005_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2005.csv'
Analyse_2006_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2006.csv'
Analyse_2007_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2007.csv'
Analyse_2008_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2008.csv'
Analyse_2009_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2009.csv'
Analyse_2010_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2010.csv'
Analyse_2011_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2011.csv'
Analyse_2012_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2012.csv'
Analyse_2013_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2013.csv'
Analyse_2014_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2014.csv'
Analyse_2015_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2015.csv'
Analyse_2016_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2016.csv'
Analyse_2017_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2017.csv'
Analyse_2018_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2018.csv'
Analyse_2019_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2019.csv'
Analyse_2020_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2020.csv'
Analyse_2021_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2021.csv'
Analyse_2022_path = 'raw_data/naiades_export/France_entiere/Physico_chimie/analyses_2022.csv'
                     

list_path = [Analyse_2007_path, Analyse_2008_path, Analyse_2009_path, Analyse_2010_path, Analyse_2011_path, Analyse_2012_path, Analyse_2013_path, Analyse_2014_path, Analyse_2015_path, Analyse_2016_path, Analyse_2017_path, Analyse_2018_path, Analyse_2019_path, Analyse_2020_path, Analyse_2021_path, Analyse_2022_path]

Stations_path = 'raw_data/naiades_export/France_entiere/Stations/StationMesureEauxSurface.csv'
Bio_path = 'raw_data/naiades_export/France_entiere/Biologie(i2M2)/resultat.csv'


def Preparation(year, Analyse_path):

    print("\033[92m" + year + ": loadind du csv" + "\033[0m")

    # Create directory if it doesn't exist
    saving_directory_path = f"./cleaned_data/{year}"
    if not os.path.exists(saving_directory_path):
        os.makedirs(saving_directory_path)


    # J'ai pas assez de RAM pour lire tout le csv d'un coup, je vais le lire par petit bouts
    # Define the chunk size
    chunk_size = 10000  # Adjust this based on your memory constraints

    # On ne garde que les lignes avec ces paramètres
    filter_values = ['Nitrates', 'Phosphore total', 'Ammonium']

    # Initialize an empty list to store the filtered chunks
    filtered_chunks = []

        # Suppress specific DtypeWarning from pandas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.DtypeWarning)
        # Read the CSV in chunks
        with tqdm(total=2000, desc="Processing CSV") as pbar:
            for chunk in pd.read_csv(Analyse_path, sep=';', header=0, chunksize=chunk_size):
                # Filter the chunk and append to the list
                filtered_chunk = chunk[chunk['LbLongParamètre'].isin(filter_values)]
                filtered_chunks.append(filtered_chunk)

                pbar.update(1)

    # Concatenate the filtered chunks into a single DataFrame
    Analyses_df = pd.concat(filtered_chunks)

    if Analyses_df.empty:
        print("\033[91mERROR: Pas de Mesure du Nitrate ni d'Amonium ni de Phorsphore\033[0m")
        return 
    # else:
    #     num_rows = len(Analyses_df)
    #     print("\033[93m" + f"Number of rows in Analyses_df: {num_rows}" + "\033[0m")

        

    Stations_df = pd.read_csv(Stations_path, sep=',', header=0)
    Bio_df = pd.read_csv(Bio_path, sep=';', header=0)

    ## Traitement des mesures sous le seuil de quantification

    # create a copy of Analyses_df
    Analyses_df_worked = Analyses_df.copy()

    # replace every value of "RsAna" with 0 where MnemoRqParEn= "Résultat < au seuil de quantification"
    Analyses_df_worked.loc[Analyses_df_worked['MnemoRqAna'] == "Résultat < au seuil de quantification", 'RsAna'] = 0


    print("\033[92m" + "Traitement des HER: " + "\033[0m")

    ### Suppression des stations dont les coordonnées dans Stations.csv ne sont pas dans la bonne projection
    Stations_mauvaise_proj = Stations_df[Stations_df['ProjStationMesureEauxSurface'] != 26]
    Stations_mauvaise_proj = Stations_mauvaise_proj[['LbStationMesureEauxSurface']]
    Stations_mauvaise_proj = Stations_mauvaise_proj.drop_duplicates()

    Analyses_df_worked = Analyses_df_worked[~Analyses_df_worked['LbStationMesureEauxSurface'].isin(Stations_mauvaise_proj['LbStationMesureEauxSurface'])]

    # num_rows = len(Analyses_df_worked)
    # print("\033[93m" + f"Number of rows remaining 1: {num_rows}" + "\033[0m")

    ### Traitement des HER (ajout des HER pour chaque station)
    # 
    # (code provenant de HER_determination.ipynb)
    # 


    Stations_df = Stations_df[['LbStationMesureEauxSurface', 'CoordXStationMesureEauxSurface', 'CoordYStationMesureEauxSurface']]

    # Convert DataFrame to GeoDataFrame
    # Assuming 'CoordXStationMesureEauxSurface' is longitude and 'CoordYStationMesureEauxSurface' is latitude
    gdf_stations = gpd.GeoDataFrame(
        Stations_df,
        geometry=gpd.points_from_xy(Stations_df.CoordXStationMesureEauxSurface, Stations_df.CoordYStationMesureEauxSurface)
    )

    # Set the coordinate reference system (CRS) for the GeoDataFrame
    # Lamber-93 EPSG is 2154
    gdf_stations.set_crs(epsg=2154, inplace=True)


    # Load the shapefile
    gdf_regions = gpd.read_file("/home/orfeu/Documents/cours/3A/Cours_d_eau/Projet-cours-d-eau/Projet-cours-d-eau/raw_data/naiades_export/HER//Hydroecoregion1.shp")

    # Reproject to Lambert-93
    gdf_regions = gdf_regions.to_crs(epsg=2154)

    # Plotting the regions (optional, for visualization)
    gdf_regions.plot()

    # Perform a spatial join
    joined = gpd.sjoin(gdf_stations, gdf_regions, how="inner", op='within')

    Station_with_HER_df = joined[['NomHER1', 'LbStationMesureEauxSurface']]

    Station_with_HER_df = Station_with_HER_df.drop_duplicates(subset='LbStationMesureEauxSurface')


    ### Ajouter les HER dans le dataset Analyse

    # Create a new column "HER" in Analyses_df_worked 
    Analyses_df_worked['HER'] = np.nan

        # Compute the total number of unique stations
    total_stations = len(Analyses_df_worked['LbStationMesureEauxSurface'].unique())

    # Iterate over each unique value with a progress bar
    for station in tqdm(Analyses_df_worked['LbStationMesureEauxSurface'].unique(), total=total_stations, desc="Processing Stations"):
        # Check if the station is in Station_with_HER_df
        if station in Station_with_HER_df['LbStationMesureEauxSurface'].values:
            # Get the corresponding "NomHER1" value from Station_with_HER_df
            her_value = Station_with_HER_df.loc[Station_with_HER_df['LbStationMesureEauxSurface'] == station, 'NomHER1'].values[0]
            # Update the "HER" column in Analyses_df_worked where the values of "LbStationMesureEauxSurface" match
            Analyses_df_worked.loc[Analyses_df_worked['LbStationMesureEauxSurface'] == station, 'HER'] = her_value
        else:
            # Handle stations not found in Station_with_HER_df
            pass


    
   

    ### On constitue les time series d'indice i2M2 par station (et en passant on enlève toutes les stations pour lesquels on a pas d'i2M2 dans Analyse_df_worked)
    # 
    # (Code provenant de Preparation_i2M2.ipynb)


    # Clean up the data

    Bio_df = Bio_df[Bio_df['LbLongParametre'] == 'Indice Invertébrés Multimétrique (I2M2)']
    columns_to_keep = ['ResIndiceResultatBiologique', 'LbStationMesureEauxSurface', 'HeureResultat','DateDebutOperationPrelBio']
    Bio_df = Bio_df[columns_to_keep]

    # Traitement des NaN 

    # On met toutes les heures Nan à 00:00:00 (toutes les autres heures sont à 00:00:00 de base)
    # Set NaN values in 'HeureResultat' to midnight ('00:00:00')
    Bio_df['HeureResultat'] = Bio_df['HeureResultat'].fillna('00:00:00')


    # Traitement du temps
    # Concatenate the date and time into a single string
    Bio_df['Date'] = Bio_df['DateDebutOperationPrelBio'] + ' ' + Bio_df['HeureResultat']
    # Convert the concatenated string into a DateTime object
    Bio_df['Date'] = pd.to_datetime(Bio_df['Date'])

    # Drop the 'DateDebutOperationPrelBio' and 'HeureResultat' columns
    Bio_df = Bio_df.drop(columns=['DateDebutOperationPrelBio', 'HeureResultat'])

    ## Détails
    Bio_df.rename(columns={'ResIndiceResultatBiologique': 'i2M2'}, inplace=True)

    ## Réarrangement du Dataset en time series
    # Group by 'LbStationMesureEauxSurface' and aggregate the lists
    i2M2_df = Bio_df.groupby('LbStationMesureEauxSurface').apply(
        lambda x: pd.Series({
            'i2M2': list(x.sort_values(by='Date')['i2M2']),
            'Date': list(x.sort_values(by='Date')['Date'])
        })
    )


    Analyses_df_worked = Analyses_df_worked[Analyses_df_worked['LbStationMesureEauxSurface'].isin(i2M2_df.index)]

    # num_rows = len(Analyses_df_worked)
    # print("\033[93m" + f"Number of rows remaining 2: {num_rows}" + "\033[0m")

    ### Traitement des mesures invalides
    # Suppression des mesures incertaines puis on supprime la colonne
    Analyses_df_worked = Analyses_df_worked[Analyses_df_worked['LbQualAna'] != 'incertaines']
    Analyses_df_worked = Analyses_df_worked.drop(columns=['LbQualAna'])
    Analyses_df_worked = Analyses_df_worked.drop(columns=['CdQualAna'])

    # num_rows = len(Analyses_df_worked)
    # print("\033[93m" + f"Number of rows remaining 3: {num_rows}" + "\033[0m")



    # Suppression des lignes avec des commentaires (qui indiquent une mesure incorrecte), puis suppression de la colonne commentaire
    Analyses_df_worked = Analyses_df_worked[pd.isna(Analyses_df_worked['CommentairesAna'])]
    Analyses_df_worked = Analyses_df_worked[pd.isna(Analyses_df_worked['ComResultatAna'])]

    # num_rows = len(Analyses_df_worked)
    # print("\033[93m" + f"Number of rows remaining 4: {num_rows}" + "\033[0m")

    Analyses_df_worked = Analyses_df_worked.drop(columns=['CommentairesAna', 'ComResultatAna'])


    ### Traitement des colonnes inutiles


    # On ne garde que les colonnes qui nous interresse
    # Analyses_df_worked.drop(columns=['CdRdd', 'NomRdd', 'CdProducteur', 'NomProducteur', 'CdPreleveur', 'NomPreleveur', 'CdLaboratoire', 'NomLaboratoire'], inplace=True)

    Analyses_df_worked = Analyses_df_worked[['LbFractionAnalysee', 'HeurePrel',  'DatePrel', 'LbLongParamètre', 'RsAna', 'LbStationMesureEauxSurface', 'HER']]

    # Suppression des lignes dupliquées
    Analyses_df_worked = Analyses_df_worked.drop_duplicates()

    # num_rows = len(Analyses_df_worked)
    # print("\033[93m" + f"Number of rows remaining 5: {num_rows}" + "\033[0m")

    ### Corrélation entre paramètres (reprise du code du prof)

    param_series = Analyses_df['LbLongParamètre']+' - '+Analyses_df['SymUniteMesure']+ ' - ' + Analyses_df['LbFractionAnalysee']
    analyses_light = Analyses_df[['CdStationMesureEauxSurface','CdPrelevement','RsAna']].copy()
    analyses_light['param'] = param_series


    an2 = analyses_light.pivot_table(values='RsAna',index=['CdStationMesureEauxSurface','CdPrelevement'],columns='param')


    cols=[]
    for col in an2.columns :
        if (an2[col].count()>=481):
            cols.append(col)
    an3=an2[cols]

    remplissage_an3 = (~an3.isnull()).sum(axis=1)

    an4 = an3[remplissage_an3>1] 

    param_correlation_matrix = an4.corr()


    ### On va enlever les colonnes qui sont corrélées à plus de 0.9

    # Create a mask to consider only the lower triangle (excluding the diagonal)
    mask = np.tril(np.ones_like(param_correlation_matrix, dtype=bool), k=-1)

    # Find the pairs where correlation is 0.9 or more
    highly_correlated_pairs = [(param1, param2) for param1 in param_correlation_matrix.columns for param2 in param_correlation_matrix.columns if (param_correlation_matrix.loc[param1, param2] >= 0.9) and mask[param_correlation_matrix.columns.get_loc(param1), param_correlation_matrix.columns.get_loc(param2)]]

    # On reformate les paires de corrélation

    transformed_pairs = []

    for pair in highly_correlated_pairs:
        # Split each element of the pair
        first_param, first_unit, first_fraction = pair[0].split(' - ')
        second_param, second_unit, second_fraction = pair[1].split(' - ')

        # Create the new format and add to the list
        transformed_pair = ((first_param, first_fraction), (second_param, second_fraction))
        transformed_pairs.append(transformed_pair)

    ### On enlève de Analyses_df_worked un élement (fraction, paramètre) de chaque couple de corrélation 


    # Set to keep track of removed elements
    removed_elements = set()

    # Iterate over each transformed pair
    for pair in transformed_pairs:
        first_param, first_fraction = pair[0]

        # Check if the second element of the pair has not been removed
        if pair[1] not in removed_elements:
            # Remove rows matching the first element
            Analyses_df_worked = Analyses_df_worked[~((Analyses_df_worked['LbLongParamètre'] == first_param) & (Analyses_df_worked['LbFractionAnalysee'] == first_fraction))]
            
            # Add the removed element to the set
            removed_elements.add(pair[0])

    ### Traitement des NaN


    # On enlève les colonnes inutiles qui ont trop de valeurs nan, puis les lignes où il en reste (de toutes façon iln'y a de nan que dans HER)


    #drop lines where there is a nan
    Analyses_df_worked = Analyses_df_worked.dropna()

    # num_rows = len( Analyses_df_worked )
    # print("\033[93m" + f"Number of rows remaining 6: {num_rows}" + "\033[0m")

    # On enlève les paramètres physico-chimique peut mesuré (c'est à dire les 80% les moins mesurés (arbitraire))
    # Les moins mesurés, c'est à dire ceux qui apparaissent le moins dans Analyses_df_worked

    # DEBUG
    # print("\n Non-NaN count in 'RsAna' before pivoting:", Analyses_df_worked['RsAna'].notna().sum())
    # print("\n)")
    
    # On pivote la table pour accéder plus facilement aux paramètres
    def take_first(series):
        return series.iloc[0]

    # Pivot the table using the custom aggregation function
    Ana_df_worked_pivoted = Analyses_df_worked.pivot_table(
        values='RsAna',
        index=['LbFractionAnalysee', 'HeurePrel', 'DatePrel', 'LbStationMesureEauxSurface', 'HER'],
        columns='LbLongParamètre',
        aggfunc=take_first  # Use the custom function to take the first value
    )    
    print("\033[92m" + "Pivoting de la table fait" + "\033[0m")

    # DEBUG
    # Check the DataFrame after pivoting
    # print("\n Non-NaN count after pivoting:", Ana_df_worked_pivoted.notna().sum().sum())
    # print("\n)")

    

    ### Ajout de l'index "DateTime", fusion et formatage de DateFrel et HeurePrel

    # Extract 'HeurePrel' and 'DatePrel' from the multi-index
    heures = Ana_df_worked_pivoted.index.get_level_values('HeurePrel')
    dates = Ana_df_worked_pivoted.index.get_level_values('DatePrel')

    # Combine and convert to DateTime
    datetimes = pd.to_datetime(dates + ' ' + heures, format='%Y-%m-%d %H:%M:%S')

    # DEBUG
    # Check DateTime conversion
    # print("\n DateTime conversion check:", datetimes.notna().all())
    # print("\n)")

    # Get other levels of the multi-index
    other_levels = Ana_df_worked_pivoted.index.droplevel(['HeurePrel', 'DatePrel'])

    # Create a new MultiIndex including the DateTime
    new_index = pd.MultiIndex.from_arrays([datetimes] + [other_levels.get_level_values(i) for i in range(other_levels.nlevels)], names=['DateTime'] + other_levels.names)

    # Set the new combined MultiIndex
    Ana_df_worked_pivoted.index = new_index


    # # Check if there is any non-NaN value across the entire DataFrame
    # any_non_nan = Ana_df_worked_pivoted.notna().any().any()

    # if not any_non_nan:
    #     # If there are no non-NaN values, print the following message
    #     print("\033[91mERROR: There are no non-NaN values in the pivoted dataframe\033[0m")
    #     return



    DateTime = Ana_df_worked_pivoted.index.get_level_values('DateTime').unique()
    Fraction = Ana_df_worked_pivoted.index.get_level_values('LbFractionAnalysee').unique()
    Station = Ana_df_worked_pivoted.index.get_level_values('LbStationMesureEauxSurface').unique()
    HER = Ana_df_worked_pivoted.index.get_level_values('HER').unique()
    Param = Ana_df_worked_pivoted.columns.tolist()


    ## Création du dataframe des séries temporelles des paramètres physico chimiques par Station et par Fraction


    # Je créer un multi index dataframe où pour chaque tuple (Station, Fraction, Param) je vais mettre la liste (time serie) correspondante

    # Create a MultiIndex
    multi_index = pd.MultiIndex.from_product([HER, Station, Fraction, Param], names=['HER','Station', 'Fraction', 'Param'])

    # Create the DataFrame with None values
    df_param_time_series = pd.DataFrame(index=multi_index, columns=['Value'])

    # Replace None with empty lists
    df_param_time_series['Value'] = df_param_time_series['Value'].apply(lambda x: [[],[]])

    data_for_df = []
    non_empty_param_time_series = []

    datetime = None
    value = None

    # Sort the DataFrame by the 'DateTime' level of the MultiIndex
    # On va parcourir le dataframe par ordre chronologique, ce qui va se traduire par des times series dans l'ordre chronologique dans le nouveau dataframe
    Ana_df_worked_pivoted.sort_index(level='DateTime', inplace=True)

    # Iterate over the rows of the DataFrame
    for row in Ana_df_worked_pivoted.itertuples():
        # Extract relevant information from the row
        datetime, fraction, station, her = row.Index

        for i, param in enumerate(Param, start=1):  # start=1 because index 0 is the row index
            value = row[i]

            if pd.notna(value):
                appending_index = (her, station, fraction, param)
                non_empty_param_time_series.append(appending_index)
                
                data_for_df.append((appending_index, [value, datetime]))
                

    # Now outside the loop, create or update the DataFrame
    for appending_index, values in tqdm(data_for_df, desc="Processing Stations"):
        # df_param_time_series.at[appending_index, 'Value'].extend(values) 

        df_param_time_series.at[appending_index, 'Value'][0].append(values[0])
        df_param_time_series.at[appending_index, 'Value'][1].append(values[1]) 

    index_key_list = list(set(tuple(x) for x in non_empty_param_time_series)) 



    ### On enlève les times series trop courtes (de taille inferieur ou égales à 2)


    index_key_list = [index for index in index_key_list if len(df_param_time_series.loc[index, 'Value'][0]) >= 3]

    print("\033[92m" + "Création des dataframes fait" + "\033[0m")

    ### On va regrouper les times series par station

    dict_station_to_param_time_series = {}

    for index in index_key_list:
        # Retrieve the station name for the current index
        station = index[1]

        # If the station is already in the dictionary, append the index to its list
        if station in dict_station_to_param_time_series:
            dict_station_to_param_time_series[station].append(index)
        else:
            # Otherwise, initialize a new list with the current index
            dict_station_to_param_time_series[station] = [index]


    ### On va regrouper les times series (leur index) par HER
    dict_HER_to_param_time_series = {}

    for index in index_key_list:
        # Retrieve the HER name for the current index
        HER = index[0]

        # If the HER is already in the dictionary, append the index to its list
        if HER in dict_HER_to_param_time_series:
            dict_HER_to_param_time_series[HER].append(index)
        else:
            # Otherwise, initialize a new list with the current index
            dict_HER_to_param_time_series[HER] = [index]


    # On sauvegarde les dataframes préparés 

    # Save the i2M2_df DataFrame using pickle
    with open(f"./cleaned_data/{year}/i2M2_df.pkl", 'wb') as file:
        pickle.dump(i2M2_df, file)

    # Save the df_param_time_series DataFrame using pickle
    with open(f"./cleaned_data/{year}/df_param_time_series.pkl", 'wb') as file:
        pickle.dump(df_param_time_series, file)


    
    

Preparation('2007', Analyse_2007_path)



