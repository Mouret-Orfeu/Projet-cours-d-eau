
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
import os



### Definition des path


Analyse_2007_path = 'raw_data/naiades_export/France_entiere/analyses_2007.csv'
Analyse_2008_path = 'raw_data/naiades_export/France_entiere/analyses_2008.csv'
Analyse_2009_path = 'raw_data/naiades_export/France_entiere/analyses_2009.csv'
Analyse_2010_path = 'raw_data/naiades_export/France_entiere/analyses_2010.csv'
Analyse_2011_path = 'raw_data/naiades_export/France_entiere/analyses_2011.csv'
Analyse_2012_path = 'raw_data/naiades_export/France_entiere/analyses_2012.csv'
Analyse_2013_path = 'raw_data/naiades_export/France_entiere/analyses_2013.csv'
Analyse_2014_path = 'raw_data/naiades_export/France_entiere/analyses_2014.csv'
Analyse_2015_path = 'raw_data/naiades_export/France_entiere/analyses_2015.csv'
Analyse_2016_path = 'raw_data/naiades_export/France_entiere/analyses_2016.csv'
Analyse_2017_path = 'raw_data/naiades_export/France_entiere/analyses_2017.csv'
Analyse_2018_path = 'raw_data/naiades_export/France_entiere/analyses_2018.csv'
Analyse_2019_path = 'raw_data/naiades_export/France_entiere/analyses_2019.csv'
Analyse_2020_path = 'raw_data/naiades_export/France_entiere/analyses_2020.csv'
Analyse_2021_path = 'raw_data/naiades_export/France_entiere/analyses_2021.csv'
Analyse_2022_path = 'raw_data/naiades_export/France_entiere/analyses_2022.csv'

list_path = [Analyse_2007_path, Analyse_2008_path, Analyse_2009_path, Analyse_2010_path, Analyse_2011_path, Analyse_2012_path, Analyse_2013_path, Analyse_2014_path, Analyse_2015_path, Analyse_2016_path, Analyse_2017_path, Analyse_2018_path, Analyse_2019_path, Analyse_2020_path, Analyse_2021_path, Analyse_2022_path]

Stations_path = 'raw_data/naiades_export/France_entiere/Stations/StationMesureEauxSurface.csv'
Bio_path = 'raw_data/naiades_export/France_entiere/Biologie(i2M2)/resultat.csv'


def Preparation(Hydroecoregion):


    # Create directory if it doesn't exist
    saving_directory_path = f"./cleaned_data/{Hydroecoregion}/2022"
    if not os.path.exists(saving_directory_path):
        os.makedirs(saving_directory_path)

    Analyses_df = pd.read_csv(Analyse_2022_path, sep=';', header=0)
    Stations_df = pd.read_csv(Stations_path, sep=',', header=0)
    Bio_df = pd.read_csv(Bio_path, sep=';', header=0)

    ## Traitement des mesures sous le seuil de quantification

    # create a copy of Analyses_df
    Analyses_df_worked = Analyses_df.copy()

    # replace every value of "RsAna" with 0 where MnemoRqParEn= "Résultat < au seuil de quantification"
    Analyses_df_worked.loc[Analyses_df_worked['MnemoRqAna'] == "Résultat < au seuil de quantification", 'RsAna'] = 0




    ### Suppression des stations dont les coordonnées dans Stations.csv ne sont pas dans la bonne projection


    Stations_mauvaise_proj = Stations_df[Stations_df['ProjStationMesureEauxSurface'] != 26]
    Stations_mauvaise_proj = Stations_mauvaise_proj[['LbStationMesureEauxSurface']]
    Stations_mauvaise_proj = Stations_mauvaise_proj.drop_duplicates()

    is_in = Analyses_df_worked['LbStationMesureEauxSurface'].isin(Stations_mauvaise_proj['LbStationMesureEauxSurface'])
    if is_in.any():
        print("There are stations in Analyses_df_worked that are also present in Stations_mauvaise_proj.")
    else:
        print("No stations in Analyses_df_worked are present in Stations_mauvaise_proj.")


    print("nb station before geographical projection cleaning: ", Analyses_df_worked['LbStationMesureEauxSurface'].nunique())
    Analyses_df_worked = Analyses_df_worked[~Analyses_df_worked['LbStationMesureEauxSurface'].isin(Stations_mauvaise_proj['LbStationMesureEauxSurface'])]
    print("nb station after geographical projection cleaning: ",Analyses_df_worked['LbStationMesureEauxSurface'].nunique())



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

    # Now you can work with your GeoDataFrame
    print(gdf_stations.head())


    # Load the shapefile
    gdf_regions = gpd.read_file("/home/orfeu/Documents/cours/3A/Cours_d_eau/Projet-cours-d-eau/Projet-cours-d-eau/data/naiades_export/HER//Hydroecoregion1.shp")

    # Reproject to Lambert-93
    gdf_regions = gdf_regions.to_crs(epsg=2154)

    # View the first few records
    print(gdf_regions.head())

    # Plotting the regions (optional, for visualization)
    gdf_regions.plot()


    # Perform a spatial join
    joined = gpd.sjoin(gdf_stations, gdf_regions, how="inner", op='within')

    Station_with_HER_df = joined[['NomHER1', 'LbStationMesureEauxSurface']]

    Station_with_HER_df = Station_with_HER_df.drop_duplicates(subset='LbStationMesureEauxSurface')

    print("nb stations : ",len(Station_with_HER_df))


    ### Ajouter les HER dans le dataset Analyse


    # Create a new column "HER" in Analyses_df_worked
    Analyses_df_worked['HER'] = np.nan

    # Iterate over each unique value in "LbStationMesureEauxSurface" in Analyses_df_worked
    for station in Analyses_df_worked['LbStationMesureEauxSurface'].unique():
        # Check if the station is in Station_with_HER_df
        if station in Station_with_HER_df['LbStationMesureEauxSurface'].values:
            # Get the corresponding "NomHER1" value from Station_with_HER_df
            her_value = Station_with_HER_df.loc[Station_with_HER_df['LbStationMesureEauxSurface'] == station, 'NomHER1'].values[0]
            # Update the "HER" column in Analyses_df_worked where the values of "LbStationMesureEauxSurface" match
            Analyses_df_worked.loc[Analyses_df_worked['LbStationMesureEauxSurface'] == station, 'HER'] = her_value
        else:
            # If the station is not found in Station_with_HER_df, you can choose to leave it as empty string or handle it differently
            pass

    # Sample 20 random rows for checking
    random_rows = Analyses_df_worked[['HER', 'LbStationMesureEauxSurface']].sample(n=20)
    print(random_rows)





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

    stations_not_in_grouped = Station_with_HER_df[~Station_with_HER_df['LbStationMesureEauxSurface'].isin(i2M2_df.index)]
    station_list = stations_not_in_grouped['LbStationMesureEauxSurface'].tolist()

    print(len(Station_with_HER_df))
    print(len(i2M2_df.index))
    print(len(station_list))



    Analyses_df_worked = Analyses_df_worked[Analyses_df_worked['LbStationMesureEauxSurface'].isin(i2M2_df.index)]
    print("Number of rows in Analyses_df_worked:", Analyses_df_worked.shape[0])
    unique_stations = Analyses_df_worked['LbStationMesureEauxSurface'].unique()
    print(len(unique_stations))




    ### Traitement des mesures invalides


    # Suppression des mesures incertaines puis on supprime la colonne
    Analyses_df_worked = Analyses_df_worked[Analyses_df_worked['LbQualAna'] != 'incertaines']
    Analyses_df_worked = Analyses_df_worked.drop(columns=['LbQualAna'])
    Analyses_df_worked = Analyses_df_worked.drop(columns=['CdQualAna'])



    # Suppression des lignes avec des commentaires (qui indiquent une mesure incorrecte), puis suppression de la colonne commentaire
    Analyses_df_worked = Analyses_df_worked[pd.isna(Analyses_df_worked['CommentairesAna'])]
    Analyses_df_worked = Analyses_df_worked[pd.isna(Analyses_df_worked['ComResultatAna'])]

    Analyses_df_worked = Analyses_df_worked.drop(columns=['CommentairesAna', 'ComResultatAna'])


    ### Traitement des colonnes inutiles


    # On ne garde que les colonnes qui nous interresse
    # Analyses_df_worked.drop(columns=['CdRdd', 'NomRdd', 'CdProducteur', 'NomProducteur', 'CdPreleveur', 'NomPreleveur', 'CdLaboratoire', 'NomLaboratoire'], inplace=True)

    Analyses_df_worked = Analyses_df_worked[['LbFractionAnalysee', 'HeurePrel',  'DatePrel', 'LbLongParamètre', 'RsAna', 'LbStationMesureEauxSurface', 'HER']]


    duplicates = Analyses_df_worked.duplicated()
    similar_rows = Analyses_df_worked[duplicates]
    print(similar_rows)

    # Suppression des lignes dupliquées
    Analyses_df_worked = Analyses_df_worked.drop_duplicates()

        


    ### Corrélation entre paramètres (reprise du code du prof)


    params = Analyses_df.groupby(["CdParametre",'CdUniteMesure','CdFractionAnalysee'])


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

    plt_pp.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(an4.corr(), dtype=bool))  
    sns.heatmap(an4.corr(), annot=False, cmap='BrBG', mask=mask)
    plt_pp.show()  

    param_correlation_matrix = an4.corr()


    ### On va enlever les colonnes qui sont corrélées à plus de 0.8


    # Create a mask to consider only the lower triangle (excluding the diagonal)
    mask = np.tril(np.ones_like(param_correlation_matrix, dtype=bool), k=-1)

    # Find the pairs where correlation is 0.8 or more
    highly_correlated_pairs = [(param1, param2) for param1 in param_correlation_matrix.columns for param2 in param_correlation_matrix.columns if (param_correlation_matrix.loc[param1, param2] >= 0.8) and mask[param_correlation_matrix.columns.get_loc(param1), param_correlation_matrix.columns.get_loc(param2)]]

    print(highly_correlated_pairs)


    # On reformate les paires de corrélation

    transformed_pairs = []

    for pair in highly_correlated_pairs:
        # Split each element of the pair
        first_param, first_unit, first_fraction = pair[0].split(' - ')
        second_param, second_unit, second_fraction = pair[1].split(' - ')

        # Create the new format and add to the list
        transformed_pair = ((first_param, first_fraction), (second_param, second_fraction))
        transformed_pairs.append(transformed_pair)

    print(len(transformed_pairs))



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




    print("Number of rows in Analyses_df_worked:", Analyses_df_worked.shape[0])



    ### Traitement des NaN


    # On enlève les colonnes inutiles qui ont trop de valeurs nan, puis les lignes où il en reste (de toutes façon iln'y a de nan que dans HER)


    #drop lines where there is a nan
    Analyses_df_worked = Analyses_df_worked.dropna()


    print("remaining columns", Analyses_df_worked.columns)




    print("Number of rows in Analyses_df_worked:", Analyses_df_worked.shape[0])
    print("Number of rows in Analyses_df:", Analyses_df.shape[0])



    # On enlève les paramètres physico-chimique peut mesuré (c'est à dire les 80% les moins mesurés (arbitraire))
    # Les moins mesurés, c'est à dire ceux qui apparaissent le moins dans Analyses_df_worked


    grouped = Analyses_df_worked.groupby('LbLongParamètre').size()
    percentile_80 = grouped.quantile(0.8)

    # Now get the LbLongParamètre values that meet or exceed this threshold
    high_count_params = grouped[grouped >= percentile_80].index.tolist()

    # Print the high count LbLongParamètre values
    print(high_count_params)
    print(len(high_count_params))


    # Filter Analyses_df to keep only rows where LbLongParamètre is in high_count_params
    filtered_df_top_param = Analyses_df_worked[Analyses_df_worked['LbLongParamètre'].isin(high_count_params)]

    # Print the filtered dataframe
    # print(filtered_df)


    filtered_df = filtered_df_top_param[filtered_df_top_param['LbLongParamètre'] != "Température de l'Eau"]
    counts = filtered_df.groupby('LbLongParamètre')['RsAna'].count()

    # Now counts contains the count of non-null 'RsAna' values for each 'LbLongParamètre', excluding 'Température de l'Eau'
    data_points = counts.values

    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(data_points, 25)
    Q3 = np.percentile(data_points, 75)
    IQR = Q3 - Q1

    # Determine outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count number of outliers
    outliers = data_points[(data_points < lower_bound) | (data_points > upper_bound)]
    number_of_outliers = len(outliers)

    # Print the number of outliers
    print("Number of outliers:", number_of_outliers)

    fig_boxplot_top_param = go.Figure()

    fig_boxplot_top_param.add_trace(go.Box(y=data_points, name='Non-Null Counts'))

    fig_boxplot_top_param.update_layout(
        title='Boxplot of Non-Null Counts for Each Parameter (Excluding Température de l\'Eau)',
        yaxis=dict(title='Count of Non-Null Values')
    )

    fig_boxplot_top_param.show()




    # On pivote la table pour accéder plus facilement aux paramètres
    Ana_df_worked_pivoted = filtered_df_top_param.pivot_table(values='RsAna',index=['LbFractionAnalysee', 'HeurePrel',  'DatePrel', 'LbStationMesureEauxSurface', 'HER'],columns='LbLongParamètre')



    ### Ajout de l'index "DateTime", fusion et formatage de DateFrel et HeurePrel


    print(Ana_df_worked_pivoted.index.names)


    # Extract 'HeurePrel' and 'DatePrel' from the multi-index
    heures = Ana_df_worked_pivoted.index.get_level_values('HeurePrel')
    dates = Ana_df_worked_pivoted.index.get_level_values('DatePrel')

    # Combine and convert to DateTime
    datetimes = pd.to_datetime(dates + ' ' + heures, format='%Y-%m-%d %H:%M:%S')

    # Get other levels of the multi-index
    other_levels = Ana_df_worked_pivoted.index.droplevel(['HeurePrel', 'DatePrel'])

    # Create a new MultiIndex including the DateTime
    new_index = pd.MultiIndex.from_arrays([datetimes] + [other_levels.get_level_values(i) for i in range(other_levels.nlevels)], names=['DateTime'] + other_levels.names)

    # Set the new combined MultiIndex
    Ana_df_worked_pivoted.index = new_index


    # Check if there is any non-NaN value across the entire DataFrame
    any_non_nan = Ana_df_worked_pivoted.notna().any().any()

    if not any_non_nan:
        # If there are no non-NaN values, print the following message
        print("ERROR: There are no non-NaN values in the pivoted dataframe")




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

    # Define the index where you want to add the item
    # index_to_modify = ("L'EHN À OTTROTT", "Eau brute", "1-(3,4-diClPhyl)-3-M-urée")
    # Add an item to the list at the specified index
    # df_param_time_series.at[index_to_modify, 'Value'].append('TEST')
    # Access and print the content of the cell
    # print(df_param_time_series.loc[index_to_modify, 'Value'])
    # df_param_time_series.head(1)
    # 'your_item' is the item you want to add. Replace it with the actual item you wish to add


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


    ### On va regrouper les times series par station


    print(index_key_list[0])

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



    # Find the key-value pair with the highest length of the value list
    max_length = max(len(value) for value in dict_HER_to_param_time_series.values())
    print(max_length)

        


    # Les time series i2M2 sont dans le dictionnaire i2M2_df (la clé est la station)


    # Les times series de paramètres physico chimiques sont dans le dataframe df_param_time_series (utiliser le dictionnaire dict_HER_to_param_time_series de liste d'index pour y accéder (les clé du dico sont les HER))





