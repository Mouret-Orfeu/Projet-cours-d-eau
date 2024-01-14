## Projet

### Description

Ce projet vise à expliquer la santé biologique des cours d'eau des France à partir des relevés physico chimiques du site naïade.

### Set up

#### Environnement
Le projet a été réalisé sous Linux, ubuntu 22.04, python 3.10.12.
Il n'y a malheureusement pas de liste des librairies utilisées et de leur version.

#### Données
Mettre les données Hydobiologique (besoin que de resultat.csv) dans "Projet-cours-d-eau/raw_data/naiades_export/France_entiere/Biologie(i2M2)"
Et les données physico chimiques dans "Projet-cours-d-eau/raw_data/naiades_export/France_entiere/Physico_chimie/"
Ces données sont sur le lien suivant https://naiades.eaufrance.fr/france-entiere#/.
(i2M2 est une colonne du fichier telechargé en cliquant sur "Hydrobiologie", et les données physico chimiques sont dans le fichier téléchargé en cliquant sur "Physico chimie")

Dand "Projet-cours-d-eau/raw_data/naiades_export/HER" dezipper le fichier (choisir le format shapefile) telechargé en cliquand sur "Hydroecoregion1" sur le lien suivant "https://www.sandre.eaufrance.fr/atlas/srv/fre/catalog.search#/metadata/24284d4e-37fa-47a2-85fe-850b37abe5a7"

Dans "Projet-cours-d-eau/Projet-cours-d-eau/raw_data/naiades_export/France_entiere/Stations" dezipper le fichier telechargé (choisir le format csv) en cliquant sur "StationMesureEauxSurface" sur le lien suivant https://www.sandre.eaufrance.fr/atlas/srv/fre/catalog.search#/metadata/daccef28-6b30-4441-a641-b04afe15e82f

Pour lancer certaines parties de l'exploration des données dans "code_détaillé", il faut aussi mettre les données correspondantes dans "Grand_est", je n'ai pas le lien par contre, mais ça se trouve surement quelque part sur Naïade.

#### Lancement
Le corps de la réalisation du projet est executé par "Préparation.py" puis Dataset_creation.py" puis "LSTM.ipynb".

### Structure du projet

Dans raw_data, il y doit y avoir les données brutes dézippées.
Dans code_détaillé il y a des notebook d'exploration des données et de test qui ont servie à la construction du corps du projet.
cleaned_data contient les données Préparées par "Préparation.py" 
Enfin, LSTM_Dataset_&_tools contient les données prêtes à l'emploi pour LSTM.ipynb, avec les différents outils de manipulatoin de ceux ci.

