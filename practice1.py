from fastai.collab import * 
path = untar_data(URLS.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')

