#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neighbors import NearestNeighbors
import numpy as np


# # eventually move this into a module
# class ImageFeatures: 
#     """Process image files and extract features
#     Attributes: 
#         model (obj): model used to extract features
#         image_vector (array): flattened and normalized array of image RGB codes
#         img_feature (array): Pooling layer of model 
#     """
#     def __init__(self, img_path, pic_width, pic_height, model):
        
#         self.model = model
#         img_vec = self.image_process(img_path, pic_width, pic_height)
#         self.feature_extraction(img_vec, model)
            
#     def image_process(self, img_path, pic_width, pic_height):
#         """Get flattened RGB image vector"""

#         img_data = image.load_img(img_path, target_size=(pic_width, pic_height))
#         image_vector = image.img_to_array(img_data)
#         return(image_vector)

#     def feature_extraction(self, img_vector, model):
#         """Extract image feature vector"""
#         img_d = img_vector.copy() #Absolutely essential, otherwise preprocess call will overwrite img_vector
#         img_d = np.expand_dims(img_d, axis=0)
#         img_d = preprocess_input(img_d) #Problem here, must be convention of keras to pass by reference?
#         img_d = imagenet_utils.preprocess_input(img_d)
#         self.img_features = model.predict(img_d)


def nearest_neighbor_image_finder(query_features, num_neighs, database): 
    ''' function to return nearest neighbors for a query
    
    Args: 
        query_features (array): numpy array of extracted image features
        num_neighs (int): number of neighbors to return 
        datbase(df): pandas dataframe containing image database
            
    return: 
        nn_results: indices of nearest_neighbor lookup
        neighbor_pd: pandas dataframe subsetted on returned values from knn search
    '''
    #########################################
    ''' i don't actually want nearest neighbors.... really want an average or latent images and the text, but need to explicitly start to model that.
    '''
    #############################################
    #Filter Database based on given price range
    # if price_filt:
    #     database  = database[(database['price']< max_price) & (database['price']> min_price)]
    
    #logic here in case the n exceeds number of items in database
    size_filter_frame = database.shape[0]
    num_neighs = min(num_neighs, size_filter_frame)
    
    #Fit Nearest Neighbor Model
    database_features = np.vstack(database['image_features'])
    
    neighs = NearestNeighbors(n_neighbors=num_neighs) 
    neighs.fit(database_features)
    distance, nn_index = neighs.kneighbors(query_features, return_distance=True)
    
    neighbor_pd = database.iloc[nn_index.tolist()[0]].copy()
    
    return(nn_index, neighbor_pd, distance.tolist()[0]) 
    
       

    
            
