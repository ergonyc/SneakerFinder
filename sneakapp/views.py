from flask import Flask, flash, request, redirect, render_template, send_from_directory
from sneakapp import app
from numpy import isnan
import numpy as np 
import pandas as pd
import json

from .model_utils import nearest_neighbor_image_finder

from werkzeug.utils import secure_filename
import os
#app = Flask(__name__)

####  specific imports
#import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


pwidth = 224
pheight = 224

global graph    
#graph = K.get_default_graph()
graph = tf.compat.v1.get_default_graph()
network_model = MobileNetV2(weights = 'imagenet', include_top = False, input_shape = (pwidth,pheight,3),pooling = 'avg')

#UPLOAD_FOLDER = os.path.join(os.getcwd(),'sneakapp/static/images')
UPLOAD_FOLDER = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath(__file__) )), 'sneakapp/static/images')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# database is 
MEDIA_FOLDER = os.path.join( os.path.dirname( os.path.dirname( os.path.abspath(__file__) )), 'database/img')
#print(MEDIA_FOLDER)
MEDIA_FOLDER =  app.config['MEDIA_FOLDER'] = MEDIA_FOLDER



# img_path = os.path.join( app.config['UPLOAD_FOLDER'], '8046698.1858.jpg')
# #Get Features of uploaded image#
# img_data = image.load_img(img_path, target_size=(pwidth, pheight))
# img_vector = image.img_to_array(img_data)
# img_vector = np.expand_dims(img_vector, axis=0)
# img_vector = preprocess_input(img_vector) #Problem here, must be convention of keras to pass by reference?
# img_vector = imagenet_utils.preprocess_input(img_vector)
# # this should prevent having to calculate the network every time... 
# with graph.as_default():
#         global img_features
#         img_features = network_model.predict(img_vector)




db_base = 'database' #'../database'
#database_path = os.path.join(db_base,'summary_withFeatures_Feb-02-2020.json')
database_path = os.path.join(db_base,'categories_with_features_Feb-05-2020.json')

database_json = json.load(open(database_path))
database_df = pd.DataFrame(database_json)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])



def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/images/<path:filename>')
def image_from_db_file(filename):
    return send_from_directory(MEDIA_FOLDER, filename, as_attachment=True)

@app.route('/uload_img/<path:filename>')
def image_from_upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)




@app.route('/')
def home():
    title = 'Sneaker Finder'
    #copy = '\"Sneaker Finder\" will find some shoes similar to an uploaded example'
    copy = 'Sneaker Finder helps you find shoes that match the aesthetic style of shoes you like. Sneaker Finder uses a deep learning model to provide examples based on your provided source of inspiration.'
    return render_template('index.html',title=title, copy=copy)  # return a string

@app.route('/upload', methods=['POST'])
def upload_file():
    #print('here')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename): 
            global filename
            filename = secure_filename(file.filename)  
            img_path = os.path.join( app.config['UPLOAD_FOLDER'], filename )
            file.save(img_path)

            # Get Features of uploaded image
            # check if we alrey have this file (by name) in our database
            have_file = database_df[database_df.loc[:,'Filename']==filename]
            if (have_file.shape[0]>0):  # we have the features in the database
                img_features = have_file.image_features.tolist()

            else:
                img_data = image.load_img(img_path, target_size=(pwidth, pheight))
                img_vector = image.img_to_array(img_data)
                img_vector = np.expand_dims(img_vector, axis=0)
                img_vector = preprocess_input(img_vector) #Problem here, must be convention of keras to pass by reference?
                img_vector = imagenet_utils.preprocess_input(img_vector)
                img_features = network_model.predict(img_vector)
                #print(type(img_features))
 
            # get 9 nearest neighbors
            n_neighs = 9
            nn_index, neighbors,distance = nearest_neighbor_image_finder(img_features, n_neighs, database_df) 
   
            # fix path to the database...
            neighbors = database_df.iloc[nn_index.tolist()[0]].copy()
            neighbors.loc[:,'db_path'] = neighbors.loc[:,'path'].astype(str).copy()

            #neighbors_db.loc[:,'db_path'] = neigh_path
            #neighbors = neighbors_db # this line is where the "filtering" should occur if we add handles on website 
    
    #image_name = os.path.join('images',filename)

    print(f'saved: {img_path}')
    print(f'<upload path>: {UPLOAD_FOLDER}')
    print(f'image: {filename}')

    npath = neighbors['db_path'][0] 
    print(f'saved: {npath}')
    print(f'<media path>: {MEDIA_FOLDER}')   

    header_copy = 'your example:'
    return render_template('album.html', header_copy = header_copy, image_name = filename, neighbors = neighbors, dist=distance)


@app.route('/contact')
def contact():
    print('contacted')
    return render_template('cover.html')


# # run the app.
# if __name__ == "__main__":
#     # Setting debug to True enables debug output. This line should be
#     # removed before deploying a production app.
#     app.debug = True
#     app.run()
