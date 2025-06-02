from xgboost import XGBClassifier
from sklearn.utils import class_weight

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
from focal_loss import SparseCategoricalFocalLoss
class WeightedXGBClassifier:
    def __init__(self, class_weight=False, **xgb_params):
        self.class_weight = class_weight
        self.xgb_params = xgb_params
        self.model = XGBClassifier(**xgb_params)

    def fit(self, X, y):
        if self.class_weight:
            sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)
        else:
            sample_weights = None

        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_params(self, deep=True):
        # Include class_weight in params for grid search
        return {'class_weight': self.class_weight, **self.model.get_params(deep)}
    
    def set_params(self, **params):
        # Extract and store class_weight separately
        if 'class_weight' in params:
            self.class_weight = params.pop('class_weight')
        
        self.model.set_params(**params)
        return self
    

class NeuralNet:
    def __init__(self, class_weight=False, **params):
        # Default hyperparameters
        self.default_params = {
            'n_neurons': [12, 8, 6],
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'dropout_rate': 0,
            'l2_reg': 0,
            'batch_size': 32,
            'validation_split': 0.2,
            'patience': 5,
            'monitor': 'val_loss',
            'min_delta': 1e-4,
            'use_batch_norm': True  
        }
        # Update defaults with provided params
        self.params = {**self.default_params, **params}
        self.class_weight = class_weight
        self.model = None
        
    def create_model(self, X, y):
        n_features_in_ = X.shape[1]  # number of input features
        n_classes_ = len(np.unique(y))  # number of unique classes

        model = Sequential()
        # Input layer
        model.add(Input(shape=(n_features_in_,)))

        # Add dense layers dynamically based on n_neurons list
        for neurons in self.params['n_neurons']:
            model.add(Dense(
                neurons, 
                activation='relu',
                kernel_regularizer=regularizers.l2(self.params['l2_reg'])
            ))
            if self.params['use_batch_norm']:
                model.add(BatchNormalization())
            model.add(Dropout(self.params['dropout_rate']))

        # Output layer for multi-class classification
        model.add(Dense(n_classes_, activation='softmax'))

        # Configure optimizer with learning rate
        if self.params['optimizer'].lower() == 'adam':
            optimizer = Adam(learning_rate=self.params['learning_rate'])
        else:
            optimizer = self.params['optimizer']

        # Add class weights if enabled
        if self.class_weight:
            sample_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            loss = SparseCategoricalFocalLoss(gamma=2, class_weight = sample_weights)
        else:
            loss = SparseCategoricalFocalLoss(gamma=2)
            
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        self.model = model
        return self

    def fit(self, X, y, validation_data=None, epochs = 10000, **kwargs):
        if self.model is None:
            self.create_model(X, y)

        
        # Setup early stopping
        callbacks = [
            EarlyStopping(
                monitor=self.params['monitor'],
                patience=self.params['patience'],
                min_delta=self.params['min_delta'],
                restore_best_weights=True,
                start_from_epoch = 0
            )
        ]
        # Handle validation data
        if validation_data is None:
            kwargs['validation_split'] = self.params['validation_split']
        else:
            kwargs['validation_data'] = validation_data
            
        # Add class weights if enabled
        if self.class_weight:
            sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)
            kwargs['sample_weight'] = sample_weights
        
        # Add callbacks and batch size
        kwargs['callbacks'] = callbacks + kwargs.get('callbacks', [])
        kwargs['batch_size'] = kwargs.get('batch_size', self.params['batch_size'])
            
        self.model.fit(X, y, epochs = epochs, **kwargs, verbose = 0)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit first.")
        return np.argmax(self.model.predict(X, verbose = 0), axis=1)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit first.")
        return self.model.predict(X, verbose = 0)
