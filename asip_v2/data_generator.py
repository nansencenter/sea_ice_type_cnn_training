import numpy as np
import json
from tensorflow import keras
from one_hot_encoding_function import one_hot_continous_sod_f, one_hot_continous_hugo


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,shuffle_on_epoch_end, batch_size, dims_input, dims_amsr2, idir_json,
     output_var_name, input_var_names, amsr2_var_names, encoding, prop=None):
        self.dims_input = dims_input
        self.dims_amsr2 = dims_amsr2
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.idir_json = idir_json
        self.input_var_names = input_var_names
        self.output_var_name = output_var_name
        self.amsr2_var_names = amsr2_var_names
        self.shuffle_on_epoch_end = shuffle_on_epoch_end
        self.encoding = encoding
        self.on_epoch_end()
        
        with open(f'{idir_json}/vector_combinations.json') as fichier_json:
            self.list_combi = json.load(fichier_json)['all_work_comb']


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        self.data_generation()

        return [self.X,self.z], self.y

    def x_y_z_initialization(self):
        # Initialization
        self.X = np.empty((self.batch_size, *self.dims_input))
        self.y = np.empty((self.batch_size, *self.dims_output))
        self.z = np.empty((self.batch_size, *self.dims_amsr2))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle_on_epoch_end:
            np.random.shuffle(self.indexes)

    def data_generation(self):
        raise NotImplementedError('The data_generation() method was not implemented')
        
    def one_hot_continous(self, vector_param):
        """
        """
        raise NotImplementedError('The one_hot_continous() method was not implemented')

class DataGeneratorFrom_npz_File(DataGenerator):

    
    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        
        one_hot_func = {
                'hugo': one_hot_continous_hugo,
                'sod_f': one_hot_continous_sod_f
            }[self.encoding]
        
        self.dims_output ={
                'hugo' : np.array([0,0,0,0]).shape,
                'sod_f' : np.array(self.list_combi).shape
        }[self.encoding]
        
        self.x_y_z_initialization()
        
        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
            vector_param = np.load(ID).get(self.output_var_name)
#             output = self.one_hot_continous(vector_param)
            output = one_hot_func(vector_param, self.list_combi)
            self.y[i,:] = output
            for j, sar_name in enumerate(self.input_var_names):
                self.X[i,:,:,j] = np.load(ID).get(sar_name)
            for j, amsr2_name in enumerate(self.amsr2_var_names):
                self.z[i,:,:,j] = np.load(ID).get(amsr2_name)

                
                
                
                
class HugoDataGenerator(DataGenerator):
    
    def ice_type(self, stage):
        """
        Gives back the index the concentration or 1 should be on.
        Each index corresponds to a particular ice type
        (0: Young ice; 1: First Year ice; 2: Multi year ice ; 3: Ice free).
        The values on which depends this classification are described ine the
        ASIP-v2 manual.
        Parameters
        ----------
        stage : integer
            stage of development.
        Returns
        -------
        index_ : integer
            index of the list where the value (0/1 or concentration) will be.
        """
        index_= None
        if stage in range (0, 83):
            #print('ice_free')
            index_ = 0
        if stage in range(83, 86):
            #print('Young ice')
            index_=1
        if stage in range(87, 94):
            #print('First year ice')
            index_=2
        if stage in range(95, 98):
            #print('multiyear ice')
            index_=3
        return index_

    def one_hot_continous(self, vector_param):
        """
        Returns the list of one-hot encoded values in terms of concentration
        corresponding to ice types based on concentration and stage of development
        of thickest, second thickest and thrid thickest ice.

        Parameters
        ----------
        vector_param : list
            all parameters in a vector.

        Returns
        -------
        result : list
            List of one-hot encoded (in terms of concentration) values
            corresponding to ice types.
        """
        result = [0, 0, 0, 0]
        vector_param = vector_param.squeeze()
        for ice in range(3): # in a output there are 3 data for the 3 most present ice
            if vector_param[1+ice*3]==(-9): 
                continue
            if vector_param[2+ice*3]==(-9): 
                continue
            icetype = ice_type(vector_param[2+ice*3])
            result[icetype] += vector_param[1+ice*3]/100
        if max(result) == 0:
             result[0] = 1
        else:
             result[0] = 1-sum(result[1:])
        return result
    
    

class DataGeneratorFromMemory(DataGenerator):
    def __init__(self,list_IDs,**kwargs):
        super().__init__(list_IDs,**kwargs)
        self.prop = kwargs.pop('prop')

    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        self.x_y_z_initialization()

        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
            self.y[i,:,:,:] = (
                ans for x, ans in zip(self.prop["_locs"], self.prop[self.output_var_name]) if x==ID
                              ).__next__()

            for j, sar_name in enumerate(self.input_var_names):
                self.X[i,:,:,j] = (
                    ans for x, ans in zip(self.prop["_locs"], self.prop[sar_name]) if x==ID
                                  ).__next__()

            for j, amsr2_name in enumerate(self.amsr2_var_names):
                self.z[i,:,:,j] = (
                    ans for x, ans in zip(self.prop["_locs"], self.prop[amsr2_name]) if x==ID
                                  ).__next__()
