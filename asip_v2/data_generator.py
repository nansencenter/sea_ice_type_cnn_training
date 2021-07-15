import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,shuffle_on_epoch_end, batch_size, dims_input, dims_output,dims_amsr2,
     output_var_name, input_var_names, amsr2_var_names, prop=None):
        self.dims_input = dims_input
        self.dims_output = dims_output
        self.dims_amsr2 = dims_amsr2
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.input_var_names = input_var_names
        self.output_var_name = output_var_name
        self.amsr2_var_names = amsr2_var_names
        self.shuffle_on_epoch_end = shuffle_on_epoch_end
        self.on_epoch_end()

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

class DataGeneratorFrom_npz_File(DataGenerator):

    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        self.x_y_z_initialization()
              
        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
            self.y[i,:,:,:] = np.load(ID).get(self.output_var_name)
            for j, sar_name in enumerate(self.input_var_names):
                self.X[i,:,:,j] = np.load(ID).get(sar_name)
            for j, amsr2_name in enumerate(self.amsr2_var_names):
                self.z[i,:,:,j] = np.load(ID).get(amsr2_name)

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
