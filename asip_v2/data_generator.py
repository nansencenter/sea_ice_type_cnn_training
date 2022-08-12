import numpy as np
import json
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, shuffle_on_epoch_end, batch_size, dims_amsr2, idir_json,
     output_var_name, input_var_names, amsr2_var_names, prop=None):
        self.dims_amsr2 = dims_amsr2
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.idir_json = idir_json
        self.input_var_names = input_var_names
        self.output_var_name = output_var_name
        self.amsr2_var_names = amsr2_var_names
        self.shuffle_on_epoch_end = shuffle_on_epoch_end
        self.on_epoch_end()

        with open(f'{idir_json}/vector_combinations.json') as fichier_json:
            self.list_combi = sorted(json.load(fichier_json)['all_work_comb'])
            

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle_on_epoch_end:
            np.random.shuffle(self.indexes)

    def data_generation(self):
        raise NotImplementedError('The data_generation() method was not implemented')

    def one_hot_continous(self, vector_param):
        raise NotImplementedError('The one_hot_continous() method was not implemented')

    def convert (self, array):
        """
        Do nothing for HugoDataGenerator and Datageneratorsod_f
        """
        return array

class HugoDataGenerator(DataGenerator):
    def __init__(self, list_IDs, **kwargs):
        super().__init__(list_IDs, **kwargs)
        self.dims_output = np.array([0,0,0,0]).shape
        self.dims_input = (50,50,2)

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
        if stage in range (0, 80):
            #print('water')
            index_ = 0
        if stage in range(83, 86):
            #print('Young ice')
            index_=1
        if stage in range(86, 94):
            #print('First year ice')
            index_=2
        if stage in range(95, 98):
            #print('multiyear ice')
            index_=3
        if stage in range (80,83):
            index_ = 4
        if stage in range (98,100):
            index_ = 4
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
        vector_param = vector_param.squeeze()
        result = [0, 0, 0, 0, 0]
        if vector_param[0] < 10 : #it is water
            result[0] = 1
            return result[:4]
        for ice in range(3): # in a output there are 3 data for the 3 most present ice
            if vector_param[1+ice*3]==(-9):
                continue
            if vector_param[2+ice*3]==(-9):
                continue
            icetype = self.ice_type(vector_param[2+ice*3])
            result[icetype] += round(vector_param[1+ice*3]/100,1)
        if max(result) == 0:
            icetype = self.ice_type(vector_param[2])
            result[icetype] += round(vector_param[0]/100,1)
            result[0] = 1-sum(result[1:])
        else:
            result[0] = 1-sum(result[1:])
        result = self.convert(result)
        result = result[:4]
        return np.round(result,1)

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        self.data_generation()
        return self.X, self.y

    def x_y_z_initialization(self):
        # Initialization
        self.X = np.empty((self.batch_size, *self.dims_input))
        self.y = np.empty((self.batch_size, *self.dims_output))

    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)

        self.x_y_z_initialization()
        # Generate data
        c=0
        for i, ID in enumerate(self.list_IDs_temp):
            vector_param = np.load(ID).get(self.output_var_name)            
            output = self.one_hot_continous(vector_param)
           
            if sum(output) == 0 :
                self.y = np.delete(self.y,(i-c), axis = 0)
                self.X = np.delete(self.X,(i-c),axis = 0)
                c=c+1
            else:
                self.y[(i-c),:] = output
                for j, sar_name in enumerate(self.input_var_names):
                    self.X[(i-c),:,:,j] = np.load(ID).get(sar_name)[:,:,0]

                    
class HugoSarDataGenerator(HugoDataGenerator):
    def __init__(self, list_IDs, **kwargs):
        super().__init__(list_IDs, **kwargs)
        self.dims_input = (50,50,4)
        
    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        self.x_y_z_initialization()
        # Generate data
        c=0
        for i, ID in enumerate(self.list_IDs_temp):
            vector_param = np.load(ID).get(self.output_var_name)            
            output = self.one_hot_continous(vector_param)
            if sum(output) == 0 :
                self.y = np.delete(self.y,(i-c), axis = 0)
                self.X = np.delete(self.X,(i-c),axis = 0)
                c=c+1
            else:
                self.y[(i-c),:] = output
                for j, sar_name in enumerate(self.input_var_names):
                    for k in range (2):
                        self.X[(i-c),:,:,(j*2+k)] = np.load(ID).get(sar_name)[:,:,k]


        
class HugoAMRS2DataGenerator(HugoSarDataGenerator):
    def __init__(self, list_IDs, **kwargs):
        super().__init__(list_IDs, **kwargs)
        
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        self.data_generation()
        return [self.X, self.z], self.y

    def x_y_z_initialization(self):
        # Initialization
        self.X = np.empty((self.batch_size, *self.dims_input))
        self.y = np.empty((self.batch_size, *self.dims_output))
        self.z = np.empty((self.batch_size, *self.dims_amsr2))
              
    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)     
        self.x_y_z_initialization()
        # Generate data
        c=0
        for i, ID in enumerate(self.list_IDs_temp):
            batch = {}
            batch.update(np.load(ID))
            vector_param = batch.get(self.output_var_name)            
            output = self.one_hot_continous(vector_param)
           
            if sum(output) == 0 :
                self.y = np.delete(self.y,(i-c), axis = 0)
                self.X = np.delete(self.X,(i-c),axis = 0)
                c=c+1
            else:
                self.y[(i-c),:] = output
                for j, sar_name in enumerate(self.input_var_names):
                    sar = batch.get(sar_name)
                    for k in range (2):
                        self.X[(i-c),:,:,(j*2+k)] =sar[:,:,k]    
                for j, amsr2_name in enumerate(self.amsr2_var_names):
                    with open(f'{self.idir_json}/stats_amsr2_test.json') as fichier_json:
                        stats = json.load(fichier_json).get(amsr2_name)
                    self.z[(i-c),:,:,j] = (batch.get(amsr2_name)-stats[0])/stats[1]
        
        
class HugoBinaryGenerator(HugoDataGenerator):
    def __init__(self, list_IDs, **kwargs):
        super().__init__(list_IDs, **kwargs)

    def convert (self, vector):
        """
        Convert the continous vector in binary vector
        """
        r = [0, 0, 0, 0, 0]
        if vector[0] > 0.70 :
            return [1, 0, 0, 0]
        else :
            index = 1
            max_ = vector[index]
            for i in range (2,len(vector)):
                if vector[i] >= max_ :
                    max_ = vector[i]
                    index = i
            r[index]=1
            if np.argmax(r) == 4 :
                return [0,0,0,0]
        return r[:4]
    

class DataGenerator_sod_f(DataGenerator):
    def __init__(self, list_IDs, **kwargs):
        super().__init__(list_IDs, **kwargs)
        self.dims_input = (50,50,4)
        self.nb_class = 17
        self.dims_output = np.array([0]*self.nb_class).shape
        
    def ice_type(self, combi):
        """
        Transform the combiantion in a index for the one_hot_encodinf function       
        
        Parameters
        ----------
        combi : string
            the combinaison of the sod and form 
        Returns
        -------
        index_ : integer
            index of the list where the value will be.
        """
        index_= -9
        if combi =='0_0':
            index_ = 0
        if combi in ('82_2', '82_3', '82_4', '82_5', '83_3', '83_4'):
            index_ = 1
        if combi =='83_5':
            index_ = 2
        if combi in ('83_6', '87_3', '87_4', '87_5'):
            index_ = 3
        if combi =='87_6':
            index_ = 4
        if combi in ('91_2', '91_3', '91_4'):
            index_ = 5
        if combi =='91_5':
            index_ = 6
        if combi =='91_6':
            index_ = 7
        if combi =='91_7':
            index_ = 8
        if combi =='93_2':
            index_ = 9
        if combi in ('93_3', '93_4', '93_5'):
            index_ = 10
        if combi =='93_6':
            index_ = 11
        if combi =='93_7':
            index_ = 12
        if combi in ('95_3', '95_4'):
            index_ = 13
        if combi =='95_5':
            index_ = 14
        if combi in ('95_6', '95_7'):
            index_ = 15
        if combi in ('96_6', '97_7'):
            index_ = 16
        if index_ == -9:
            index_ = 17
        return index_   
    
    def one_hot_continous(self, vector_param):
        """
        Converts the output parameter vector ([ct,ca,sa,fa,...])
        into a vector that contains the concentration percentages for the combinations.

        Parameters
        ----------
        vector_param : list
            all parameters in a vector.

        Returns
        -------
        result : list
            List of percentage concentrations for each work combination.
        """

        result = [0]*(self.nb_class+1)
        vector_param = vector_param.squeeze()
        if vector_param[0] < 10 : #open weter
            combi = "0_0"
            index_combi = self.ice_type(combi)
            result[index_combi] = 1
        for ice in range(3): # in a output there are 3 data for the 3 most present ice
            if vector_param[1+ice*3] == (-9):
                continue
            if vector_param[2+ice*3] == (-9):
                continue
            if vector_param[3+ice*3] == (-9):
                continue
            combi = str(int(vector_param[2+ice*3])) + '_' + str(int(vector_param[3+ice*3]))
            index_combi = self.ice_type(combi)
            result[index_combi] += vector_param[1+ice*3]/100
        
        if max(result) == 0:
            combi = str(int(vector_param[2])) + '_' + str(int(vector_param[3]))
            index_combi = self.ice_type(combi)
            result[index_combi] += round(vector_param[0]/100,1)
            result[0] = 1-sum(result[1:])
        else:
            result[0] = 1-sum(result[1:])
        result = self.convert(result)
        result = result[:self.nb_class]
        return np.round(result,1)

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

    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)

        self.x_y_z_initialization()
        
        # Generate data
        c = 0
        for i, ID in enumerate(self.list_IDs_temp):
            batch = {}
            batch.update(np.load(ID))
            vector_param = batch.get(self.output_var_name)            
            output = self.one_hot_continous(vector_param)   

            if sum(output) != 1 :
                self.y = np.delete(self.y,(i-c), axis = 0)
                self.X = np.delete(self.X,(i-c), axis = 0)
                self.z = np.delete(self.z,(i-c), axis = 0)
                c=c+1
            else:
                self.y[(i-c),:] = output
                for j, sar_name in enumerate(self.input_var_names):
                    sar = batch.get(sar_name)
                    for k in range (2):
                        self.X[(i-c),:,:,(j*2+k)] = sar[:,:,k]

                for j, amsr2_name in enumerate(self.amsr2_var_names):
                    with open(f'{self.idir_json}/stats_amsr2_test.json') as fichier_json:
                        stats = json.load(fichier_json).get(amsr2_name)
                    self.z[(i-c),:,:,j] = (batch.get(amsr2_name)-stats[0])/stats[1]
