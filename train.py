from utils import utils
import keras.callbacks as callbacks
import random
import numpy

class train:
    def __init__(self, model, atom_table, input_type):
        self.model = model
        self.atom_table = atom_table[0]
        self.table_shape = atom_table[1]
        self.input_type = input_type
        self.validation_split = 0
        self.batch_size = 128
        self.epochs = 2500
        self.log_dir = './tf_logs'
        self.model_dir = './model'
        self.data_paths = []
        self.component_vector = []
        self.Tc = []

    def read_data(self):
        for data_path in self.data_paths:
            compss, Tc = utils.read_data(data_path, 0)
            component_vector = [utils.get_component_vector(self.atom_table, comps) for comps in compss]
            self.component_vector += component_vector
            self.Tc += Tc

    def train(self, model_name):
        self.read_data()
        data = list(zip(self.component_vector, self.Tc))
        random.shuffle(data)
        self.component_vector[:], self.Tc[:] = zip(*data)
        if self.input_type == 'Atom_Table':
            x_train = numpy.reshape(self.component_vector, self.table_shape(self.component_vector))
        elif self.input_type == 'Seq_to_Seq':
            x_train = numpy.array(utils.embedding(self.component_vector))
        y_train = numpy.reshape(self.Tc, (len(self.Tc),1))
        history = self.model.fit(x_train, y_train, batch_size = self.batch_size, epochs = self.epochs, callbacks=callbacks.TensorBoard(log_dir=self.log_dir, write_images=True), validation_split = self.validation_split, shuffle = True)
        self.model.save(self.model_dir+'/'+model_name+'.h5')
        return history

#print(numpy.reshape(preprocess.get_component_vector(atom_table.OATCNN()[0], 'Ag0.15Pb0.975Sn0.01'), (22, 7)))