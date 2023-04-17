from utils import utils
from keras.models import load_model
import numpy
import os

class predict:
    def __init__(self, model_path, atom_table, input_type, custom_objects):
        self.model = load_model(model_path, custom_objects=custom_objects)
        self.atom_table = atom_table[0]
        self.table_shape = atom_table[1]
        self.input_type = input_type
        self.batch_size = 128
        self.answer_dir = './predict'
        self.predict_paths = []
        self.compss = []
        self.component_vector = []
        self.Tc = []

    def read_data(self):
        for predict_path in self.predict_paths:
            compss= utils.read_data(predict_path, 1)
            self.compss += compss

    def predict(self, answer_name):
        self.read_data()
        self.component_vector = [utils.get_component_vector(self.atom_table, comps) for comps in self.compss]
        if self.input_type == 'Atom_Table':
            x_predict = numpy.reshape(self.component_vector, self.table_shape(self.compss))
        elif self.input_type == 'Seq_to_Seq':
            x_predict = numpy.array(utils.embedding(self.component_vector))
        self.Tc = list(self.model.predict(x_predict, batch_size = min(self.batch_size, len(self.compss))).reshape(-1))
        if not os.path.exists(self.answer_dir):
            os.makedirs(self.answer_dir)
        with open(self.answer_dir +'/'+answer_name+'.txt', 'w') as file:
            for i in range(0, len(self.compss)):
                file.write("{:<60}{:.2f}".format(self.compss[i], self.Tc[i])+'\n')
            file.close()
        return self.compss, self.Tc