import model
from train import train
from predict import predict
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class run:
    def __init__(self):
        self.metric = [
            model.metrics.Tc_accuracy, 
            model.metrics.superconductor_accuracy, 
            model.metrics.gen_performance
        ]
        self.model = model.models.ATCNN_model(self.metric)                  #
        self.atom_table = model.atom_table.ATCNN()                          #
        self.input_type = 'Atom_Table'                                          #

        self.validation_split = 0.02                                                #
        self.batch_size = 128                                                       #
        self.epochs = 2500                                                          #

        self.log_dir = './logs/ATCNN'                                   #           
        self.model_dir = './model'  
        self.model_name = 'ATCNN1'                                           #
        self.data_paths = [                                                         #
            "./data/superconductor.txt", 
            "./data/insulator.txt"
            #"./data/test.txt" #80+50=130
        ]

        self.custom_objects = {
            "Tc_accuracy" : model.metrics.Tc_accuracy, 
            "superconductor_accuracy" : model.metrics.superconductor_accuracy, 
            "gen_performance" : model.metrics.gen_performance,

            "Positional_Encoding" : model.customlayers.Positional_Encoding,
            "Scaled_Dot_Product_Attention" : model.customlayers.Scaled_Dot_Product_Attention,
            "Multi_Head_Attention" : model.customlayers.Multi_Head_Attention,

            "Conv2D_Norm_Act" : model.wraplayers.Conv2D_Norm_Act,
            "Identity_Block" : model.wraplayers.Identity_Block,
            "Conv_Block" : model.wraplayers.Conv_Block,
            "Encoder" : model.wraplayers.Encoder,
            "Decoder" : model.wraplayers.Decoder,
            "Transform" : model.wraplayers.Transform
        }

        self.answer_dir = './predict'
        self.answer_name = 'test_answer'
        self.predict_paths = ["./data/test1.txt"]

    def train(self):
        Train = train(self.model, self.atom_table, self.input_type)
        Train.validation_split = self.validation_split
        Train.batch_size = self.batch_size
        Train.epochs = self.epochs
        Train.log_dir = self.log_dir
        Train.model_dir = self.model_dir
        Train.data_paths = self.data_paths
        history = Train.train(self.model_name)
        return history

    def predict(self):
        Predict = predict(self.model_dir+'/'+self.model_name+'.h5', self.atom_table, self.input_type, self.custom_objects)
        Predict.batch_size = self.batch_size
        Predict.answer_dir = self.answer_dir
        Predict.predict_paths = self.predict_paths
        compss, Tc = Predict.predict(self.answer_name)
        return compss, Tc


Run = run()

#Run.train()

Run.predict()

