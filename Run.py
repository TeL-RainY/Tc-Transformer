import framework, model, utils

def Run(module:model.torch.nn.Module, module_arg:list, embed_model:tuple, model_name:str, train:bool=True, predict:bool=True)->None:
    opt_train_data = framework.option.data()
    opt_train_data.data_dir_list = [    #'./data/test.txt',
                                        './data/insulator.txt', './data/superconductor.txt'
    ]
    opt_train_data.embed_func = utils.utils.Embed
    opt_train_data.embed_func_arg = [embed_model]
    opt_train_data.read_func = utils.utils.Read_Data
    opt_train_data.read_func_arg = ['WithTc']

    opt_train = framework.option.train()
    opt_train.validate = True
    opt_train.validate_rate = 0.02
    opt_train.epoch = 2500
    opt_train.batch_size = 128
    opt_train.model = module(embed_model[1](), *module_arg)
    opt_train.optimizer_func = model.torch.optim.Adadelta
    opt_train.optimizer_arg = []
    opt_train.lossfunc = model.torch.nn.L1Loss
    opt_train.lossfunc_arg = []
    opt_train.metrics = {   'Tc_accuracy' : (utils.metrics.Tc_accuracy, [1.0]),
                            'Sc_accuracy,Sc_precision,Sc_recall,Sc_F1' : (utils.metrics.Sc_accuracy, [0.1]),
                            'R2,RMSE,MAE,CC' : (utils.metrics.Performance, [])
    }
    opt_train.model_dir = './models'
    opt_train.log_dir = './logs/'+model_name+'_log'
    opt_train.model_name = model_name

    opt_predict_data = framework.option.data()
    opt_predict_data.data_dir_list = ['./data/predict.txt'
                                      #'./data/test.txt'
                                    ]
    opt_predict_data.embed_func = utils.utils.Embed
    opt_predict_data.embed_func_arg = [embed_model]
    opt_predict_data.read_func = utils.utils.Read_Data
    opt_predict_data.read_func_arg = ['WithoutTc']

    opt_predict = framework.option.predict()
    opt_predict.model_path = './models/'+model_name+'.pkl'
    opt_predict.predict_dir = './predict'
    opt_predict.predict_name = model_name+'_answer'

    if train:framework.train(opt_train_data, opt_train).Run()
    if predict:framework.predict(opt_predict_data, opt_predict).Run()

#   Run(model.models.SetTransformer, ['Dense_Decoder', 6, 3], utils.embed_models().OAT_without_channels(), 'SetTransformer7-2', train=False)

#   Run(model.models.ATCNN, [64], utils.embed_models().Atom_Table(), 'ATCNN1-1', train=False)

for i in range(1, 6):
    Run(model.models.SetTransformer, ['Dense_Decoder', 6, 3], utils.embed_models().OAT_without_channels(), 'SetTransformer7-'+'{}'.format(i), train=False)
    Run(model.models.ATCNN, [64], utils.embed_models().Atom_Table(), 'ATCNN1-'+'{}'.format(i), train=False)