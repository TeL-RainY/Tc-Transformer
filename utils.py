import torch
import numpy, re

class embed_models:
    
    __atom_list_short =   ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
                                    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca', 
                                    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
                                    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 
                                    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 
                                    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 
                                    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
                                    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 
                                    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm' 
                                    ]
    __atom_list_long = [   'H' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'He', 'Ce', 'Lu', 'Th', 'Lr',
                                    'Li', 'Be', 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Pr', 'Yb', 'Pa', 'No',
                                    'Na', 'Mg', 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'Nd', 'Tm', 'U' , 'Md',
                                    'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Pm', 'Er', 'Np', 'Fm',
                                    'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe', 'Sm', 'Ho', 'Pu', 'Es',
                                    'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Eu', 'Dy', 'Am', 'Cf',
                                    'Fr', 'Ra', 'Ac', 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'Gd', 'Tb', 'Cm', 'Bk'
                                ]
    __embed_type = ('Full', 'Short')

    def Atom_Table(self)->tuple:
        def shape(vector=None):
            if vector == None: return (None, 1, 10, 10)
            return (len(vector), 1, 10, 10)
        atom_list = list(numpy.reshape(self.__atom_list_short, (10, 10)).T.flatten())
        atom_table = {}
        for location, atom in enumerate(atom_list):
            atom_table[atom] = location
        return atom_table, shape, self.__embed_type[0]

    def Dense_AT(self)->tuple:
        def shape(vector=None):
            if vector == None: return (None, 100)
            return (len(vector), 100)
        atom_list = self.__atom_list_short
        atom_table = {}
        for location, atom in enumerate(atom_list):
            atom_table[atom] = location
        return atom_table, shape, self.__embed_type[0]

    def Origin_AT(self)->tuple:
        def shape(vector=None):
            if vector == None: return (None, 1, 22, 7)
            return (len(vector), 1, 22, 7)
        atom_list = list(numpy.reshape(self.__atom_list_long, (7, 22)).T.flatten())
        atom_table = {}
        for location, atom in enumerate(atom_list):
            atom_table[atom] = location
        return atom_table, shape, self.__embed_type[0]

    def OAT_without_channels(self)->tuple:
        def shape(vector=None):
            if vector == None: return (None, 22, 7)
            return (len(vector), 22, 7)
        atom_list = list(numpy.reshape(self.__atom_list_long, (7, 22)).T.flatten())
        atom_table = {}
        for location, atom in enumerate(atom_list):
            atom_table[atom] = location
        return atom_table, shape, self.__embed_type[0]

    def Short_Embed(self)->tuple:
        def Transform_shape(vector=None):
            if vector == None: return (None, 10, 2)
            return (len(vector), 10, 2)
        atom_list = self.__atom_list_short
        atom_table = {}
        for location, atom in enumerate(atom_list):
            atom_table[atom] = location
        return atom_table, Transform_shape, self.__embed_type[1]
        
class metrics:
    def Tc_accuracy(y_true:torch.Tensor, y_pred:torch.Tensor, temp:float =1.0)->float:
        y_delta = torch.abs(y_true - y_pred).reshape((-1,)).tolist()
        accuracy = sum(delta <= temp for delta in y_delta)/len(y_delta)
        return accuracy                                                                     #认为误差在1k之内为预测正确, 准确率

    def Sc_accuracy(y_true:torch.Tensor, y_pred:torch.Tensor, temp:float =0.1)->tuple:
        y_t = y_true.reshape((-1,)).tolist()
        y_p = y_pred.reshape((-1,)).tolist()
        TP = FN = FP = TN = 0
        for i in range(len(y_t)):
            if y_t[i] > temp:
                if y_p[i] > temp: TP += 1
                else: FN += 1
            else:
                if y_p[i] > temp: FP += 1
                else: TN += 1
        P = TP + FN
        N = FP + TN                                                                         #认为0.1k一下的为非超导体, 以此得到的分类结果
        accuracy = (TP+TN)/(P+N)                                                            #准确率
        precision = TP/(TP+FP if TP+FP!=0 else 100)                                         #精确率
        recall = TP/P                                                                       #召回率
        F1 = 2*precision*recall/(precision+recall if precision+recall!=0 else 1)            #F1值
        return accuracy, precision, recall, F1

    def Performance(y_true:torch.Tensor, y_pred:torch.Tensor)->tuple:
        y_t = y_true.reshape((-1,)).tolist()
        y_p = y_pred.reshape((-1,)).tolist()
        y_t_mean = sum(y_t)/len(y_t)
        y_p_mean = sum(y_p)/len(y_p)
        y_delta = [y_t[i] - y_p[i] for i in range(len(y_t))]
        y_t_delta = [y_ti - y_t_mean for y_ti in y_t]
        y_p_delta = [y_pi - y_p_mean for y_pi in y_p]
        SSres = sum(delta**2 for delta in y_delta)
        SStot = sum(delta**2 for delta in y_t_delta)
        SSreg = sum(delta**2 for delta in y_p_delta)
        R2 = 1.0 - SSres/SStot                                                                          #拟合系数
        RMSE = (SSres/len(y_delta))**0.5                                                                #均方根误差
        MAE = sum(abs(delta) for delta in y_delta)/len(y_delta)                                         #平均绝对误差
        CC = sum((y_t[i]-y_t_mean)*(y_p[i]-y_p_mean) for i in range(len(y_t)))/(SSreg*SStot)**0.5       #相关系数
        return R2, RMSE, MAE, CC

class utils:
    def Read_Component(comps:str)->dict:
        component = {}
        tokens = re.findall('[A-Z][a-z]*|\d+\.\d+|\d+', comps)
        i = 0
        while i < len(tokens):
            if(i+1 >= len(tokens) or tokens[i+1].isalpha()):
                component[tokens[i]] = 1.0
                i += 1
            else:
                component[tokens[i]] = float(tokens[i+1])
                i += 2
        return component

    def Read_Data(data_path:str, flag:str='WithTc')->tuple:
        compss = []
        Tc = []
        with open(data_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                token = line.split()
                compss.append(token[0])
                if flag == 'WithTc':
                    Tc.append(float(token[1]))
        return compss, Tc

    def Embed(data_list:list, embed_model:tuple)->list:
        atom_table, shape_func, embed_type = embed_model
        shape = shape_func()[1:]
        data_embeded_list = []
        for comps in data_list:
            component = utils.Read_Component(comps)
            embed = (max(atom_table.values())+1) * [0]
            sum_num = float(sum(list(component.values())))
            for atom in component.keys():
                embed[atom_table[atom]] = component[atom]/sum_num
            if embed_type == 'Short':
                vector = []
                for i in range(len(embed)):
                    if embed[i] != 0: vector.append([i+1, embed[i]])
                    if len(vector) >= 10: break
                while len(vector) < 10:
                    vector.append([-2**32, -2**32])
                embed = vector
            embed = numpy.reshape(embed, shape).tolist()
            data_embeded_list.append(embed)
        return data_embeded_list

    def Summary_Param(model:torch.nn.Module):
        print("Param: {}".format(sum(x.numel() for x in model.parameters())))