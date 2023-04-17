import keras.models as kmodels
import keras.layers as layers
import keras.losses as losses
import keras.optimizers as optimizers
import keras.backend as backend
import tensorflow
import numpy

class atom_table:
    def ATCNN():
        def ATCNN_shape(vector):
            return (len(vector), 10, 10, 1)
        atom_list = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
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
        atom_list = list(numpy.reshape(atom_list, (10, 10)).T.flatten())
        at = {}
        for location, atom in enumerate(atom_list):
            at[atom] = location
        return (at, ATCNN_shape)

    def Dense():
        def Dense_shape(vector):
            return (len(vector), 100)
        atom_list = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
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
        at = {}
        for location, atom in enumerate(atom_list):
            at[atom] = location
        return (at, Dense_shape)

    def OATCNN():
        def OATCNN_shape(vector):
            return (len(vector), 22, 7, 1)
        atom_list = [   'H' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'He', 'Ce', 'Lu', 'Th', 'Lr',
                        'Li', 'Be', 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Pr', 'Yb', 'Pa', 'No',
                        'Na', 'Mg', 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'Nd', 'Tm', 'U' , 'Md',
                        'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Pm', 'Er', 'Np', 'Fm',
                        'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe', 'Sm', 'Ho', 'Pu', 'Es',
                        'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Eu', 'Dy', 'Am', 'Cf',
                        'Fr', 'Ra', 'Ac', 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'X' , 'Gd', 'Tb', 'Cm', 'Bk'
                    ]
        atom_list = list(numpy.reshape(atom_list, (7, 22)).T.flatten())
        at = {}
        for location, atom in enumerate(atom_list):
            at[atom] = location
        return (at, OATCNN_shape)

    def RESATCNN():
        return atom_table.OATCNN()

    def SelfAttention():
        def SelfAttention_shape(vector):
            return (len(vector), 22, 7)
        return (atom_table.OATCNN()[0], SelfAttention_shape)

    def Transform():
        def Transform_shape(vector):
            return (len(vector), 10, 2)
        atom_list = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
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
        at = {}
        for location, atom in enumerate(atom_list):
            at[atom] = location
        return (at, Transform_shape)

    def Set_Transform():
        return atom_table.SelfAttention()

class metrics:
    def Tc_accuracy(y_true, y_pred):
        y_delta = backend.abs(backend.reshape(y_true - y_pred, [-1]))
        tensor_one = backend.ones_like(y_delta)
        accuracy = backend.cast(backend.less_equal(y_delta, tensor_one), 'float32')
        accuracy = backend.sum(accuracy)/float(len(y_delta))
        return accuracy                                                #认为误差在1k之内为预测正确, 准确率

    def superconductor_accuracy(y_true, y_pred):
        y_t = backend.reshape(y_true, [-1])
        y_p = backend.reshape(y_pred, [-1])
        tensor = backend.ones_like(y_t) * 0.1
        zero = backend.zeros_like(y_t, 'float32')
        y_t = backend.not_equal(y_t, zero)
        y_p = backend.greater(y_p, tensor)
        TP = backend.sum(backend.cast(y_t & y_p, 'float32'))
        FN = backend.sum(backend.cast(y_t & ~y_p, 'float32'))
        FP = backend.sum(backend.cast(~y_t & y_p, 'float32'))
        TN = backend.sum(backend.cast(~y_t & ~y_p, 'float32'))
        P = TP + FN
        N = FP + TN                                                                     #认为0.1k一下的为非超导体, 以此得到的分类结果
        accuracy = (TP+TN)/(P+N)                                                        #准确率
        precision = TP/(TP+FP)                                                          #精确率
        recall = TP/P                                                                   #召回率
        F1 = 2*precision*recall/(precision+recall)                                      #F1值
        return accuracy

    def gen_performance(y_true, y_pred):
        y_t = backend.reshape(y_true, [-1])
        y_p = backend.reshape(y_pred, [-1])
        y_t_mean = backend.mean(y_t)
        y_p_mean = backend.mean(y_p)
        y_delta = y_t - y_p
        SSres = backend.sum(backend.square(y_delta))
        SStot = backend.sum(backend.square(y_t - y_t_mean))
        SSreg = backend.sum(backend.square(y_p - y_p_mean))
        R2 = 1.0 - SSres/SStot                                                          #拟合系数
        RMSE = backend.sqrt(SSres/float(len(y_delta)))                                  #均方根误差
        MAE = backend.mean(backend.abs(y_delta))                                        #平均绝对误差
        CC = backend.sum((y_t-y_t_mean)*(y_p-y_p_mean))/backend.sqrt(SSreg*SStot)       #相关系数
        return R2

class customlayers:

    class Positional_Encoding(layers.Layer):
        def __init__(self, **kwargs):
            super(customlayers.Positional_Encoding, self).__init__(**kwargs)
        def get_config(self):
            return super().get_config()
        def build(self, input_shape):
            super().build(input_shape)
        def call(self, inputs):
            if backend.dtype(inputs) != 'float32': inputs = backend.cast(inputs, 'float32')
            model_dim = inputs.shape[-1]
            seq_length = inputs.shape[1]
            position_encodings = numpy.zeros((seq_length, model_dim))
            for pos in range(seq_length):
                for i in range(model_dim):
                    position_encodings[pos][i] = pos / numpy.power(10000, (i-i%2) / model_dim)
            position_encodings[:, 0::2] = numpy.sin(position_encodings[:, 0::2])
            position_encodings[:, 1::2] = numpy.cos(position_encodings[:, 1::2])
            position_encodings = backend.cast(position_encodings, 'float32')
            outputs = inputs + position_encodings
            return outputs
        def compute_output_shape(self, input_shape):
            return input_shape

    class Scaled_Dot_Product_Attention(layers.Layer):
        def __init__(self, **kwargs):
            super(customlayers.Scaled_Dot_Product_Attention, self).__init__(**kwargs)
        def get_config(self):
            return super().get_config()
        def build(self, input_shape):
            super().build(input_shape)
        def call(self, inputs):
            queries, keys, values = inputs
            if backend.dtype(queries) != 'float32': queries = backend.cast(queries, 'float32')
            if backend.dtype(keys) != 'float32': keys = backend.cast(keys, 'float32')
            if backend.dtype(values) != 'float32': values = backend.cast(values, 'float32')
            matmul = backend.batch_dot(queries, tensorflow.transpose(keys, [0, 2, 1]))
            scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5
            softmax_out = backend.softmax(scaled_matmul)
            outputs = backend.batch_dot(softmax_out, values)
            return outputs
        def comopute_output_shape(self, input_shape):
            shape_q, shape_k, shape_v = input_shape
            return shape_v

    class Multi_Head_Attention(layers.Layer):
        def __init__(self, n_heads, keys_dim, values_dim, create_queries=False, **kwargs):
            self.n_heads = n_heads
            self.keys_dim = keys_dim
            self.values_dim = values_dim
            self.create_queries = create_queries
            super(customlayers.Multi_Head_Attention, self).__init__(**kwargs)
        def get_config(self):
            config = super(customlayers.Multi_Head_Attention, self).get_config()
            config.update({
                'n_heads' : self.n_heads,
                'keys_dim' : self.keys_dim,
                'values_dim' : self.values_dim,
                'create_queries' : self.create_queries
            })
            return config
        def build(self, input_shape):
            self.weights_q = self.add_weight(name='weights_queries', shape=(input_shape[0][-1], self.n_heads * self.keys_dim), initializer='glorot_uniform', trainable=True)
            self.weights_k = self.add_weight(name='weights_keys', shape=(input_shape[-2][-1], self.n_heads * self.keys_dim), initializer='glorot_uniform', trainable=True)
            self.weights_v = self.add_weight(name='weights_values', shape=(input_shape[-1][-1], self.n_heads * self.values_dim), initializer='glorot_uniform', trainable=True)
            self.weights_linear = self.add_weight(name='weights_linear', shape=(self.n_heads * self.values_dim, input_shape[0][-1]), initializer='glorot_uniform', trainable=True)
            if self.create_queries:
                self.queries = self.add_weight(name='queries', shape=(input_shape[0][-2], input_shape[0][-1]), initializer='glorot_uniform', trainable=True)
            super(customlayers.Multi_Head_Attention, self).build(input_shape)
        def call(self, inputs):
            if self.create_queries:
                keys, values = inputs
                queries = tensorflow.transpose(backend.repeat(self.queries, backend.shape(keys)[0]), [1,0,2])
            else:
                queries, keys, values = inputs
            queries_linear = backend.dot(queries, self.weights_q)
            keys_linear = backend.dot(keys, self.weights_k)
            values_linear = backend.dot(values, self.weights_v)
            queries_multi_heads = backend.concatenate(tensorflow.split(queries_linear, self.n_heads, axis=2), axis=0)
            keys_multi_heads = backend.concatenate(tensorflow.split(keys_linear, self.n_heads, axis=2), axis=0)
            values_multi_heads = backend.concatenate(tensorflow.split(values_linear, self.n_heads, axis=2), axis=0)
            attention_out = customlayers.Scaled_Dot_Product_Attention()([queries_multi_heads, keys_multi_heads, values_multi_heads])
            attention_concat = backend.concatenate(tensorflow.split(attention_out, self.n_heads, axis=0), axis=2)
            outputs = backend.dot(attention_concat, self.weights_linear)
            return outputs
        def compute_output_shape(self, input_shape):
            if self.create_queries:
                k_shape, v_shape = input_shape
                return v_shape
            else:
                q_shape, k_shape, v_shape = input_shape
                return q_shape

class wraplayers:
    def Conv2D_Norm_Act(filters, kernel_size, padding='same', activate=True, activation='relu'):
        def call(inputs):
            conv_outputs = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
            outputs = layers.BatchNormalization()(conv_outputs)
            if activate:
                outputs = layers.Activation(activation)(outputs)
            return outputs
        return call

    def Identity_Block(filters, kernel_size, activation='relu'):
        def call(inputs):
            outputs = wraplayers.Conv2D_Norm_Act(filters, kernel_size, activation=activation)(inputs)
            outputs = wraplayers.Conv2D_Norm_Act(filters, kernel_size, activation=activation)(outputs)
            outputs = wraplayers.Conv2D_Norm_Act(filters, kernel_size, activate=False)(outputs)
            outputs = layers.add([outputs, inputs])
            outputs = layers.Activation(activation)(outputs)
            return outputs
        return call

    def Conv_Block(filters, kernel_size, activation='relu'):
        def call(inputs):
            outputs = wraplayers.Conv2D_Norm_Act(filters, kernel_size, activation=activation)(inputs)
            outputs = wraplayers.Conv2D_Norm_Act(filters, kernel_size, activation=activation)(outputs)
            outputs = wraplayers.Conv2D_Norm_Act(filters, kernel_size, activate=False)(outputs)
            outputs2 = wraplayers.Conv2D_Norm_Act(filters, kernel_size, activate=False)(inputs)
            outputs = layers.add([outputs, outputs2])
            outputs = layers.Activation(activation)(outputs)
            return outputs
        return call

    def Feed_Forward(size, activation):
        def call(inputs):
            outputs = layers.Dense(size, activation=activation)(inputs)
            outputs = layers.Dense(inputs.shape[-1])(outputs)
            return outputs
        return call

    def Encoder(n_heads, keys_dim, values_dim, feed_forward_size):
        def call(inputs):
            attention_outputs = customlayers.Multi_Head_Attention(n_heads, keys_dim, values_dim)([inputs, inputs, inputs])
            add_outputs = layers.add([attention_outputs, inputs])
            norm_outputs = layers.LayerNormalization()(add_outputs)
            dense_outputs = wraplayers.Feed_Forward(feed_forward_size, 'relu')(norm_outputs)
            add_outputs = layers.add([dense_outputs, norm_outputs])
            norm_outputs = layers.LayerNormalization()(add_outputs)
            return norm_outputs
        return call

    def Decoder(n_heads, keys_dim, values_dim, feed_forward_size):
        def call(inputs):
            inputs_from_encoder, inputs_from_outputs = inputs
            attention_outputs = customlayers.Multi_Head_Attention(n_heads, keys_dim, values_dim)([inputs_from_outputs, inputs_from_outputs, inputs_from_outputs])
            add_outputs = layers.add([attention_outputs, inputs_from_outputs])
            norm_outputs = layers.LayerNormalization()(add_outputs)
            attention_outputs = customlayers.Multi_Head_Attention(n_heads, keys_dim, values_dim)([norm_outputs, inputs_from_encoder, inputs_from_encoder])
            add_outputs = layers.add([attention_outputs, norm_outputs])
            norm_outputs = layers.LayerNormalization()(add_outputs)
            dense_outputs = wraplayers.Feed_Forward(feed_forward_size, 'relu')(norm_outputs)
            add_outputs = layers.add([dense_outputs, norm_outputs])
            norm_outputs = layers.LayerNormalization()(add_outputs)
            return norm_outputs
        return call

    def Set_Decoder(n_heads, keys_dim, values_dim, feed_forward_size):
        def call(inputs):
            inputs_from_encoder = inputs
            attention1_outputs = customlayers.Multi_Head_Attention(n_heads, keys_dim, values_dim, True)([inputs_from_encoder, inputs_from_encoder])
            attention2_outputs = customlayers.Multi_Head_Attention(n_heads, keys_dim, values_dim)([attention1_outputs, attention1_outputs, attention1_outputs])
            add_outputs = layers.add([attention1_outputs, attention2_outputs])
            norm_outputs = layers.LayerNormalization()(add_outputs)
            dense_outputs = wraplayers.Feed_Forward(feed_forward_size, 'relu')(norm_outputs)
            add_outputs = layers.add([dense_outputs, norm_outputs])
            norm_outputs = layers.LayerNormalization()(add_outputs)
            dense_outputs = wraplayers.Feed_Forward(feed_forward_size, 'relu')(norm_outputs)
            norm_outputs = layers.LayerNormalization()(add_outputs)
            return norm_outputs
        return call

    def Transform(n_heads, keys_dim, values_dim, encoder_stack, decoder_stack, feed_forward_size):
        def call(inputs):
            xe = inputs
            for i in range(encoder_stack):
                xe = wraplayers.Encoder(n_heads, keys_dim, values_dim, feed_forward_size)(xe)
            xd = backend.zeros_like(inputs)
            xd = layers.Reshape((1, -1))(xd)
            for i in range(decoder_stack):
                xd = wraplayers.Decoder(n_heads, keys_dim, values_dim, feed_forward_size)([xe, xd])
            outputs = xd
            return outputs
        return call

    def Set_Transform(n_heads, keys_dim, values_dim, encoder_stack, decoder_stack, feed_forward_size):
        def call(inputs):
            xe = inputs
            for i in range(encoder_stack):
                xe = wraplayers.Encoder(n_heads, keys_dim, values_dim, feed_forward_size)(xe)
            xd = wraplayers.Feed_Forward(feed_forward_size, 'relu')(xe)
            for i in range(decoder_stack):
                xd = wraplayers.Set_Decoder(n_heads, keys_dim, values_dim, feed_forward_size)(xe)
            outputs = xd
            return outputs
        return call
        
class models:
    def ATCNN_model(metric = [metrics.Tc_accuracy]):
        model = kmodels.Sequential()

        model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', input_shape=(10, 10, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(200))

        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))

        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(1))
        model.add(layers.Activation('linear'))

        model.compile(optimizer=optimizers.adadelta_v2.Adadelta(), loss=losses.mean_absolute_error, metrics = metric)

        return model

    def Dense_model(metric = [metrics.Tc_accuracy]):
        model = kmodels.Sequential()

        model.add(layers.Dense(units=100, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))

        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))

        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(1))
        model.add(layers.Activation('linear'))

        model.compile(optimizer=optimizers.adadelta_v2.Adadelta(), loss=losses.mean_absolute_error, metrics = metric)

        return model

    def OATCNN_model(metric = [metrics.Tc_accuracy]):
        model = kmodels.Sequential()

        model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', input_shape=(22, 7, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.MaxPool2D(pool_size=(2, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(200))

        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(100))

        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Activation('relu'))

        model.add(layers.Dense(1))
        model.add(layers.Activation('linear'))

        model.compile(optimizer=optimizers.adadelta_v2.Adadelta(), loss=losses.mean_absolute_error, metrics = metric)

        return model

    def RESATCNN_model(metric = [metrics.Tc_accuracy]):
        input = layers.Input((22, 7, 1))

        x = layers.ZeroPadding2D((2, 2))(input)

        x = wraplayers.Conv2D_Norm_Act(filters=64, kernel_size=(3, 2), padding='valid')(x)
        
        for i in range(0, 4):
            x = wraplayers.Conv_Block(filters=64, kernel_size=(2, 2), activation='relu')(x)
            for j in range(0, 3):
                x = wraplayers.Identity_Block(filters =64, kernel_size=(2, 2), activation='relu')(x)
        
        x = layers.AveragePooling2D(pool_size=(2, 1))(x)
        x = layers.Flatten()(x)

        x = layers.Dense(200)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(100)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(1)(x)
        x = layers.Activation('linear')(x)
        
        model = kmodels.Model(inputs=input, outputs=x)
        model.compile(optimizer=optimizers.adadelta_v2.Adadelta(), loss=losses.mean_absolute_error, metrics = metric)

        return model

    def Transform_model(metric = [metrics.Tc_accuracy]):
        inputs = layers.Input((10, 2))
        tran_outputs = wraplayers.Transform(n_heads=8, keys_dim=64, values_dim=64, encoder_stack=6, decoder_stack=6, feed_forward_size=100)(inputs)

        x = layers.Flatten()(tran_outputs)
        x = layers.Dense(100)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(100)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(1)(x)
        x = layers.Activation('linear')(x)

        model = kmodels.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizers.adadelta_v2.Adadelta(), loss=losses.mean_absolute_error, metrics = metric)

        return model

    def Set_Transform_model(metric = [metrics.Tc_accuracy]):
        inputs = layers.Input(shape=(22, 7))

        x = customlayers.Positional_Encoding()(inputs)
        x = wraplayers.Set_Transform(n_heads=8, keys_dim=64, values_dim=64, encoder_stack=6, decoder_stack=2, feed_forward_size=100)(x)

        x = layers.Flatten()(x)

        for i in range(2):
            x = layers.Dense(100)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Activation('relu')(x)

        x = layers.Dense(1)(x)
        x = layers.Activation('linear')(x)

        model = kmodels.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizers.adam_v2.Adam(), loss=losses.mean_absolute_error, metrics = metric)

        return model

    def SelfAttention_model(metric = [metrics.Tc_accuracy]):
        inputs = layers.Input((22, 7))

        x = customlayers.Positional_Encoding()(inputs)
        for i in range(8):
            x = wraplayers.Encoder(n_heads=8, keys_dim=64, values_dim=64, feed_forward_size=100)(x)
            x = layers.Activation('relu')(x)

        for i in range(2):
            dense_outputs = wraplayers.Feed_Forward(100, 'relu')(x)
            x = layers.add([x, dense_outputs])
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Activation('relu')(x)

        x = layers.Flatten()(x)

        for i in range(2):
            x = layers.Dense(100)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Activation('relu')(x)

        x = layers.Dense(1)(x)
        x = layers.Activation('linear')(x)

        model = kmodels.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=optimizers.adadelta_v2.Adadelta(), loss=losses.mean_absolute_error, metrics = metric)

        return model
        
#models.Set_Transform_model().summary()
