import torch.nn.functional as torch_F
import torch
import math, utils

class wrap:
    class Conv2D_Norm_Act(torch.nn.Module):
        def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple, activate:bool=True):
            super(wrap.Conv2D_Norm_Act, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.activate = activate

            self.pad = torch.nn.ConstantPad2d((math.floor((self.kernel_size[0]-1)/2), math.ceil((self.kernel_size[0]-1)/2), math.floor((self.kernel_size[1]-1)/2), math.ceil((self.kernel_size[1]-1)/2)), 0.0)
            self.conv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
            self.norm = torch.nn.BatchNorm2d(self.out_channels)

        def forward(self, input):
            x = self.pad(input)
            x = self.conv(x)
            x = self.norm(x)
            if self.activate: x = torch_F.relu(x)
            return x

    class Linear_Norm_Drop_Act(torch.nn.Module):
        def __init__(self, in_features:int, out_features:int, bias:bool=True, drop:bool=True, drop_rate:float=0.2, activate:bool=True):
            super(wrap.Linear_Norm_Drop_Act, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            self.drop = drop
            self.drop_rate = drop_rate
            self.activate = activate

            self.linear = torch.nn.Linear(self.in_features, self.out_features, self.bias)
            self.norm = torch.nn.BatchNorm1d(self.out_features)
            if self.drop: self.dropout = torch.nn.Dropout(self.drop_rate)

        def forward(self, input):
            x = self.linear(input)
            x = self.norm(x)
            if self.drop: x = self.dropout(x)
            if self.activate: x = torch_F.relu(x)
            return x

    class Identity_Block(torch.nn.Module):
        def __init__(self, in_channels:int, kernel_size:tuple):
            super(wrap.Identity_Block, self).__init__()
            self.channel = in_channels
            self.kernel_size = kernel_size

            self.conv = torch.nn.Sequential(wrap.Conv2D_Norm_Act(self.channel, self.channel, self.kernel_size), wrap.Conv2D_Norm_Act(self.channel, self.channel, self.kernel_size), wrap.Conv2D_Norm_Act(self.channel, self.channel, self.kernel_size, activate=False))

        def forward(self, input):
            shortcut = input
            x = self.conv(input)
            x += shortcut
            output = torch_F.relu(x)
            return output

    class Conv_Block(torch.nn.Module):
        def __init__(self, in_channels:int, filter_num:int, kernel_size:tuple):
            super(wrap.Conv_Block, self).__init__()
            self.channel = in_channels
            self.filter = filter_num
            self.kernel_size = kernel_size

            self.conv1 = torch.nn.Sequential(wrap.Conv2D_Norm_Act(self.channel, self.filter, self.kernel_size), wrap.Conv2D_Norm_Act(self.filter, self.filter, self.kernel_size), wrap.Conv2D_Norm_Act(self.filter, self.filter, self.kernel_size, activate=False))
            self.conv2 = wrap.Conv2D_Norm_Act(self.channel, self.filter, (1, 1), activate=False)

        def forward(self, input):
            shortcut = input
            x = self.conv1(input)
            x += self.conv2(shortcut)
            output = torch_F.relu(x)
            return output

    class Feed_Forward(torch.nn.Module):
        def __init__(self, in_features:int, feed_forward_size:int):
            super(wrap.Feed_Forward, self).__init__()
            self.in_features = in_features
            self.feed_forward_size = feed_forward_size

            self.linear1 = torch.nn.Linear(self.in_features, self.feed_forward_size)
            self.linear2 = torch.nn.Linear(self.feed_forward_size, self.in_features)

        def forward(self, input):
            x = self.linear1(input)
            x = torch_F.relu(x)
            output = self.linear2(x)
            return output

    class Positional_Encoding(torch.nn.Module):
        def __init__(self, input_shape:tuple):
            super(wrap.Positional_Encoding, self).__init__()
            self.batch, self.length, self.dim = input_shape
            position_encodings = torch.zeros((self.length, self.dim), requires_grad=False)
            for pos in range(self.length):
                for i in range(self.dim):
                    position_encodings[pos][i] = 2*pos#pos/10000**((i-i%2)/self.dim)
            #position_encodings[:, 0::2] = torch.sin(position_encodings[:, 0::2])
            #position_encodings[:, 1::2] = torch.cos(position_encodings[:, 1::2])
            position_encodings = position_encodings.unsqueeze(0)
            self.register_buffer('PE', position_encodings)

        def forward(self, input:torch.Tensor)->torch.Tensor:
            position_encodings = self.PE.expand((input.shape[0], -1, -1))
            return input + position_encodings

    class Dense_Decoder(torch.nn.Module):
        def __init__(self, input_shape:tuple, decoder_stack:int, feed_forward_size:int, drop_rate:float):
            super(wrap.Dense_Decoder, self).__init__()
            self.input_shape = input_shape
            self.batch, self.length, self.dim = self.input_shape
            self.decoder_stack = decoder_stack
            self.feed_forward_size = feed_forward_size
            self.drop_rate = drop_rate

            decoder_list = [torch.nn.Linear(self.dim, self.feed_forward_size)]
            for _ in range(self.decoder_stack):
                decoder_list.append(torch.nn.Linear(self.feed_forward_size, self.feed_forward_size))
                decoder_list.append(torch.nn.Linear(self.feed_forward_size, self.feed_forward_size))
                decoder_list.append(torch.nn.LayerNorm(self.feed_forward_size))
                decoder_list.append(torch.nn.Dropout(self.drop_rate))
                decoder_list.append(torch.nn.ReLU())
            decoder_list.append(torch.nn.Linear(self.feed_forward_size, self.dim))
            self.decoder = torch.nn.Sequential(*decoder_list)

        def forward(self, tgt:torch.Tensor, memory:torch.Tensor, tgt_mask= None, memory_mask= None, tgt_key_padding_mask= None, memory_key_padding_mask= None)->torch.Tensor:
            return self.decoder(memory)

    class Pooling_Decoder_Layer(torch.nn.Module):
        def __init__(self, in_dim:int, feed_forward_size:int, drop_rate:float):
            super(wrap.Pooling_Decoder_Layer, self).__init__()
            self.dim = in_dim
            self.dropout = drop_rate
            self.feed_forward_size = feed_forward_size

            self.atte1 = torch.nn.MultiheadAttention(self.dim, 1, self.dropout, batch_first=True)
            #self.norm1 = torch.nn.LayerNorm(self.dim)
            self.ff1 = wrap.Feed_Forward(self.dim, self.feed_forward_size)
            self.norm2 = torch.nn.LayerNorm(self.dim)
            self.atte2 = torch.nn.MultiheadAttention(self.dim, 1, self.dropout, batch_first=True)
            #self.norm3 = torch.nn.LayerNorm(self.dim)
            self.ff2 = wrap.Feed_Forward(self.dim, self.feed_forward_size)
            self.norm4 = torch.nn.LayerNorm(self.dim)
            #self.ff3 = wrap.Feed_Forward(self.dim, self.feed_forward_size)

        def forward(self, tgt:torch.Tensor, memory:torch.Tensor, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)->torch.Tensor:
            #shortcut = tgt
            x, _ = self.atte1(tgt, memory, memory)
            #x += shortcut
            #x = self.norm1(x)
            #shortcut = x
            x = self.ff1(x)
            #x += shortcut
            x = self.norm2(x)
            #shortcut = x
            x, _ = self.atte2(x, x, x)
            #x += shortcut
            #x = self.norm3(x)
            #shortcut = x
            x = self.ff2(x)
            #x += shortcut
            #x = self.norm4(x)
            output = self.norm4(x)
            #output = self.ff3(x)
            return output

    class Pooling_Decoder(torch.nn.Module):
        def __init__(self, input_shape:tuple, decoder_stack:int, feed_forward_size:int, drop_rate:float):
            super(wrap.Pooling_Decoder, self).__init__()
            self.input_shape = input_shape
            self.batch, self.length, self.dim = self.input_shape
            self.decoder_stack = decoder_stack
            self.feed_forward_size = feed_forward_size
            self.drop_rate = drop_rate

            trainable_queries = torch.nn.parameter.Parameter(torch.empty((1, self.length, self.dim)))
            trainable_queries.data.uniform_(-1, 1)
            self.register_parameter('queries', trainable_queries)
            self.decoderlayer_mlist = torch.nn.ModuleList(self.decoder_stack * [wrap.Pooling_Decoder_Layer(self.dim, self.feed_forward_size, self.drop_rate)])
            self.dropout = torch.nn.Dropout(self.drop_rate)

        def forward(self, tgt:torch.Tensor, memory:torch.Tensor, tgt_mask= None, memory_mask= None, tgt_key_padding_mask= None, memory_key_padding_mask= None)->torch.Tensor:
            x = self.queries.expand(memory.shape[0], -1, -1)
            x = self.dropout(x)
            for decoderlayer in self.decoderlayer_mlist:
                x = decoderlayer(x, memory)
            output = x
            return output

class models:
    class ATCNN(torch.nn.Module):
        def __init__(self, input_shape:tuple, filter_num:int):
            #Input : (batch_size, channels, length, dim) (None, 1, 10, 10) or (None, 1, 22, 7)
            super(models.ATCNN, self).__init__()
            self.batch, self.channel, self.length, self.dim = input_shape
            self.filter = filter_num

            self.conv_list = [wrap.Conv2D_Norm_Act(self.channel, self.filter, (5, 5))]
            for _ in range(3):
                self.conv_list.append(wrap.Conv2D_Norm_Act(self.filter, self.filter, (3, 3)))
            self.conv_list.append(wrap.Conv2D_Norm_Act(self.filter, self.filter, (2, 2)))
            self.conv = torch.nn.Sequential(*self.conv_list)
            self.maxpool = torch.nn.Sequential(torch.nn.MaxPool2d((2, 2)), torch.nn.BatchNorm2d(self.filter), torch.nn.ReLU())
            self.flatten = torch.nn.Flatten(1)
            self.linear = torch.nn.Sequential(wrap.Linear_Norm_Drop_Act(self.filter*math.floor((self.length-2)/2+1)*math.floor((self.dim-2)/2+1), 200, drop_rate=0.2), wrap.Linear_Norm_Drop_Act(200, 100, drop_rate=0.2))
            self.output = torch.nn.Linear(100, 1)

        def forward(self, input):
            x = self.conv(input)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.linear(x)
            output = self.output(x)
            return output

    class Dense(torch.nn.Module):
        def __init__(self, input_shape:tuple):
            #Input : (batch_size, dim) (None, 100)
            super(models.Dense, self).__init__()
            self.dim = input_shape[-1]

            linear1_list = 5 * [wrap.Linear_Norm_Drop_Act(self.dim, self.dim, drop=False)]
            self.linear1 = torch.nn.Sequential(*linear1_list)
            self.linear2 = torch.nn.Sequential(wrap.Linear_Norm_Drop_Act(self.dim, self.dim, drop_rate=0.2), wrap.Linear_Norm_Drop_Act(self.dim, self.dim, drop_rate=0.2))
            self.output = torch.nn.Linear(self.dim, 1)

        def forward(self, input):
            x = self.linear1(input)
            x = self.linear2(x)
            output = self.output(x)
            return output

    class ResNet_50(torch.nn.Module):
        def __init__(self, input_shape:tuple, filter_num:int):
            #Input : (batch_size, channels, length, dim) (None, 1, 22, 7)
            super(models.ResNet_50, self).__init__()

            self.batch, self.channel, self.length, self.dim = input_shape
            self.filter = filter_num

            self.conv_list = [wrap.Conv2D_Norm_Act(self.channel, self.filter, (3, 3))]
            for _ in range(4):
                self.conv_list.append(wrap.Conv_Block(self.filter, self.filter, (2, 2)))
                for _ in range(3):
                    self.conv_list.append(wrap.Identity_Block(self.filter, (2, 2)))
            self.conv = torch.nn.Sequential(*self.conv_list)
            self.avgpool = torch.nn.AvgPool2d((2, 2))
            self.flatten = torch.nn.Flatten(1)
            self.linear = torch.nn.Sequential(wrap.Linear_Norm_Drop_Act(self.filter*math.floor((self.length-2)/2+1)*math.floor((self.dim-2)/2+1), 200, drop_rate=0.2), wrap.Linear_Norm_Drop_Act(200, 100, drop_rate=0.2))
            self.output = torch.nn.Linear(100, 1)
        
        def forward(self, input):
            x = self.conv(input)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.linear(x)
            output = self.output(x)
            return output

    class Transformer(torch.nn.Module):
        def __init__(self, input_shape:tuple, encoder_stack:int, decoder_stack:int):
            #Input : (batch_size, length, dim) (None, 22, 7) or (None, 10, 2)
            super(models.Transformer, self).__init__()
            self.input_shape = input_shape
            self.batch, self.length, self.dim = self.input_shape
            self.encoder_stack = encoder_stack
            self.decoder_stack = decoder_stack

            self.positional_encoder =wrap.Positional_Encoding(self.input_shape)
            self.transformer = torch.nn.Transformer(self.dim, 1, self.encoder_stack, self.decoder_stack, 100, 0.1, batch_first=True)
            self.flatten = torch.nn.Flatten(1)
            self.linear = torch.nn.Sequential(wrap.Linear_Norm_Drop_Act(self.length*self.dim, 200, drop_rate=0.2), wrap.Linear_Norm_Drop_Act(200, 100, drop_rate=0.1))
            self.output = torch.nn.Linear(100, 1)

        def forward(self, input:torch.Tensor, tgt:torch.Tensor):
            y = tgt.reshape((tgt.shape[0], 1, 1))
            y = y.expand(-1, input.shape[1], input.shape[2])
            x = self.positional_encoder(input)
            x = self.transformer(x, y)
            x = self.flatten(x)
            x = self.linear(x)
            output = self.output(x)
            return output

    class SetTransformer(torch.nn.Module):
        def __init__(self, input_shape:tuple, flag:str, encoder_stack:int, decoder_stack:int):
            #Input : (batch_size, length, dim) (None, 22, 7) or (None, 10, 2)
            super(models.SetTransformer, self).__init__()
            self.flag = flag
            self.input_shape = input_shape
            self.batch, self.length, self.dim = self.input_shape
            self.encoder_stack = encoder_stack
            self.decoder_stack = decoder_stack

            self.positional_encoder =wrap.Positional_Encoding(self.input_shape)
            if flag == 'Dense_Decoder':
                #decoder_layer = wrap.Dense_Decoder_Layer(self.dim, 100, 0.05)
                #decoder = torch.nn.TransformerDecoder(decoder_layer, self.decoder_stack)
                decoder = wrap.Dense_Decoder(self.input_shape, self.decoder_stack, 100, 0.02)
            elif flag == 'Pooling_Decoder':
                decoder = wrap.Pooling_Decoder(self.input_shape, self.decoder_stack, 100, 0.0)
            self.transformer = torch.nn.Transformer(self.dim, 1, self.encoder_stack, self.decoder_stack, 100, 0.0, custom_decoder=decoder, batch_first=True)
            self.flatten = torch.nn.Flatten(1)
            self.linear = torch.nn.Sequential(wrap.Linear_Norm_Drop_Act(self.length*self.dim, 200, drop_rate=0.0), wrap.Linear_Norm_Drop_Act(200, 100, drop_rate=0.0))
            self.output = torch.nn.Linear(100, 1)

        def forward(self, input):
            tgt = torch.zeros_like(input)
            x = self.positional_encoder(input)
            x = self.transformer(x, tgt)
            x = self.flatten(x)
            x = self.linear(x)
            output = self.output(x)
            return output

#utils.utils.Summary_Param(models.SetTransformer((128, 22, 7), 'Dense_Decoder', 6, 3))