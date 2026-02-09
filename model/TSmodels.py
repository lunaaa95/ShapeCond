import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def get_network(args, model_name, mode=None):
    # 随机初始化一个种子。这个种子就保证了没法准确复现？
    torch.random.manual_seed(2025)
    if(model_name == 'MLP'):
        if mode == 'sl':
            net = MLP_sl(input_dim=args.time_step * args.channel, hid_dim = 128, hid2_dim=128, out_dim=args.num_classes, channel=args.channel, num_shapelets=args.num_shapelets, sl_net=args.sl_net)
        else:
            net = MLP(input_dim=args.time_step * args.channel, hid_dim = 128, hid2_dim=128, out_dim=args.num_classes, num_shapelets=args.num_shapelets, sl_net=args.sl_net)
    elif(model_name == "CNNIN"):
        net = ConvNet(channel=args.channel, num_classes=args.num_classes, net_width=32, net_depth=3, net_act='relu', net_norm='IN', net_pooling='maxpooling', im_size=args.time_step, num_shapelets=args.num_shapelets, sl_net=args.sl_net)
    elif(model_name == "CNNBN"):
        if mode == 'sl':
            net = ConvNet_sl(channel=args.channel, num_classes=args.num_classes, net_width=32, net_depth=3, net_act='relu', net_norm='BN', net_pooling='maxpooling', im_size=args.time_step, num_shapelets=args.num_shapelets, sl_net=args.sl_net)
        else:
            net = ConvNet(channel=args.channel, num_classes=args.num_classes, net_width=32, net_depth=3, net_act='relu', net_norm='BN', net_pooling='maxpooling', im_size=args.time_step, num_shapelets=args.num_shapelets, sl_net=args.sl_net)
    elif(model_name == "CNNLN"):
        net = ConvNet(channel=args.channel, num_classes=args.num_classes, net_width=32, net_depth=3, net_act='relu', net_norm='LN', net_pooling='maxpooling', im_size=args.time_step, num_shapelets=args.num_shapelets, sl_net=args.sl_net)
    elif(model_name == 'CNNBN2D'):
        net = ConvNet2D(channel=args.channel, num_classes=args.num_classes, net_width=32, net_depth=3, net_act='relu', net_norm='BN', net_pooling='maxpooling')
    elif(model_name == 'AlexNetBN'):
        net = AlexNetBN(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'AlexNet'):
        net = AlexNet(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'VGG11'):
        net = VGG11(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'VGG11BN'):
        net = VGG11BN(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'VGG13'):
        net = VGG13(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'VGG13BN'):
        net = VGG13BN(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'VGG16'):
        net = VGG16(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'VGG19'):
        net = VGG19(channel=args.channel, num_classes=args.num_classes,time_step = args.time_step)
    elif(model_name == 'ResNet18BN'):
        net = ResNet18BN(channel = args.channel, num_classes=args.num_classes, time_step = args.time_step)
    elif(model_name == 'ResNet18'):
        net = ResNet18(channel = args.channel, num_classes=args.num_classes, time_step = args.time_step)
    elif(model_name == 'ResNet34'):
        net = ResNet34(channel = args.channel, num_classes=args.num_classes, time_step = args.time_step)
    elif(model_name == 'ResNet50'):
        net = ResNet50(channel = args.channel, num_classes=args.num_classes, time_step = args.time_step)
    elif(model_name == 'ResNet101'):
        net = ResNet101(channel = args.channel, num_classes=args.num_classes, time_step = args.time_step)
    elif(model_name == 'ResNet152'):
        net = ResNet152(channel = args.channel, num_classes=args.num_classes, time_step = args.time_step)
    elif(model_name == 'TCN'):
        if('dual' not in vars(args) or args.dual == 0):
            net = TCN(c_in=args.channel, c_out=args.num_classes, layers = 1 * [64])
        else:
            net = TCN(c_in=args.channel, c_out=args.num_classes, layers = 1 * [48])
    elif(model_name == 'Transformer'):
        if mode == 'sl':
            net = TransformerModel_sl(c_in=args.channel, c_out=args.num_classes, d_model = 64, num_shapelets = args.num_shapelets, sl_net = args.sl_net)
        else:
            net = TransformerModel(c_in=args.channel, c_out=args.num_classes, d_model = 64, num_shapelets = args.num_shapelets, sl_net = args.sl_net)
    elif(model_name == 'LSTM'):
        if('dual' not in vars(args) or args.dual == 0):
            net = LSTM(c_in=args.channel, c_out=args.num_classes, hidden_size = 100)
        else:
            net = LSTM(c_in=args.channel, c_out=args.num_classes, hidden_size = 64)
    elif(model_name == 'GRU'):
        if('dual' not in vars(args) or args.dual == 0):
            net = GRU(c_in=args.channel, c_out=args.num_classes, hidden_size = 100)
        else:
            net = GRU(c_in=args.channel, c_out=args.num_classes, hidden_size = 64)
    else:
        raise NotImplementedError
    return net


class _RNN_Base(nn.Module):
    def __init__(self, c_in, c_out, hidden_size=100, n_layers=1, bias=True, rnn_dropout=0, bidirectional=False, fc_dropout=0., init_weights=True):
        super(_RNN_Base, self).__init__()
        self.rnn = self._cell(c_in, hidden_size, num_layers=n_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.final_rep = hidden_size * (1 + bidirectional)
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), c_out)
        if init_weights: self.apply(self._weights_init)

    def forward(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        output = self.fc(self.dropout(output))
        return output
    def embed(self, x): 
        x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        return output
    
    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
        
class RNN(_RNN_Base):
    _cell = nn.RNN
    
class LSTM(_RNN_Base):
    _cell = nn.LSTM
    
class GRU(_RNN_Base):
    _cell = nn.GRU

from torch.nn.modules.transformer import My_TransformerEncoderLayer, My_TransformerEncoder


class TransformerModel(nn.Module):
    def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1, num_shapelets=10, sl_net=None):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
            
        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """
        super(TransformerModel,self).__init__()
        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        encoder_layer = My_TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, activation=activation)
        encoder_norm = nn.BatchNorm1d(d_model)        
        self.transformer_encoder = My_TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
        self.final_rep = d_model
        self.outlinear = nn.Linear(d_model, c_out)
        self.features = self.transformer_encoder.features
                        
    def forward(self,x):
        x = x.permute(2, 0, 1) # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x).permute(1, 0, 2) # seq_len x bs x d_model -> bs x seq_len x d_model
        x = x.max(1, keepdim=False)[0]
        x = self.relu(x)
        x = self.outlinear(x)
        return x
    def embed(self,x):
        x = x.permute(2, 0, 1) # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x).permute(1, 0, 2) # seq_len x bs x d_model -> bs x seq_len x d_model
        x = x.max(1, keepdim=False)[0]
        x = self.relu(x)
        return x

class TransformerModel_sl(nn.Module):
    def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="relu", n_layers=1, num_shapelets=10, sl_net=None):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
            c_out: the number of target classes
            d_model: total dimension of the model.
            nhead:  parallel attention heads.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
            
        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
            """
        super(TransformerModel_sl,self).__init__()
        self.inlinear = nn.Linear(c_in, d_model)
        self.relu = nn.ReLU()
        encoder_layer = My_TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, activation=activation)
        encoder_norm = nn.BatchNorm1d(d_model)        
        self.transformer_encoder = My_TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
        self.final_rep = d_model + num_shapelets * c_in
        self.outlinear = nn.Linear(self.final_rep, c_out)
        self.features = self.transformer_encoder.features
        self.sl_trans = sl_net.transform
        
    def forward(self,x):
        x_ = x
        x = x.permute(2, 0, 1) # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x).permute(1, 0, 2) # seq_len x bs x d_model -> bs x seq_len x d_model
        x = x.max(1, keepdim=False)[0]
        x = self.relu(x)
        sl_out = self.sl_trans(x_).reshape(x_.size(0), -1)
        x = torch.cat([sl_out, x], dim=1)
        x = self.outlinear(x)
        return x

    def embed(self,x):
        x = x.permute(2, 0, 1) # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x).permute(1, 0, 2) # seq_len x bs x d_model -> bs x seq_len x d_model
        x = x.max(1, keepdim=False)[0]
        x = self.relu(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(ni,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(nf,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, 
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(ni,nf,1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

def TemporalConvNet(c_in, layers, ks=2, dropout=0.):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i-1]
        nf = layers[i]
        temp_layers += [TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size, padding=(ks-1) * dilation_size, dropout=dropout)]
    return nn.Sequential(*temp_layers)

class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        
    def forward(self, x):
        return self.gap(x).reshape(x.shape[0], -1)

class TCN(nn.Module):
    def __init__(self, c_in, c_out, layers=4*[32], ks=7, conv_dropout=0., fc_dropout=0.):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1],c_out)
        self.init_weights()
        self.final_rep = layers[-1]
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)
    def embed(self,x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, hid2_dim, out_dim, num_shapelets, sl_net):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, hid2_dim)
        self.bn2 = nn.BatchNorm1d(hid2_dim, affine=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hid2_dim, out_dim)
        self.final_rep = hid2_dim
        self.features = [self.bn1, self.bn2]
    def forward(self, x):
        x = self.fc1(x.reshape(x.shape[0], -1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class MLP_sl(nn.Module):
    def __init__(self, input_dim, hid_dim, hid2_dim, out_dim, channel, num_shapelets, sl_net):
        super(MLP_sl, self).__init__()
        hid2_dim_t = hid2_dim - num_shapelets*channel
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid_dim, hid2_dim_t)
        self.bn2 = nn.BatchNorm1d(hid2_dim_t, affine=True)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hid2_dim, out_dim)
        self.final_rep = hid2_dim
        self.features = [self.bn1, self.bn2]
        self.sl_trans = sl_net.transform
    def forward(self, x):
        src = x
        sl_out = self.sl_trans(src).reshape(src.size(0), -1)
        x = self.fc1(x.reshape(x.shape[0], -1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = torch.cat([sl_out, x], dim=1)
        x = self.fc3(x)
        return x

    
''' ConvNet '''
class ConvNet_sl(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size, num_shapelets, sl_net):
        super(ConvNet_sl, self).__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size, num_shapelets)
        # self.sl_bn, _ = self._make_sl_bn(num_shapelets, channel, net_norm)
        num_feat = shape_feat[0]*shape_feat[1] + num_shapelets*channel
        # print("DEBUG, shape feat : {}".format(shape_feat))
        self.classifier = nn.Linear(num_feat, num_classes)
        self.final_rep = num_feat
        self.sl_trans = sl_net.transform

    def forward(self, x):
        out = self.features(x)
        # print("DEBUG, Real output : {}".format(out.shape))
        out = out.view(out.size(0), -1)
        sl_out = self.sl_trans(x)
        # temp = self.sl_bn(sl_out)
        sl_out = sl_out.reshape(sl_out.size(0), -1)
        out = torch.cat([sl_out, out], dim=1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        sl_out = self.sl_net(x)
        out = torch.cat([sl_out, out])
        return out
    
    #def embed_before_pool(self, x):
        #out = nn.Sequential(*list(self.features.children())[:-1])(x)
        #out = out.view(out.size(0), -1)
        #return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool1d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool1d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (C,L)
        if net_norm == 'BN':
            return nn.BatchNorm1d(shape_feat[0], affine=True)
        elif net_norm == 'LN':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'IN':
            return nn.InstanceNorm1d(shape_feat[0], affine=True)
        elif net_norm == 'GN':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size, num_shapelets):
        layers = []
        in_channels = channel
        shape_feat = [channel,im_size]
        for d in range(net_depth):
            if d == net_depth - 1:
                net_depth = net_depth - (num_shapelets * channel) // net_width
            layers += [nn.Conv1d(in_channels, net_width, kernel_size=3, padding=1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
        return nn.Sequential(*layers), shape_feat
    def _make_sl_bn(self, num_shapelets, channel, net_norm):
        layers = []
        if net_norm != 'none':
            layers += [self._get_normlayer(net_norm, [channel, num_shapelets])]
        return nn.Sequential(*layers), [channel, num_shapelets]
    
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size, num_shapelets, sl_net):
        super(ConvNet, self).__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]
        self.classifier = nn.Linear(num_feat, num_classes)
        self.final_rep = num_feat
    def forward(self, x):
        out = self.features(x)
        # print("DEBUG, Real output : {}".format(out.shape))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool1d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool1d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (C,L)
        if net_norm == 'BN':
            return nn.BatchNorm1d(shape_feat[0], affine=True)
        elif net_norm == 'LN':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'IN':
            return nn.InstanceNorm1d(shape_feat[0], affine=True)
        elif net_norm == 'GN':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        shape_feat = [channel,im_size]
        for d in range(net_depth):
            layers += [nn.Conv1d(in_channels, net_width, kernel_size=3, padding=1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
        return nn.Sequential(*layers), shape_feat
    



''' LeNet '''
class LeNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=(5, 1), padding=2 if channel==1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def embed(self, x):
        if(x.dim() == 3):
            x = x.unsqueeze(-1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
    def forward(self, x):
        if(x.dim() == 3):
            x = x.unsqueeze(-1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


''' AlexNet '''
class AlexNet(nn.Module):
    def __init__(self, channel, num_classes, time_step):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=(5, 1), stride=1, padding=4 if channel==1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
            nn.Conv2d(128, 192, kernel_size=(5, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
            nn.Conv2d(192, 256, kernel_size=(3, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=(3, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
        )

        k_h = [5, 2, 5 ,2 ,3 ,3, 3, 2]
        k_w = [1, 1, 1, 1, 1, 1, 1, 1]
        stride = [1, 2, 1, 2, 1, 1, 1, 2]        
        padding = [4, 0, 2, 0, 1, 1, 1, 0]
        if(channel==1):
            padding[0] = 4
        else:
            padding[0] = 2
        final_chanel = 192
        cur_size = [time_step, 1]
        for idx, h in enumerate(k_h):
            w = k_w[idx]
            strid = stride[idx]
            pad = padding[idx]
            
            cur_size[0] = (cur_size[0] + 2 * pad - (h - 1) - 1) // strid + 1
            cur_size[1] = (cur_size[1] + 2 * pad - (w - 1) - 1) // strid + 1
        self.final_rep = cur_size[0] * cur_size[1] * final_chanel
        self.fc = nn.Linear(self.final_rep, num_classes)
        
    def forward(self, x):
        if(x.dim() == 3):
            x.unsqueeze(-1)
        x = self.features(x)
        print('feature size = : {}'.format(x.shape))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        if(x.dim() == 3):
            x.unsqueeze(-1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


''' AlexNetBN '''
class AlexNetBN(nn.Module):
    def __init__(self, channel, num_classes, time_step = 128):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=(5, 1), stride=1, padding=4 if channel==1 else 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
            nn.Conv2d(128, 192, kernel_size=(5, 1), padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
            nn.Conv2d(192, 256, kernel_size=(3, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=(3, 1), padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 1), padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
        )
        
        k_h = [5, 2, 5 ,2 ,3 ,3, 3, 2]
        k_w = [1, 1, 1, 1, 1, 1, 1, 1]
        stride = [1, 2, 1, 2, 1, 1, 1, 2]        
        padding = [4, 0, 2, 0, 1, 1, 1, 0]
        if(channel==1):
            padding[0] = 4
        else:
            padding[0] = 2
        final_chanel = 192
        cur_size = [time_step, 1]
        for idx, h in enumerate(k_h):
            w = k_w[idx]
            strid = stride[idx]
            pad = padding[idx]
            
            cur_size[0] = (cur_size[0] + 2 * pad - (h - 1) - 1) // strid + 1
            cur_size[1] = (cur_size[1] + 2 * pad - (w - 1) - 1) // strid + 1
        self.final_rep = cur_size[0] * cur_size[1] * final_chanel
        self.fc = nn.Linear(self.final_rep, num_classes)
    def forward(self, x):
        if(x.dim() == 3):
            x.unsqueeze(-1)
        x = self.features(x)
        print('feature size = : {}'.format(x.shape))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def embed(self, x):
        if(x.dim() == 3):
            x.unsqueeze(-1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


''' VGG '''
cfg_vgg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, time_step, norm='instancenorm'):
        super(VGG, self).__init__()
        self.channel = channel
        self.num_classes = num_classes
        self.time_step = time_step
        self.features, _ = self._make_layers(cfg_vgg[vgg_name], norm)
        
        # self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)
        self.classifier = nn.Linear(self.final_rep, num_classes)

    def forward(self, x):
        if(x.dim() == 3):
            x.unsqueeze(-1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def embed(self, x):
        if(x.dim() == 3):
            x.unsqueeze(-1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_layers(self, cfg, norm):
        layers = []
        k_h, k_w, stride, padding = [], [], [], []

        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 1), stride=2)]
                k_h.append(2)
                k_w.append(1)
                stride.append(2)
                padding.append(0)
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=(3, 1), padding=3 if self.channel==1 and ic==0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm=='instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                k_h+=[3]
                k_w+=[1]
                if(self.channel == 1 and ic==0):
                    padding += [3]
                else:
                    padding += [1]
                stride += [1]
                
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=(1, 1), stride=1)]
        k_w += [1]
        k_h += [1]
        stride += [1]
        padding += [0]
        
        
        
        final_chanel = in_channels
        cur_size = [self.time_step, 1]
        for idx, h in enumerate(k_h):
            w = k_w[idx]
            strid = stride[idx]
            pad = padding[idx]
            
            cur_size[0] = (cur_size[0] + 2 * pad - (h - 1) - 1) // strid + 1
            cur_size[1] = (cur_size[1] + 2 * pad - (w - 1) - 1) // strid + 1
        self.final_rep = cur_size[0] * cur_size[1] * final_chanel
        
        return nn.Sequential(*layers), self.final_rep


def VGG11(channel, num_classes, time_step):
    return VGG('VGG11', channel, num_classes, time_step)
def VGG11BN(channel, num_classes, time_step):
    return VGG('VGG11', channel, num_classes, time_step, norm='batchnorm')
def VGG13(channel, num_classes, time_step):
    return VGG('VGG13', channel, num_classes, time_step)
def VGG13BN(channel, num_classes, time_step):
    return VGG('VGG13', channel, num_classes, time_step, norm = 'batchnorm')
def VGG16(channel, num_classes, time_step):
    return VGG('VGG16', channel, num_classes, time_step)
def VGG19(channel, num_classes, time_step):
    return VGG('VGG19', channel, num_classes, time_step)


''' ResNet_AP '''
# The conv(stride=2) is replaced by conv(stride=1) + avgpool(kernel_size=(2, 1), stride=2)

class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 1), stride=1, padding=1, bias=False) # modification
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1, 1), stride=1, bias=False),
                nn.AvgPool2d(kernel_size=(2, 1), stride=2), # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=(2, 1), stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_AP(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=1, padding=1, bias=False) # modification
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1, 1), stride=1, bias=False),
                nn.AvgPool2d(kernel_size=(2, 1), stride=2),  # modification
                nn.GroupNorm(self.expansion * planes, self.expansion * planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.stride != 1: # modification
            out = F.avg_pool2d(out, kernel_size=(2, 1), stride=2)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_AP(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm'):
        super(ResNet_AP, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=(3, 1), stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion * 3 * 3 if channel==1 else 512 * block.expansion * 4 * 4, num_classes)  # modification

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=(1, 1), stride=1) # modification
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, kernel_size=(1, 1), stride=1) # modification
        out = out.view(out.size(0), -1)
        return out

def ResNet18BN_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes, norm='batchnorm')

def ResNet18_AP(channel, num_classes):
    return ResNet_AP(BasicBlock_AP, [2,2,2,2], channel=channel, num_classes=num_classes)


''' ResNet '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 1), stride=stride, padding=(1,0), bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=1, padding=(1,0), bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=(1, 1), stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if(x.dim() == 3):
            x = x.unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=stride, padding=(1,0), bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=(1, 1), stride=stride, bias=False),
                nn.GroupNorm(self.expansion*planes, self.expansion*planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        if(x.dim() == 3):
            x = x.unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10,time_step = 128, norm='instancenorm'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm
 
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=(3, 1), stride=1, padding=(1,0), bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512*block.expansion * 4, num_classes)
        self.final_rep = 512*block.expansion * 4
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers)

    def forward(self, x):
        if(x.dim() == 3):
            x = x.unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,(4,1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        if(x.dim() == 3):
            x = x.unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, (4,1))
        
        out = out.view(out.size(0), -1)

        return out


def ResNet18BN(channel, num_classes, time_step):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, time_step =time_step,norm='batchnorm')

def ResNet18(channel, num_classes, time_step):
    return ResNet(BasicBlock, [2,2,2,2], channel=channel, num_classes=num_classes, time_step =time_step)

def ResNet34(channel, num_classes, time_step):
    return ResNet(BasicBlock, [3,4,6,3], channel=channel, num_classes=num_classes, time_step =time_step)

def ResNet50(channel, num_classes, time_step):
    return ResNet(Bottleneck, [3,4,6,3], channel=channel, num_classes=num_classes, time_step =time_step)

def ResNet101(channel, num_classes, time_step):
    return ResNet(Bottleneck, [3,4,23,3], channel=channel, num_classes=num_classes, time_step =time_step)

def ResNet152(channel, num_classes, time_step):
    return ResNet(Bottleneck, [3,8,36,3], channel=channel, num_classes=num_classes, time_step =time_step)

class ConvNet2D(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet2D, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def embed_before_pool(self, x):
        out = nn.Sequential(*list(self.features.children())[:-1])(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=(2, 1), stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=(2, 1), stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'BN':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'LN':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'IN':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'GN':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=(3, 1), padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

''' MLP '''
# class MLP(nn.Module):
#     def __init__(self, channel, num_classes, time_step):
#         super(MLP, self).__init__()
#         self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
#         self.fc_2 = nn.Linear(128, 128)
#         self.fc_3 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         out = x.view(x.size(0), -1)
#         out = F.relu(self.fc_1(out))
#         out = F.relu(self.fc_2(out))
#         out = self.fc_3(out)
#         return out