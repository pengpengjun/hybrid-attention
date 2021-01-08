#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import time
import pynvml
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# In[2]:


def hybrid_attention():
    
    # data read
    
    x_train=np.load(r'G:/Trajectory Prediction/data/x_train.npy')
    x_test=np.load(r'G:/Trajectory Prediction/data/x_test.npy')
    x_validation=np.load(r'G:/Trajectory Prediction/data/x_validation.npy')
    y_train=np.load(r'G:/Trajectory Prediction/data/y_train.npy')
    y_test=np.load(r'G:/Trajectory Prediction/data/y_test.npy')
    y_validation=np.load(r'G:/Trajectory Prediction/data/y_validation.npy')
    
    '''
    x_train=np.load(r'G:/Trajectory Prediction/data/x_test.npy')
    x_test=np.load(r'G:/Trajectory Prediction/data/x_test.npy')
    x_validation=np.load(r'G:/Trajectory Prediction/data/x_test.npy')
    y_train=np.load(r'G:/Trajectory Prediction/data/y_test.npy')
    y_test=np.load(r'G:/Trajectory Prediction/data/y_test.npy')
    y_validation=np.load(r'G:/Trajectory Prediction/data/y_test.npy')    
    '''
    
    #data standard normalization
    a,b,c=x_train.shape
    x_train=x_train.reshape(a*b,c)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    x_train=x_train.reshape(a,b,c)

    a,b,c=x_validation.shape
    x_validation=x_validation.reshape(a*b,c)
    x_validation=scaler.transform(x_validation)
    x_validation=x_validation.reshape(a,b,c)

    a,b,c=x_test.shape
    x_test=x_test.reshape(a*b,c)
    x_test=scaler.transform(x_test)
    x_test=x_test.reshape(a,b,c)

    x1=torch.from_numpy(x_train).float()
    y1=torch.from_numpy(y_train).float()
    x2=torch.from_numpy(x_validation).float()
    y2=torch.from_numpy(y_validation).float()
    x3=torch.from_numpy(x_test).float()
    y3=torch.from_numpy(y_test).float()
    
    #data from.npy to pytorch data

    global BATCH_SIZE
    BATCH_SIZE=512


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Data.TensorDataset(x1,y1)

    trainloader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False, # 要不要打乱数据 (打乱比较好)
        num_workers=0,              # 多线程来读数据
        drop_last=True,
    )



    vali_dataset = Data.TensorDataset(x2,y2)

    valiloader = Data.DataLoader(
        dataset=vali_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False, # 要不要打乱数据 (打乱比较好)
        num_workers=0,              # 多线程来读数据
        drop_last=True,
    )


    test_dataset = Data.TensorDataset(x3,y3)

    testloader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False, # 要不要打乱数据 (打乱比较好)
        num_workers=0,              # 多线程来读数据
        drop_last=True,
    )

    
#Encoder structure

    class Encoder(nn.Module):
        def __init__(self, input_dim,emb_dim, hid_dim, n_layers, dropout=0):
            #n_layers represents the layer of LSTM
            
            
            super().__init__()

            self.input_dim = input_dim
            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = nn.Linear(input_dim, emb_dim)

            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

            self.dropout = nn.Dropout(dropout)

        def forward(self, x):

            #x = [src sent len, batch size,2]

            embedded = self.dropout(self.embedding(x))

            #embedded = [src sent len, batch size, emb dim]

            outputs, (hidden, cell) = self.rnn(embedded)

            #outputs = [len, batch size, hid dim * n directions]
            #direction==1 hier
            #hidden = [n layers * n directions, batch size, hid dim]
            #cell = [n layers * n directions, batch size, hid dim]

            #outputs are always from the top hidden layer

            return outputs,hidden, cell

        
#Decoder structure

    class Decoder(nn.Module):
        def __init__(self, decoder_input_dim,emb_dim, hid_dim, n_layers, dropout=0.2):
            super().__init__()

            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.decoder_input_dim = decoder_input_dim
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = nn.Linear(decoder_input_dim, emb_dim)


            self.attn = nn.Linear(240 + 80 + self.emb_dim, 1) #为什么要把attention的特征量压成一个值？为什么不能和context_attention用一样的办法呢？压成一列不会损失太多信息吗？
    #         self.attn_combine = nn.Linear(self.hidden_size+self.de_emb_dim , self.hidden_size)

            self.rnn = nn.LSTM(emb_dim+hid_dim, hid_dim, n_layers, dropout = dropout)

            self.out = nn.Linear(hid_dim, decoder_input_dim)

            self.dropout = nn.Dropout(dropout)

        def forward(self, input,context1, context2,context3,context_selfattn,hidden, cell):

            #hidden = [n layers * n directions, batch size, hid dim]
            #cell = [n layers * n directions, batch size, hid dim]

            #n directions in the decoder will both always be 1, therefore:
            #hidden = [n layers, batch size, hid dim]
            #context = [n layers, batch size, hid dim]
            input = input.unsqueeze(0)
            embedded = self.dropout(self.embedding(input))
            context1=context1.permute(1, 0, 2) #(512,1,40)
            context2=context2.permute(1, 0, 2)
            context3=context3.permute(1, 0, 2)
    #         context=[batch,1,hidden_dim]
            embedded=embedded.permute(1,0,2)

    #          embedded[batch,1,embedded_dim]
            #print('embedded.shape is:\n',embedded.shape,'\n\n')
            #print('context1.shape is:\n',context1.shape,'\n\n')
            #print('context1.shape is:\n',context_selfattn.shape,'\n\n')
            attn1=self.attn(torch.cat((embedded, context1,context_selfattn),dim=2))#为什么要用同一个网络，三个不应该用不同的网络吗？
            attn2=self.attn(torch.cat((embedded, context2,context_selfattn),dim=2))
            attn3=self.attn(torch.cat((embedded, context3,context_selfattn),dim=2))

    #         attn1=np.array(attn1)
            attn1=attn1.squeeze(1)
            attn2=attn2.squeeze(1)
            attn3=attn3.squeeze(1)
            #print(attn1.size())
            attn=torch.cat((attn1,attn2,attn3),1) #(512,3)
            #print(attn.size())
            attn_weights=F.softmax(attn,dim=1)
            #print(attn_weights.size())
            #print(attn_weights[200,:])
            attn_weights=attn_weights.unsqueeze(1)
            context1=context1.permute(0, 2, 1)#(512,40,1)
            context2=context2.permute(0, 2, 1)
            context3=context3.permute(0, 2, 1)

            #print(context1.shape)
            #print(attn_weights[:,:,0].shape)
            context1=torch.bmm(context1,attn_weights[:,:,0].unsqueeze(2)) #这里是(512,40,1)乘上(512,1,1)相当于batch中每帧只乘上了一个值而已。为什么不能用和之前context_attention一样的做法呢？
            context2=torch.bmm(context2,attn_weights[:,:,1].unsqueeze(2))
            context3=torch.bmm(context3,attn_weights[:,:,2].unsqueeze(2))

            context1=context1.permute(0, 2, 1)
            context2=context2.permute(0, 2, 1)
            #permute change the dimesion between the last and the middel positiom
            context3=context3.permute(0, 2, 1)


            context=torch.cat((context1,context2,context3),dim=2)
            #print(context.shape)

            embedded=embedded.permute(1,0,2)
            context=context.permute(1, 0, 2)

            emb_con = torch.cat((embedded, context), 2)


            output, (hidden, cell) = self.rnn(emb_con, (hidden, cell))

            #output = [len, batch size, hid dim * n directions]
            #hidden = [n layers * n directions, batch size, hid dim]
            #cell = [n layers * n directions, batch size, hid dim]

            #sent len and n directions will always be 1 in the decoder, therefore:
            #output = [1, batch size, hid dim]
            #hidden = [n layers, batch size, hid dim]
            #cell = [n layers, batch size, hid dim]
            #print("the size of output is:\n",output.shape,'\n\n')
            prediction = self.out(output.squeeze(0))
            #print("the size of prediction is:\n",prediction.shape,'\n\n')

            #prediction = [batch size, output dim]

            return prediction, hidden, cell

    class Multihead_Attention(nn.Module):
        def __init__(self,enc_hidden_dim,attn_emb_dim):
            super().__init__()
            self.enc_hidden_dim = enc_hidden_dim
            self.attn_emb_dim = attn_emb_dim

            self.q_linear_1 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.k_linear_1 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.v_linear_1 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)

            self.q_linear_2 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.k_linear_2 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.v_linear_2 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)

            self.q_linear_3 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.k_linear_3 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.v_linear_3 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)

            self.q_linear_4 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.k_linear_4 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)
            self.v_linear_4 = nn.Linear(self.enc_hidden_dim, self.attn_emb_dim)

            self.fusion = nn.Linear(self.attn_emb_dim * 4, self.attn_emb_dim)
        
        def forward(self,tar_self,tar_sur):

            def attn_cal(q,k,v):
                d_k = q.shape[-1]
                #print('d_k is:\n',d_k,'\n\n')
                product = torch.bmm(q,k.transpose(-1,-2))/np.sqrt(d_k)
                attn = F.softmax(product,dim=-1)
                return (attn.transpose(-1,-2)*v).sum(axis=1)

            q_1 = self.q_linear_1(tar_self)
            k_1 = self.k_linear_1(tar_sur)
            v_1 = self.v_linear_1(tar_sur)
            context_1 = attn_cal(q_1, k_1, v_1)
        
            q_2 = self.q_linear_2(tar_self)
            k_2 = self.k_linear_2(tar_sur)
            v_2 = self.v_linear_2(tar_sur)
            context_2 = attn_cal(q_2, k_2, v_2)
        
            q_3 = self.q_linear_3(tar_self)
            k_3 = self.k_linear_3(tar_sur)
            v_3 = self.v_linear_3(tar_sur)
            context_3 = attn_cal(q_3, k_3, v_3)

            q_4 = self.q_linear_4(tar_self)
            k_4 = self.k_linear_4(tar_sur)
            v_4 = self.v_linear_4(tar_sur)
            context_4 = attn_cal(q_4, k_4, v_4)

            context = torch.cat((context_1,context_2,context_3,context_4),dim=-1)
            return self.fusion(context)
        
# seq-seq combine encoder-decoder and 
#the decoder process is done step by step 
        
    class Seq2Seq(nn.Module):
        global firstinput
        def __init__(self, encoder , encoder_selfattn, attn, decoder, device):
            super().__init__()

            self.encoder = encoder
            self.decoder = decoder
            self.device = device
            self.encoder_selfattn = encoder_selfattn
            self.muti_attn = attn

            #assert encoder.hid_dim == decoder.hid_dim,             "Hidden dimensions of encoder and decoder must be equal!"
            assert encoder.n_layers == decoder.n_layers,             "Encoder and decoder must have equal number of layers!"

        def forward(self, x, y, teacher_forcing_ratio = 0.5):

            #teacher_forcing_ratio is probability to use teacher forcing
            #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

            batch_size = BATCH_SIZE
            max_len = 25
            trg_vocab_size = 2

            #tensor to store decoder outputs
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
            

            #lane attention--------------------------------------------------------------------------------------------------------------------------------------
            #last hidden state of the encoder is used as the initial hidden state of the decoder
            #choose the cell state of second layer as context 
            encoder_outputs1,hidden1, cell1 = self.encoder(x[:,:,[x for x in range(0,21)]]) #为什么要用同一个encoder，不能用不同的encoder吗？
            #print("the size of x[:,:,[x for x in range(0,21)]] is:",x[:,:,[x for x in range(0,21)]].shape)
            encoder_outputs2,hidden2, cell2 = self.encoder(x[:,:,[x for x in range(21,42)]])
            encoder_outputs3,hidden3, cell3 = self.encoder(x[:,:,[x for x in range(42,63)]])
            #take the cell of last layer as context
            context1=cell1[1,:,:]
            context1=context1.unsqueeze(0)
            context2=cell2[1,:,:]
            context2=context2.unsqueeze(0)
            context3=cell3[1,:,:]
            context3=context3.unsqueeze(0)
            hidden=torch.cat((hidden1,hidden2,hidden3),dim=2)
            cell=torch.cat((cell1,cell2,cell3),dim=2)
            print("the size of context1",context1.shape)
            
            #self attention---------------------------------------------------------------------------------------------------------------------------------------
            cell_selfattns = []
            #print('the shape of x[:,:,[x for x in range(0+i*7,7+i*7)]] is:\n',x[:,:,[x for x in range(0,7)]].shape,'\n\n')
            for i in range(0,9):
                encoder_output,hidden_selfattn, cell_selfattn = self.encoder_selfattn(x[:,:,[x for x in range(0+i*7,7+i*7)]]) 
                #print('the shape of hidden_selfattn is:',hidden_selfattn.shape,'\n\n') #[2, 512, 32]
                cell_selfattns.append(cell_selfattn[1,:,:])
            
            tar_self = cell_selfattns[0].unsqueeze(1)
            #print('the shape of tar_self is:\n',tar_self.shape,'\n\n')
            tar_sur = cell_selfattns[1].unsqueeze(1)
            
            for cell_selfattn in cell_selfattns[2:]:
                cell_selfattn = cell_selfattn.unsqueeze(1)
                tar_sur = torch.cat((tar_sur,cell_selfattn),1) 
            #print('the shape of tar_sur is:',tar_sur.shape,'\n\n')   #[512, 8, 32]
            #print('the shape of encoder_output is:\n',encoder_outputs[0].shape,'\n\n')   #[15, 512, 32]
            #print('the shape of hidden_selfattn is:\n',hidden_selfattns[0].shape,'\n\n')   #[2, 512, 32]
            #print('the shape of cell_selfattn is:\n',cell_selfattns[0].shape,'\n\n')   #[2, 512, 32]

            context_selfattn = self.muti_attn(tar_self, tar_sur)
            context_selfattn = context_selfattn.unsqueeze(1)           

            #cell=cell1+cell2+cell3
    #         print('c-shape:',context.shape)
    
            input=firstinput
            #the firstinput of decoder process, we should give the true vaule
            # input = input.unsqueeze(0)
            #print(input.size())
            for t in range(max_len):

                output, hidden, cell = self.decoder(input, context1,context2,context3,context_selfattn,hidden, cell)
                outputs[t] = output
                #print(output)
                #input = output.unsqueeze(0)
                #print(input.size())
                #context=cell[1,:,:]
                #context=context.unsqueeze(0)
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output
                if t==24:
                    break
                input = ((y[t,:,:]) if teacher_force else top1)
                #outputs[t] = output
                #print('output',output.size())
                #input = output.unsqueeze(0)


            return outputs
        
    INPUT_DIM =21
    DECODER_INPUT_DIM = 2
    HID_DIM = 80
    HID_DIM1 = 240
    N_LAYERS = 2
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 32

    INPUT_DIM_SELF = 7
    ENC_EMB_DIM_SELF = 32
    HID_DIM_SELF = 32

    attn = Multihead_Attention(HID_DIM_SELF,HID_DIM1)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM , HID_DIM, N_LAYERS)
    enc_selfattn = Encoder(INPUT_DIM_SELF, ENC_EMB_DIM_SELF , HID_DIM_SELF, N_LAYERS)
    dec = Decoder(DECODER_INPUT_DIM,DEC_EMB_DIM , HID_DIM1, N_LAYERS)

    model = Seq2Seq(enc, enc_selfattn, attn, dec, device).to(device)


    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.15, 0.15)
    #         nn.init.orthogonal_(param.data)

    model.apply(init_weights)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')


    optimizer = optim.Adam(model.parameters(),weight_decay=0.00001,lr=0.01)
    criterion = nn.MSELoss()

    
    def train(model, dataloader,optimizer, criterion, clip):
        global firstinput
        model.train()

        epoch_loss = 0

        for x,y in dataloader:

            x=x.transpose(1,0)
            y=y.transpose(1,0)
            x=x.to('cuda')
            y=y.to('cuda')
            firstinput=y[0,:,:]
            y=y[1:,:,:]
            optimizer.zero_grad()

            output = model(x, y)
            output = output.to('cuda')


    #         loss = criterion(output, y)
            #print(output.size())
            # for lateral position,we give more attention,so his penalization is *3
            loss = 3*criterion(output[:,:,1],y[:,:,1])+criterion(output[:,:,0],y[:,:,0])
            loss.backward()

            #print(type(loss))

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
            #print(epoch_loss)
            loss.detach()

        return epoch_loss/len(dataloader)


    # In[ ]:


    def evaluate(model, validataloader, criterion):

        model.eval()

        epoch_loss = 0
        # no loss backward,
        with torch.no_grad():

            for x,y in validataloader:

                x=x.transpose(1,0)
                y=y.transpose(1,0)
                x=x.to('cuda')
                y=y.to('cuda')
                firstinput=y[0,:,:]
                y=y[1:,:,:]
                optimizer.zero_grad()

                output = model(x, y, 0) #turn off teacher forcing
                output = output.to('cuda')


                loss = 3*criterion(output[:,:,1],y[:,:,1])+criterion(output[:,:,0],y[:,:,0])
                epoch_loss += loss.item()


        return epoch_loss / len(validataloader)


    # In[ ]:


    def test(model, testdataloader, criterion):
        global j
        global firstinput
        global test_result
        model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for x,y in testdataloader:

                x=x.transpose(1,0)
                y=y.transpose(1,0)
                x=x.to('cuda')
                y=y.to('cuda')
                firstinput=y[0,:,:]
                y=y[1:,:,:]
                optimizer.zero_grad()

                output = model(x, y, 0) #turn off teacher forcing
                test_result[:,j:j+BATCH_SIZE,:]=output
                #print('the shape of ouput is:\n',output.shape,'\n\n')
                #print('the shape of test_result is:\n',test_result[:,j:j+BATCH_SIZE,:].shape,'\n\n')
                j=j+BATCH_SIZE
                #print('j is:\n',j,'\n\n')
                output = output.to('cuda')


    #             loss = criterion(output, y)
                loss = 3*criterion(output[:,:,1],y[:,:,1])+criterion(output[:,:,0],y[:,:,0])
                epoch_loss += loss.item()

    #     print(len(testdataloader))

        return epoch_loss / len(testdataloader)


    # In[ ]:


    N_EPOCHS =160
    CLIP = 1
    #CLIP clip the gradients to prevent them from exploding
    global test_result
    test_loss_min = 50000
    test_result=np.zeros([25,110000,2])
    test_result_min=np.zeros([25,110000,2])
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('直接在lane attention 中 cat 上multiattention的内容\n')
    if __name__ == '__main__':
        for epoch in range(N_EPOCHS):
            global j
            j=0
            start_time = time.process_time()
            train_loss = train(model, trainloader, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valiloader, criterion)
            end_time = time.process_time()
            print(f'Epoch: {epoch+1:02} | Time: {end_time-start_time}s')
            print(f'\tTrain Loss: {train_loss:.3f} |  Val. Loss: {valid_loss:.3f}')

            test_loss = test(model, testloader, criterion)
            

            print(f'\tTest Loss: {test_loss:.3f}','\n')

            if test_loss < test_loss_min:
                test_loss_min = test_loss
                test_result_min[:,:j,:] = test_result[:,:j,:]
            
            print('the min test loss is:',test_loss_min,'\n')
            print('-----------------------------------------------')


            if epoch ==159:
                print('the min test loss is:',test_loss_min)
                #test_result=test_result[:,:j,:]
                np.save(r'F:/Jupyter_Workspace/Spatial Attention Mechanism for Vehicle Trajectory/Spatial-attention-master/Encoder-Decoder-Net/result/lane_directplus_car_attention.npy',test_result_min[:,:j,:])
                np.save(r'F:/Jupyter_Workspace/Spatial Attention Mechanism for Vehicle Trajectory/Spatial-attention-master/Encoder-Decoder-Net/result/true_tra_highdmodi.npy',y_test[:,1:,:])
                break
    print('meminfo.used:',meminfo.used/(1024*1024))
    print('meminfo.total:',meminfo.total/(1024*1024))
    
    return 0

hybrid_attention()




# %%
