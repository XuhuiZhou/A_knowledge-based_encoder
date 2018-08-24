# -*- coding: utf-8 -*-

from torch import nn
from torch import autograd
import numpy as np
import torch
import sys
import os 
import torch.nn.functional as F

#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#rnn = SLSTM(256*2, dropout=0.33, step= 1, gpu = True)
#rnn(input,mask,num_layers=self.num_layers)

class SLSTM_SW(nn.Module):

    def __init__(self, word_dim, hidden_size,dropout,step, gpu):
        #current
        super(SLSTM_SW, self).__init__()
        #self.config = config
        self.hidden_size = hidden_size
        self.gpu = gpu
        self.step = step
        self.word_dim = word_dim

        # forget gate for left
        self.Wxf1, self.Whf1, self.Wif1, self.Wdf1, self.Wsf1 = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)
        #forget gate for right
        self.Wxf2, self.Whf2, self.Wif2, self.Wdf2, self.Wsf2 = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)
        #forget gate for inital states
        self.Wxf3, self.Whf3, self.Wif3, self.Wdf3, self.Wsf3 = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)
        #forget gate for dummy states
        self.Wxf4, self.Whf4, self.Wif4, self.Wdf4, self.Wsf4 = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)
        #input gate for current state
        self.Wxi, self.Whi, self.Wii, self.Wdi, self.Wsi = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)
        #input gate for output gate
        self.Wxo, self.Who, self.Wio, self.Wdo, self.Wso = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)
        #actual input 
        self.Wxu, self.Whu, self.Wiu, self.Wdu, self.Wsu = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)


        self.bi = self.create_bias_variable(hidden_size, self.training, gpu)
        self.bo = self.create_bias_variable(hidden_size, self.training, gpu)
        self.bf1 = self.create_bias_variable(hidden_size, self.training, gpu)
        self.bf2 = self.create_bias_variable(hidden_size, self.training, gpu)
        self.bf3 = self.create_bias_variable(hidden_size, self.training, gpu)
        self.bf4 = self.create_bias_variable(hidden_size, self.training, gpu)
        self.bu = self.create_bias_variable(hidden_size, self.training, gpu)

        self.gated_Wxd = self.create_to_hidden_variable(hidden_size, hidden_size, self.training, gpu)
        self.gated_Whd = self.create_to_hidden_variable(hidden_size, hidden_size, self.training, gpu)

        self.gated_Wxo = self.create_to_hidden_variable(hidden_size, hidden_size, self.training, gpu)
        self.gated_Who = self.create_to_hidden_variable(hidden_size, hidden_size, self.training, gpu)

        self.gated_Wxf = self.create_to_hidden_variable(hidden_size, hidden_size, self.training, gpu)
        self.gated_Whf = self.create_to_hidden_variable(hidden_size, hidden_size, self.training, gpu)

        self.gated_bd = self.create_bias_variable(hidden_size, self.training, gpu)
        self.gated_bo = self.create_bias_variable(hidden_size, self.training, gpu)
        self.gated_bf = self.create_bias_variable(hidden_size, self.training, gpu)

        ##parameter for switch
        self.Wxf5, self.Whf5, self.Wif5, self.Wdf5, self.Wsf5 = self.create_a_lstm_gate(hidden_size,word_dim,self.training, gpu)
        #self.Wsf5 = self.create_to_hidden_variable(hidden_size, hidden_size, self.training, gpu)
        self.bf5 = self.create_bias_variable(hidden_size, self.training, gpu)

        self.h_drop = nn.Dropout(dropout)
        self.c_drop = nn.Dropout(dropout)
        if gpu:
            self.h_drop = self.h_drop.cuda()
            self.c_drop = self.c_drop.cuda()

    def create_bias_variable(self, size, is_training, gpu=False, mean=0.0, stddev=0.1):
        data = torch.zeros(size)
        if gpu:
            data = data.cuda()
        var = nn.Parameter(data, requires_grad=True)
        var.data.normal_(mean, std=stddev)
        # torch.nn.init.kaiming_uniform(var)
        #if gpu:
        #    var = var.cuda()
        return var

    def create_to_hidden_variable(self, size1, size2, is_training, gpu=False, mean=0.0, stddev=0.1):
        data = torch.zeros((size1, size2))
        if gpu:
            data = data.cuda()
        var = nn.Parameter(data, requires_grad=True)
        var.data.normal_(mean, std=stddev)
        # torch.nn.init.xavier_normal(var)
        #if gpu:
        #    var = var.cuda()
        return var

    def create_a_lstm_gate(self, hidden_size, word_dim,is_training, gpu=False, mean=0.0, stddev=0.1):

        wxf = self.create_to_hidden_variable(hidden_size, hidden_size, is_training, gpu, mean, stddev)
        whf = self.create_to_hidden_variable(2 * hidden_size, hidden_size, is_training, gpu, mean, stddev)
        wif = self.create_to_hidden_variable(word_dim, hidden_size, is_training, gpu, mean, stddev)
        wdf = self.create_to_hidden_variable(hidden_size, hidden_size, is_training, gpu, mean, stddev)
        wsf = self.create_to_hidden_variable(hidden_size, hidden_size, is_training, gpu, mean, stddev)

        return wxf, whf, wif, wdf, wsf

    def create_nograd_variable(self, minval, maxval, is_training, gpu, *shape):
        data = torch.zeros(*shape)
        if gpu:
            data = data.cuda()
        var = autograd.Variable(data, volatile=not is_training, requires_grad=False)
        var.data.uniform_(minval, maxval)
        # torch.nn.init.xavier_normal(var)
        #if gpu:
        #    var = var.cuda()
        return var

    def create_padding_variable(self, is_training, gpu, *shape):
        data = torch.zeros(*shape)
        if gpu:
            data = data.cuda()
        var = autograd.Variable(data, volatile=not is_training, requires_grad=False)
        #if gpu:
        #    var = var.cuda()
        return var


    def get_hidden_states_before(self, padding, hidden_states, step):
        #padding zeros
        #padding = create_padding_variable(self.training, self.config.HP_gpu, (shape[0], step, hidden_size))
        
        #print(hidden_states.size())
        if step < hidden_states.size()[1]: 
            #remove last steps
            displaced_hidden_states = hidden_states[:, :-step, :]
            #concat padding
            #return torch.cat([padding]*hidden_states.size()[1], dim=1)
            return torch.cat([padding, displaced_hidden_states], dim=1)
        else:
            return padding



    def get_hidden_states_after(self, padding, hidden_states, step):
        #padding zeros
        #padding = create_padding_variable(self.training, self.config.HP_gpu, (shape[0], step, hidden_size))
        #remove last steps
        if step < hidden_states.size()[1]:
            displaced_hidden_states = hidden_states[:, step:, :]
            #concat padding
            return torch.cat([displaced_hidden_states, padding], dim=1)
        else:
            return torch.cat([padding]*hidden_states.size()[1], dim=1)

    def sum_together(self, l):
        return sum(l)

    ##enable knowledge comes in
    def position_switch(self, hidden_states, position_table):
        #print(position_table)
        #position_table = torch.from_numpy(position_table)
        size = hidden_states.shape
        pad = self.create_padding_variable(self.training, self.gpu, (size[0], 1, size[2]))
        #print(hidden_states, pad)
        hidden_states_add = torch.cat([pad, hidden_states], dim = 1)
        position_table = position_table.unsqueeze(dim = 2)
        position_table = position_table.expand_as(hidden_states)
        position_table = autograd.Variable(position_table.cuda(), volatile=not self.training, requires_grad=False)
        switch_states = torch.gather(hidden_states_add,1,position_table)
        return switch_states

        

    def forward(self, word_inputs, position_table, mask, num_layers):

        # filters for attention
        #print("[tlog] mask: " + str(mask))
        mask_softmax_score = mask.float() * 1e25 - 1e25  # 10, 40
        #print("[tlog] mask_softmax_score: " + str(mask_softmax_score))

        mask_softmax_score_expanded = torch.unsqueeze(mask_softmax_score, dim=2)  # 10, 40, 1
        #print("[tlog] mask_softmax_expanded: " + str(mask_softmax_score_expanded))
        # filter invalid steps
        sequence_mask = torch.unsqueeze(mask.float(), dim=2)  # 10, 40, 1
        #print("[tlog] sequence_mask: " + str(mask_softmax_score_expanded))

        initial_hidden_states = word_inputs
        initial_cell_states = word_inputs

        initial_hidden_states = initial_hidden_states * sequence_mask # 10, 40, 600
        ##print("[tlog] initial_hidden_states: " + str(initial_hidden_states))
        #sys.exit(0)
        initial_cell_states = initial_cell_states * sequence_mask  # 10, 40, 600
        ##print("[tlog] initial_cell_states: " + str(initial_cell_states))

        # record shape of the batch
        shape = list(initial_hidden_states.size())[:-1]
        shape.append(self.hidden_size)
        ##print("[tlog] shape: " + str(shape)) # 10, 37, 100

        # initial embedding states
        embedding_hidden_state_pre = initial_hidden_states.view(-1, self.word_dim)  # 10*40, 600
        #embedding_cell_state = initial_cell_states.view(-1, self.word_dim)  # 10*40, 600

        # randomly initialize the states
        initial_hidden_states = self.create_nograd_variable(-0.05, 0.05, self.training, self.gpu, shape)
        initial_cell_states = self.create_nograd_variable(-0.05, 0.05, self.training, self.gpu, shape)

        #switch_hidden_states = position_switch(initial_hidden_states, position_table)
        #switch_cell_states = position_switch(initial_cell_states, position_table)

        # filter it
        initial_hidden_states = initial_hidden_states * sequence_mask
        initial_cell_states = initial_cell_states * sequence_mask
        ##print("[tlog] initial_hidden_states: " + str(initial_hidden_states))
        ##print("[tlog] initial_cell_states: " + str(initial_cell_states))
        #switch_hidden_states = switch_hidden_states * sequence_mask
        #switch_cell_states = switch_cell_states * sequence_mask

        #sys.exit(0)
        # inital dummy node states
        dummynode_hidden_states = torch.mean(initial_hidden_states, dim=1) # batch_size * hidden_dim
        # self.debug = dummynode_hidden_states
        ##print("[tlog] dummynode_hidden_states: " + str(dummynode_hidden_states))

        dummynode_cell_states = torch.mean(initial_cell_states, dim=1)
        ##print("[tlog] dummynode_cell_states: " + str(dummynode_cell_states)) # batch_size * hidden_dim

        hidden_size = self.hidden_size

        padding_list = [self.create_padding_variable(self.training, self.gpu, (shape[0], step+1, hidden_size)) for step in range(self.step)]

        for i in range(num_layers):
            switch_hidden_states = self.position_switch(initial_hidden_states, position_table)
            switch_cell_states = self.position_switch(initial_cell_states, position_table)
            initial_hidden_states = self.h_drop(initial_hidden_states)
            switch_hidden_states = self.h_drop(switch_hidden_states)
            embedding_hidden_state = self.c_drop(embedding_hidden_state_pre)
            dummynode_hidden_states = self.h_drop(dummynode_hidden_states)
            #print("[tlog] layers: " + str(i))
            # update dummy node states
            # average states
            combined_word_hidden_state = torch.mean(initial_hidden_states, dim=1)

            ##print("[tlog] combined_word_hidden_state: " + str(combined_word_hidden_state))

            reshaped_hidden_output = initial_hidden_states.view(-1, hidden_size)
            # copy dummy states for computing forget gate
            transformed_dummynode_hidden_states = torch.unsqueeze(dummynode_hidden_states, dim=1).repeat(1, shape[1], 1).view(-1, hidden_size)
            #print("[tlog] transformed_dummynode_hidden_states: " + str(transformed_dummynode_hidden_states)) # batch_size * seq_len, hidden_size


            gated_d_t = torch.sigmoid(
                torch.matmul(dummynode_hidden_states, self.gated_Wxd) + torch.matmul(combined_word_hidden_state,
                                                                          self.gated_Whd) + self.gated_bd
            )
            ##print("[tlog] gated_d_t: " + str(gated_d_t))
            #sys.exit(0)
            # output gate
            gated_o_t = torch.sigmoid(
                torch.matmul(dummynode_hidden_states, self.gated_Wxo) + torch.matmul(combined_word_hidden_state,
                                                                          self.gated_Who) + self.gated_bo
            )
            ##print("[tlog] gated_o_t: " + str(gated_o_t))
            # forget gate for hidden states
            gated_f_t = torch.sigmoid(
                torch.matmul(transformed_dummynode_hidden_states, self.gated_Wxf) + torch.matmul(reshaped_hidden_output,
                                                                                      self.gated_Whf) + self.gated_bf
            )
            ##print("[tlog] gated_f_t: " + str(gated_f_t))
            #sys.exit(0)

            # softmax on each hidden dimension
            reshaped_gated_f_t = gated_f_t.view(shape[0], shape[1], hidden_size) + mask_softmax_score_expanded
            ##print("[tlog] reshaped_gated_f_t: " + str(reshaped_gated_f_t))

            gated_softmax_scores = F.softmax(
                torch.cat([reshaped_gated_f_t, torch.unsqueeze(gated_d_t, dim=1)], dim=1), dim=1)
            ##print("[tlog] gated_softmax_scores: " + str(gated_softmax_scores))

            # self.debug = gated_softmax_scores
            # split the softmax scores
            new_reshaped_gated_f_t = gated_softmax_scores[:, :shape[1], :]
            new_gated_d_t = gated_softmax_scores[:, shape[1]:, :]
            ##print("[tlog] new_reshaped_gated_f_t: " + str(new_reshaped_gated_f_t))
            ##print("[tlog] new_gated_d_t: " + str(new_gated_d_t))

            # new dummy states
            dummy_c_t = torch.sum(new_reshaped_gated_f_t * initial_cell_states, dim=1) + torch.squeeze(new_gated_d_t, dim=1) * dummynode_cell_states

            dummy_h_t = gated_o_t * torch.tanh(dummy_c_t)
            ##print("[tlog] dummy_c_t: " + str(dummy_c_t))
            ##print("[tlog] dummy_h_t: " + str(dummy_h_t))
            #sys.exit(0)

            # update word node states
            # get states before
            #for i in padding_list:
            #    print(i.shape)
            #print(initial_hidden_states)

            initial_hidden_states_before = [self.get_hidden_states_before(padding_list[step], initial_hidden_states, step + 1).view(-1, hidden_size) \
                                            for step in range(self.step)]
            #for i in initial_hidden_states_before:
            #    print(i.shape)

            initial_hidden_states_before = self.sum_together(initial_hidden_states_before)

            ##print("[tlog] initial_hidden_states_before: " + str(initial_hidden_states_before))

            initial_hidden_states_after = [self.get_hidden_states_after(padding_list[step], initial_hidden_states, step + 1).view(-1, hidden_size) \
                                           for step in range(self.step)]

            initial_hidden_states_after = self.sum_together(initial_hidden_states_after)
            ##print("[tlog] initial_hidden_states_after: " + str(initial_hidden_states_after))
            #sys.exit(0)
            # get states after
            initial_cell_states_before = [self.get_hidden_states_before(padding_list[step], initial_cell_states, step + 1).view(-1, hidden_size) \
                                          for step in range(self.step)]

            initial_cell_states_before = self.sum_together(initial_cell_states_before)
            ##print("[tlog] initial_cell_states_before: " + str(initial_cell_states_before))
            #sys.exit(0)
            initial_cell_states_after = [self.get_hidden_states_after(padding_list[step], initial_cell_states, step + 1).view(-1, hidden_size) \
                                         for step in range(self.step)]

            initial_cell_states_after = self.sum_together(initial_cell_states_after)
            ##print("[tlog] initial_cell_states_after: " + str(initial_cell_states_after))

            #sys.exit(0)
            # reshape for matmul
            initial_hidden_states = initial_hidden_states.view(-1, hidden_size)
            initial_cell_states = initial_cell_states.view(-1, hidden_size)

            switch_hidden_states = switch_hidden_states.view(-1, hidden_size)
            switch_cell_states = switch_cell_states.view(-1, hidden_size)

            # concat before and after hidden states
            concat_before_after = torch.cat([initial_hidden_states_before, initial_hidden_states_after], dim=1)
            ##print("[tlog] concat_before_after: " + str(concat_before_after))

            # copy dummy node states

            transformed_dummynode_cell_states = torch.unsqueeze(dummynode_cell_states, dim=1).repeat(1, shape[1], 1).view(-1, hidden_size)

            f1_t = torch.sigmoid(
                torch.matmul(initial_hidden_states, self.Wxf1) + torch.matmul(concat_before_after, self.Whf1) +
                torch.matmul(embedding_hidden_state, self.Wif1) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf1) + torch.matmul(switch_hidden_states, self.Wsf1) +self.bf1
            )

            f2_t = torch.sigmoid(
                torch.matmul(initial_hidden_states, self.Wxf2) + torch.matmul(concat_before_after, self.Whf2) +
                torch.matmul(embedding_hidden_state, self.Wif2) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf2) + torch.matmul(switch_hidden_states, self.Wsf2) +self.bf2
            )

            f3_t = torch.sigmoid(
                torch.matmul(initial_hidden_states, self.Wxf3) + torch.matmul(concat_before_after, self.Whf3) +
                torch.matmul(embedding_hidden_state, self.Wif3) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf3) + torch.matmul(switch_hidden_states, self.Wsf3) +self.bf3
            )

            f4_t = torch.sigmoid(
                torch.matmul(initial_hidden_states, self.Wxf4) + torch.matmul(concat_before_after, self.Whf4) +
                torch.matmul(embedding_hidden_state, self.Wif4) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf4) + torch.matmul(switch_hidden_states, self.Wsf4) + self.bf4
            )
            f5_t = torch.sigmoid(
                torch.matmul(initial_hidden_states, self.Wxf5) + torch.matmul(concat_before_after, self.Whf5) +
                torch.matmul(embedding_hidden_state, self.Wif5) + torch.matmul(transformed_dummynode_hidden_states, self.Wdf5) + torch.matmul(switch_hidden_states, self.Wsf5)+ self.bf5
            )

            i_t = torch.sigmoid(
                torch.matmul(initial_hidden_states, self.Wxi) + torch.matmul(concat_before_after, self.Whi) +
                torch.matmul(embedding_hidden_state, self.Wii) + torch.matmul(transformed_dummynode_hidden_states, self.Wdi) + torch.matmul(switch_hidden_states, self.Wsi) +self.bi
            )

            o_t = torch.sigmoid(
                torch.matmul(initial_hidden_states, self.Wxo) + torch.matmul(concat_before_after, self.Who) +
                torch.matmul(embedding_hidden_state, self.Wio) + torch.matmul(transformed_dummynode_hidden_states, self.Wdo) + torch.matmul(switch_hidden_states, self.Wso) +self.bo
            )
            #print(embedding_hidden_state.shape, self.Wiu.shape)
            embedding_cell_state = torch.tanh(
                torch.matmul(initial_hidden_states, self.Wxu) +torch.matmul(concat_before_after, self.Whu)+
                torch.matmul(embedding_hidden_state,self.Wiu) +torch.matmul(transformed_dummynode_hidden_states,self.Wdu) +self.bu
                )

            f1_t, f2_t, f3_t, f4_t, f5_t, i_t = torch.unsqueeze(f1_t, dim=1), torch.unsqueeze(f2_t, dim=1), torch.unsqueeze(
                f3_t, dim=1), torch.unsqueeze(f4_t, dim=1), torch.unsqueeze(f5_t, dim=1), torch.unsqueeze(i_t, dim=1)

            six_gates = torch.cat([f1_t, f2_t, f3_t, f4_t, f5_t, i_t], dim=1)
            six_gates = F.softmax(six_gates, dim=1)
            #print("[tlog] five_gates: " + str(five_gates))

            f1_t, f2_t, f3_t, f4_t, f5_t, i_t = torch.chunk(six_gates, 6, dim=1)
            #print("[tlog] f1_t: " + str(f1_t))
            #sys.exit(0)
            f1_t, f2_t, f3_t, f4_t, f5_t,i_t = torch.squeeze(f1_t, dim=1), torch.squeeze(f2_t, dim=1), torch.squeeze(f3_t, dim=1), torch.squeeze(
                f4_t, dim=1), torch.squeeze(f5_t, dim =1), torch.squeeze(i_t, dim=1)

            c_t = ( initial_cell_states_before * f1_t) + ( initial_cell_states_after * f2_t) + (
             embedding_cell_state * f3_t) + ( transformed_dummynode_cell_states * f4_t ) + (switch_cell_states * f5_t)+(initial_cell_states * i_t)

            h_t = o_t * torch.tanh(c_t)

            ##print("[tlog] c_t: " + str(c_t))
            ##print("[tlog] h_t: " + str(h_t))
            #sys.exit(0)

            # update states
            initial_hidden_states = h_t.view(shape[0], shape[1], hidden_size)
            initial_cell_states = c_t.view(shape[0], shape[1], hidden_size)

            initial_hidden_states = initial_hidden_states * sequence_mask
            initial_cell_states = initial_cell_states * sequence_mask

            dummynode_hidden_states = dummy_h_t
            dummynode_cell_states = dummy_c_t

        #initial_hidden_states = self.h_drop(initial_hidden_states)
        #initial_cell_states = self.c_drop(initial_cell_states)
        ##print("[tlog] initial_hidden_states: " + str(initial_hidden_states))
        return initial_hidden_states, initial_cell_states