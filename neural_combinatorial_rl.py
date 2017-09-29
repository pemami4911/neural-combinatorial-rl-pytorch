import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math

from beam_search import Beam

USE_CUDA = True


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                dropout=dropout)
    
    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden
    
    def init_hidden(self, inputs):
        batch_size = inputs.size(1)
        hx = autograd.Variable(torch.zeros(self.n_layers,
            batch_size,
            self.hidden_dim),
            requires_grad=False)
        cx = autograd.Variable(torch.zeros(self.n_layers,
            batch_size,
            self.hidden_dim),
            requires_grad=False)
        if USE_CUDA:
            return hx.cuda(), cx.cuda()
        else:
            return hx, cx

class Attention(nn.Module):
    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        
        self.v = autograd.Variable(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)) , 1. / math.sqrt(dim))

        if USE_CUDA:
            self.v = self.v.cuda()
        
    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
            time step. batch x dim
            ref: the set of hidden states from the encoder. sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            scores = self.sm(self.C * self.tanh(u))
        else:
            scores = self.sm(u)  # batch_size x sourceL
        # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = 
        # [batch_size x h_dim x 1]
        att_state = torch.bmm(e, scores.unsqueeze(2)).squeeze(2)  
        return att_state, scores


class Decoder(nn.Module):
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            max_length,
            tanh_exploration,
            terminating_symbol,
            decode_type,
            n_glimpses=1,
            beam_size=0):
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.terminating_symbol = terminating_symbol 
        self.decode_type = decode_type
        self.beam_size = beam_size

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear_hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=True, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim)

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded inputs: [sourceL x batch_size x embeddign_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim] 
        """
        def recurrence(x, hidden):
            hx, cx = hidden  # batch_size x hidden_dim
            
            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # batch_size x hidden_dim
            
            g_l = hy
            for i in range(self.n_glimpses):
                g_l, _ = self.glimpse(g_l, context)
            
            h_tilde, output = self.pointer(g_l, context)
            h_tilde = F.tanh(self.linear_hidden_out(torch.cat((h_tilde, hy), 1)))

            return h_tilde, cy, output
    
        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(self.max_length)  # or until terminating symbol ?
        inps = []
        idxs = None
        mask = None
       
        if self.decode_type == "stochastic":
            for i in steps:
                hx, cx, outs = recurrence(decoder_input, hidden)
                hidden = (hx, cx)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs, mask = self.decode_stochastic(outs,
                        embedded_inputs,
                        idxs, 
                        mask)
                inps.append(decoder_input) 
                # use outs to point to next object
                outputs.append(outs)
                selections.append(idxs)

            return (outputs, selections), hidden
        
        elif self.decode_type == "beam_search":
            
            # Expand input tensors for beam search
            decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))
            context = Variable(context.data.repeat(1, self.beam_size, 1))
            hidden = (Variable(hidden[0].data.repeat(self.beam_size, 1)),
                    Variable(hidden[1].data.repeat(self.beam_size, 1)))
            
            beam = [
                    Beam(self.beam_size, self.max_length, cuda=USE_CUDA) 
                    for k in range(batch_size)
            ]
            
            for i in steps:
                hx, cx, outs = recurrence(decoder_input, hidden)
                hidden = (hx, cx)
                
                outs = outs.view(self.beam_size, batch_size, -1
                        ).transpose(0, 1).contiguous()
                
                # if i < steps[-1]:
                #     n_best = self.beam_size
                # else:
                n_best = 1
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs, active, mask = self.decode_beam(outs,
                        embedded_inputs, beam, batch_size, n_best, idxs, mask)
               
                inps.append(decoder_input) 
                # use outs to point to next object
                if self.beam_size > 1:
                    outputs.append(outs[:, 0,:])
                else:
                    outputs.append(outs.squeeze(0))
                # Check for indexing
                selections.append(idxs)
                 # Should be done decoding
                if len(active) == 0:
                    break
                decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))

            return (outputs, selections), hidden

    def decode_stochastic(self, logits, embedded_inputs, prev=None, logit_mask=None):
        """
        Return the next input for the decoder by selecting the 
        input corresponding to the max output

        Args: 
            logits: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            prev: list of Tensors containing previously selected indices
        Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding
        """
        batch_size = logits.size(0)
        logits_ = logits.clone()
        
        if logit_mask is None:
            logit_mask = torch.zeros(logits.size()).byte()
            if USE_CUDA:
                logit_mask = logit_mask.cuda()

        # to prevent them from being reselected. 
        # Or, allow re-selection and penalize in the objective function
        if prev is not None:
            #import pdb; pdb.set_trace()
            logit_mask[[x for x in range(batch_size)],
                    prev.data] = 1
            logits_[logit_mask] = 0
            # renormalize
            logits_ /= logits_.sum()
        
        # idxs is [batch_size]
        idxs = logits_.multinomial().squeeze(1)
        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :] 
        return sels, idxs, logit_mask

    def decode_beam(self, logits, embedded_inputs, beam, batch_size, n_best, prev=None, logit_mask=None):
        logits_ = logits.clone()

        if logit_mask is None:
            logit_mask = torch.zeros(logits_.size()).byte()
            if USE_CUDA:
                logit_mask = logit_mask.cuda()

        # to prevent them from being reselected. 
        # Or, allow re-selection and penalize in the objective function
        if prev is not None:
            logit_mask[0,:,prev.data[0]] = 1
            logits_[logit_mask] = 0
            logits_ /= logits_.sum()

        active = []
        for b in range(batch_size):
            if beam[b].done:
                continue

            if not beam[b].advance(logits_.data[b]):
                active += [b]
        
        
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]
        
        all_idxs = Variable(torch.LongTensor([[x for x in hyp] for hyp in all_hyp]).squeeze())
      
        if all_idxs.dim() == 2:
            if all_idxs.size(1) > n_best:
                idxs = all_idxs[:,-1]
            else:
                idxs = all_idxs
        elif all_idxs.dim() == 3:
            idxs = all_idxs[:, -1, :]
        else:
            if all_idxs.size(0) > 1:
                idxs = all_idxs[-1]
            else:
                idxs = all_idxs
        
        if USE_CUDA:
            idxs = idxs.cuda()

        if idxs.dim() > 1:
            x = embedded_inputs[idxs.transpose(0,1).contiguous().data, 
                    [x for x in range(batch_size)], :]
        else:
            x = embedded_inputs[idxs.data, [x for x in range(batch_size)], :]
        return x.view(idxs.size(0) * n_best, embedded_inputs.size(2)), idxs, active, logit_mask

class PointerNetwork(nn.Module):
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            max_decoding_len,
            terminating_symbol,
            n_glimpses,
            n_layers,
            tanh_exploration,
            dropout,
            beam_size):
        super(PointerNetwork, self).__init__()

        self.encoder = Encoder(
                embedding_dim,
                hidden_dim,
                n_layers,
                dropout)

        self.decoder = Decoder(
                embedding_dim,
                hidden_dim,
                max_length=max_decoding_len,
                tanh_exploration=tanh_exploration,
                terminating_symbol=terminating_symbol,
                decode_type="stochastic",
                n_glimpses=n_glimpses,
                beam_size=beam_size)
        
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.decoder_in_0.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                1. / math.sqrt(embedding_dim))
            
    def forward(self, inputs):
        """ Propagate inputs through the network
        Args: 
            inputs: [sourceL x batch_size x embedding_dim]
        """
        
        encoder_h0, encoder_c0 = self.encoder.init_hidden(inputs)        

        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_h0, encoder_c0))
        
        dec_init_state = (enc_h_t[-1], enc_c_t[-1])
    
        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)
        
        (pointer_logits, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                inputs,
                dec_init_state,
                enc_outputs)

        return pointer_logits, input_idxs
        

class CriticNetwork(nn.Module):
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters,
            n_layers,
            dropout):
        super(CriticNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(self.embedding_dim,
                self.hidden_dim,
                n_layers,
                dropout)
        self.process_block = Attention(hidden_dim)
        self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        encoder_h0, encoder_c0 = self.encoder.init_hidden(inputs)        
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_h0, encoder_c0))
       
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            process_block_state, _ = self.process_block(process_block_state, enc_outputs)        
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out

class NeuralCombOptRL(nn.Module):
    """
    This module contains the PointerNetwork (actor) and
    CriticNetwork (critic). It requires
    an application-specific objective function
    """
    def __init__(self, 
            input_dim,
            embedding_dim,
            hidden_dim,
            max_decoding_len,
            terminating_symbol,
            n_glimpses,
            n_process_block_iters,
            n_layers,
            tanh_exploration,
            dropout,
            beam_size,
            objective_fn,
            is_train):
        super(NeuralCombOptRL, self).__init__()
        self.objective_fn = objective_fn
        self.input_dim = input_dim
        self.is_train = is_train

        self.actor_net = PointerNetwork(
                embedding_dim,
                hidden_dim,
                max_decoding_len,
                terminating_symbol,
                n_glimpses,
                n_layers,
                tanh_exploration,
                dropout,
                beam_size)
        self.critic_net = CriticNetwork(embedding_dim,
                hidden_dim,
                n_process_block_iters,
                n_layers,
                dropout)
       
        embedding_ = torch.FloatTensor(input_dim,
            embedding_dim)
        
        if USE_CUDA: 
            embedding_ = embedding_.cuda()

        self.embedding = nn.Parameter(embedding_)
         
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_dim)),
            1. / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_dim, sourceL]
        """
        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        sourceL = inputs.size(2)

        # repeat embeddings across batch_size
        # result is [batch_size x input_dim x embedding_dim]
        embedding = self.embedding.repeat(batch_size, 1, 1)  
        embedded_inputs = []
        # result is [batch_size, 1, input_dim, sourceL] 
        ips = inputs.unsqueeze(1)
        
        for i in range(sourceL):
            # [batch_size x 1 x input_dim] * [batch_size x input_dim x embedding_dim]
            # result is [batch_size, embedding_dim]
            embedded_inputs.append(torch.bmm(
                ips[:, :, :, i].float(),
                embedding).squeeze(1))

        # Result is [sourceL x batch_size x embedding_dim]
        embedded_inputs = torch.cat(embedded_inputs).view(
                sourceL,
                batch_size,
                embedding.size(2))

        # query the actor net for the input indices 
        # making up the output, and the pointer attn 
        logits_, action_idxs = self.actor_net(embedded_inputs)
       
        # Select the actions (inputs pointed to 
        # by the pointer net) and the corresponding
        # logits
        # should be size [batch_size x 
        actions = []
        # inputs is [batch_size, input_dim, sourceL]
        inputs_ = inputs.transpose(1, 2)
        # inputs_ is [batch_size, sourceL, input_dim]
        for action_id in action_idxs:
            actions.append(inputs_[[x for x in range(batch_size)], action_id.data, :])

        if self.is_train:
            # logits_ is a list of len sourceL of [batch_size x sourceL]
            logits = []
            for logit, action_id in zip(logits_, action_idxs):
                logits.append(logit[[x for x in range(batch_size)], action_id.data])
        else:
            # return the list of len sourceL of [batch_size x sourceL]
            logits = logits_

        # get the critic value fn estimates for the baseline
        # [batch_size]
        v = self.critic_net(embedded_inputs)
    
        # [batch_size]
        R = self.objective_fn(actions, USE_CUDA)
        
        return R, v, logits, actions, action_idxs
    
