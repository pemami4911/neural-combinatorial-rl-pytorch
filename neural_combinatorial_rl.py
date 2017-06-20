import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import math

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
            n_glimpses=1):
        super(Decoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.terminating_symbol = terminating_symbol 
        self.decode_type = decode_type
        
        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear_hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=True, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim)

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]
            embedded inputs: [sourceL x batch_size x embeddign_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]
            context: [sourceL x batch_size x hidden_dim]
            context_mask:
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
    
        outputs = []
        selections = []
        steps = range(self.max_length)  # or until terminating symbol ?
        inps = []
        idxs = None
        mask = None
        
        for i in steps:
            hx, cx, outs = recurrence(decoder_input, hidden)
            hidden = (hx, cx)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            decoder_input, idxs, mask = self.decode_fn(outs,
                    embedded_inputs,
                    idxs, 
                    mask)
            inps.append(decoder_input) 
            # use outs to point to next object
            outputs.append(outs)
            selections.append(idxs)

        return (outputs, selections), hidden


    def decode_fn(self, logits, embedded_inputs, prev=None, logit_mask=None):
        """
        Return the next input or the decoder by selecting the 
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

        logits_ = logits.clone()
        # Set the logits in prev to -inf
        #for p in prev:
        if logit_mask is None:
            logit_mask = torch.zeros(logits.size(0) * logits.size(1)).byte()
            if USE_CUDA:
                logit_mask = logit_mask.cuda()

        # to prevent them from being reselected. 
        # Or, allow re-selection and penalize in the objective function
        if prev is not None:
            for i in range(logits.size(0)):
                val = (logits.size(1) * i) + prev[i]
                logit_mask[val.data] = 1
            logit_mask = logit_mask.view(logits.size(0), logits.size(1))
            logits_[logit_mask] = 0
            logit_mask = logit_mask.view(logits.size(0) * logits.size(1))

        if self.decode_type == "Greedy":
            # get argmax for each entry in the batch
            _, idxs = torch.max(logits_, 1)  
        elif self.decode_type == "Stochastic":
            idxs = logits_.multinomial()

        # idxs is [batch_size]
        
        # select next embeddings based on idx
        # No gather_nd in PyTorch :'( so just flatten the mask into 1D vector
        # Will reshape into the proper size later
        mask = torch.zeros(idxs.size(0) * embedded_inputs.size(0)).byte()
        
        if USE_CUDA:
            mask = mask.cuda()

        for i in range(idxs.size(0)):
            val = (embedded_inputs.size(0) * i) + idxs[i]
            mask[val.data] = 1
        
        mask = mask.view(embedded_inputs.size(0), idxs.size(0))
        # repeat across the embedding dimension
        # [sourceL x batch_dim x embedding_dim
        mask = mask.unsqueeze(2).repeat(1, 1, embedded_inputs.size(2))
        return embedded_inputs[mask].view(idxs.size(0), embedded_inputs.size(2)), idxs, logit_mask
    
class PointerNetwork(nn.Module):
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            max_decoding_len,
            terminating_symbol,
            n_glimpses,
            n_layers,
            tanh_exploration,
            dropout):
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
                decode_type="Stochastic",
                n_glimpses=n_glimpses)
        
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
                dropout)
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

        actions = []
        fl_inputs = inputs.view(batch_size * sourceL)

        for action_id in action_idxs:
            idx_mask = torch.zeros(batch_size * sourceL).byte()
            
            if USE_CUDA: 
                idx_mask = idx_mask.cuda()

            for i in range(batch_size):
                val = (sourceL * i) + action_id[i]
                idx_mask[val.data] = 1

            actions.append(fl_inputs[idx_mask].view(batch_size)) 
        
        if self.is_train:
            # logits_ is a list of len sourceL of [batch_size x sourceL]
            logits = []
            for logit, action_id in zip(logits_, action_idxs):
                logit_mask = torch.zeros(batch_size * sourceL).byte()
                logit = logit.view(batch_size * sourceL)

                if USE_CUDA: 
                    logit_mask = logit_mask.cuda()
                
                # assemble the mask by iterating over the batch
                for i in range(batch_size):
                    val = (sourceL * i) + action_id[i]
                    logit_mask[val.data] = 1
                    
                logits.append(logit[logit_mask].view(batch_size))
        else:
            # return the list of len sourceL of [batch_size x sourceL]
            logits = logits_

        # get the critic value fn estimates for the baseline
        # [batch_size]
        v = self.critic_net(embedded_inputs)
    
        # [batch_size]
        R = self.objective_fn(actions, USE_CUDA)
        
        return R, v, logits, actions, action_idxs
    
