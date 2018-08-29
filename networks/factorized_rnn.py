from torch.nn.modules import rnn
from torch import nn
from torch.nn import functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
import torch as th
import math

def diag_lstm_size(input_size, hidden_size, fact_size, bias=True):
    W_ih = 4 * hidden_size * fact_size + fact_size * input_size + fact_size
    W_hh = 4 * hidden_size * fact_size + fact_size * hidden_size + fact_size
    total_size = W_hh + W_ih
    if bias:
        total_size += 4 * hidden_size * 2
    return total_size

def lstm_size(input_size, hidden_size, bias=True):
    W_ih = 4*hidden_size*input_size
    W_hh = 4*hidden_size*hidden_size
    total_size = W_hh + W_ih
    if bias:
        total_size += 4*hidden_size*2
    return total_size


class DiagonalLSTMCell(rnn.RNNCellBase):
    def __init__(
      self,
      input_size,
      hidden_size,
      num_task,
      task_embed_size=100,
      bias=True
    ):
        super(DiagonalLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih1 = nn.Parameter(th.Tensor(4*hidden_size, task_embed_size))
        self.weight_ih2 = nn.Parameter(th.Tensor(task_embed_size, input_size))

        self.weight_hh1 = nn.Parameter(th.Tensor(4 * hidden_size, task_embed_size))
        self.weight_hh2 = nn.Parameter(th.Tensor(task_embed_size, hidden_size))

        self.embedding = nn.Embedding(num_task, task_embed_size)
        if bias:
            self.bias_ih = nn.Parameter(th.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(th.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx, task_ids):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        W_ih, W_hh = self.restore_weights_batch(task_ids)
        return batch_weights_lstm_forward(
            input, hx,
            W_ih, W_hh,
            self.bias_ih, self.bias_hh
        )

    def restore_weights_batch(self, task_ids):
        task_embeds = self.embedding(task_ids)
        diag_embeds = th.stack([th.diag(e) for e in task_embeds])
        W_ih = th.matmul(th.matmul(self.weight_ih1, diag_embeds), self.weight_ih2)
        W_hh = th.matmul(th.matmul(self.weight_hh1, diag_embeds), self.weight_hh2)
        return W_ih, W_hh

    def forward_old(self, input, hx, task_ids):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        cell = self._backend.LSTMCell
        hx,cx = hx
        bias_ih = self.bias_ih
        bias_hh = self.bias_hh
        batch_size = input.size(0)
        new_hx, new_cx = [None]*batch_size, [None]*batch_size

        for t in range(batch_size):
            W_ih, W_hh = self.restore_weights_old(task_ids[t])
            new_hx[t], new_cx[t] = cell(
                input[t:t+1],
                (hx[t:t+1],cx[t:t+1]),
                W_ih, W_hh,
                bias_ih, bias_hh
            )
        return th.cat(new_hx), th.cat(new_cx)

    def restore_weights_old(self, task_ids):
        task_embed = self.embedding(task_ids)
        task_diag = th.diag(task_embed)
        W_ih = th.mm(th.mm(self.weight_ih1,task_diag),self.weight_ih2)
        W_hh = th.mm(th.mm(self.weight_hh1,task_diag),self.weight_hh2)
        return W_ih, W_hh


def batch_weights_lstm_forward(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    """
    the main difference with LSTMCell function from torch.nn._functions.rnn.py
    is that layer weights comes in a batch!

    That's right instead of a tensor of shape (input_size, hidden_size) this
    function expects a tensor of shape (batch_size, input_size, hidden_size).
    This weird shape comes from the fact that the weights for the task
    specific LSTM depend on the current tasks.
    """
    if input.is_cuda:
        igates = batch_linear(input, w_ih)
        hgates = batch_linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = batch_linear(input, w_ih, b_ih) + batch_linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def batch_linear(input_batch, weight_batch, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, in\_features)`
        - Weight: :math:`(N, out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    input = input_batch.unsqueeze(-1)  #shape (N, in_features, 1)
    if input.dim() == 3 and bias is not None:
        bias = bias.unsqueeze(-1)
        return th.baddbmm(bias, weight_batch, input).squeeze(-1)

    output = th.bmm(weight_batch, input).squeeze(-1)
    if bias is not None:
        output += bias
    return output
