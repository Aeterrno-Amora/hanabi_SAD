# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from typing import Tuple, Dict
import common_utils, sym_utils
import math


class SymLinear(nn.Module):
    __constants__ = ['n', 'in_sizes', 'in_size', 'out_sizes', 'out_size']

    def __init__(self, n : int, in_sizes, out_sizes):
        super().__init__()
        self.n = n
        self.in_sizes = in_sizes
        self.in_size = sym_utils.sizes_to_size(self.n, self.in_sizes)
        self.out_sizes = out_sizes
        self.out_size = sym_utils.sizes_to_size(self.n, self.out_sizes)

        self.weight = sym_utils.build_weight(self, "weight", self.n, self.in_sizes, self.out_sizes)
        self.bias = nn.Parameter(torch.empty(self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        assert(len(list(self.parameters())))
        stdv = 1.0 / math.sqrt(self.out_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input):
        return sym_utils.linear(self.n, self.in_sizes, self.out_sizes, self.out_size, input, self.weight, self.bias)

class SymLSTMCell(nn.Module):
    __constants__ = ['n', 'in_sizes', 'in_size', 'hid_sizes', 'hid_size']

    def __init__(self, n, in_sizes, hid_sizes):
        super().__init__()
        self.n = n
        self.in_sizes = in_sizes
        self.in_size = sym_utils.sizes_to_size(self.n, self.in_sizes)
        self.hid_sizes = hid_sizes
        self.hid_size = sym_utils.sizes_to_size(self.n, self.hid_sizes)
        self.gate_sizes = tuple(d * 4 for d in hid_sizes)
        self.gate_size = sym_utils.sizes_to_size(self.n, self.gate_sizes)

        self.weight_ih = sym_utils.build_weight(self, "weight_ih", self.n, self.in_sizes, self.gate_sizes)
        self.weight_hh = sym_utils.build_weight(self, "weight_hh", self.n, self.hid_sizes, self.gate_sizes)
        self.bias_ih = nn.Parameter(torch.empty(self.gate_size))
        self.bias_hh = nn.Parameter(torch.empty(self.gate_size))
        self.reset_parameters()

    def reset_parameters(self):
        assert(len(list(self.parameters())))
        stdv = 1.0 / math.sqrt(self.hid_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        if hx is None:
            zeros = torch.zeros(*input.size()[:-1], self.hid_size, device=input.device)
            h0, c0 = zeros, zeros
        else: h0, c0 = hx
        gates = sym_utils.linear(self.n, self.in_sizes, self.gate_sizes, self.gate_size, input, self.weight_ih, self.bias_ih) + sym_utils.linear(self.n, self.hid_sizes, self.gate_sizes, self.gate_size, h0, self.weight_hh, self.bias_hh)
        dim = gates.dim()
        h1 = torch.empty_like(h0)
        c1 = torch.empty_like(c0)
        st = 0
        for m in range(len(self.hid_sizes)):
            if self.hid_sizes[m]:
                for pi in range(sym_utils.num_perms(self.n, m)):
                    (g, i, f, o) = gates.narrow(dim-1, st * 4, self.gate_sizes[m]).chunk(4, dim-1)
                    c1_i = c1.narrow(dim-1, st, self.hid_sizes[m])
                    c1_i.copy_(g.tanh().mul(i.sigmoid()) + c0.narrow(dim-1, st, self.hid_sizes[m]).mul(f.sigmoid()))
                    h1.narrow(dim-1, st, self.hid_sizes[m]).copy_(c1_i.tanh().mul(o.sigmoid()))
                    st += self.hid_sizes[m]
        return (h1, c1)

class SymLSTM(nn.Module):
    __constants__ = ['num_layers', 'n', 'in_sizes', 'in_size', 'hid_sizes', 'hid_size']

    def __init__(self, n, in_sizes, hid_sizes, num_layers):
        super().__init__()
        self.n = n
        self.in_sizes = in_sizes
        self.in_size = sym_utils.sizes_to_size(self.n, self.in_sizes)
        self.hid_sizes = hid_sizes
        self.hid_size = sym_utils.sizes_to_size(self.n, self.hid_sizes)
        self.num_layers = num_layers

        self.lstm = []
        for i in range(self.num_layers):
            cell = SymLSTMCell(self.n, self.hid_sizes if i else self.in_sizes, self.hid_sizes)
            self.add_module("lstm_%d" % i, cell)
            self.lstm.append(cell)
        assert(len(list(self.parameters())))

    def forward(self, input, hx=None):
        ispacked = isinstance(input, nn.utils.rnn.PackedSequence)
        if ispacked:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes, sorted_indices, unsorted_indices = None, None, None
            max_batch_size = input.size(1)

        if hx is None:
            zeros = torch.zeros(self.num_layers, max_batch_size, self.hid_size, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = self.permute_hid(hx, sorted_indices)

        seq_len = input.size(0)
        output = torch.empty(seq_len, max_batch_size, self.hid_size)
        for i in range(seq_len):
            for t in self.num_layers:
                hx[0][t], hx[1][t] = self.lstm[t](x, (hx[0][t], hx[1][t]))
                x = hx[0][t]
            output[i] = x

        if ispacked:
            output = nn.utils.rnn.PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hx, unsorted_indices)

    def permute_hidden(self, hx, permutation):
        if permutation is None: return hx
        return hx[0].index_select(1, permutation), hx[1].index_select(1, permutation)

class R2D2Net(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer", "hand_size"]

    def __init__(self, device, symnet, in_size, hid_size, out_size, num_lstm_layer, hand_size):
        print("R2D2Net(",device, symnet, in_size, hid_size, out_size, num_lstm_layer, hand_size,")")
        super().__init__()
        self.symnet = symnet
        self.in_sizes = tuple(in_size)
        self.hid_sizes = tuple(hid_size)
        self.out_sizes = tuple(out_size)
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer
        self.hand_size = hand_size

        if symnet:
            self.in_dim = sym_utils.sizes_to_size(5, self.in_sizes)
            self.hid_dim = sym_utils.sizes_to_size(5, self.hid_sizes)
            self.out_dim = sym_utils.sizes_to_size(5, self.out_sizes)
            self.net = nn.Sequential(SymLinear(5, self.in_sizes, self.hid_sizes), nn.ReLU())
            self.lstm = SymLSTM(5, self.hid_sizes, self.hid_sizes, self.num_lstm_layer).to(device)
            self.fc_v = SymLinear(5, self.hid_sizes, (1,))
            self.fc_a = SymLinear(5, self.hid_sizes, self.out_sizes)
            self.pred = SymLinear(5, self.hid_sizes, (self.hand_size*3,))

        else:
            self.in_dim = in_size[0]
            self.hid_dim = hid_size[0]
            self.out_dim = out_size[0]
            self.net = nn.Sequential(nn.Linear(self.in_dim, self.hid_dim), nn.ReLU())
            self.lstm = nn.LSTM(
                self.hid_dim,
                self.hid_dim,
                num_layers=self.num_lstm_layer,  # , batch_first=True
            ).to(device)
            self.lstm.flatten_parameters()

            self.fc_v = nn.Linear(self.hid_dim, 1)
            self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

            # for aux task
            self.pred = nn.Linear(self.hid_dim, self.hand_size * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % priv_s.dim()

        priv_s = priv_s.unsqueeze(0)
        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}#, t_pred

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert priv_s.dim() == 3 or priv_s.dim() == 2, \
            "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.net(priv_s)
        if len(hid) == 0:
            o, (h, c) = self.lstm(x)
        else:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)

        q = self._duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    @torch.jit.script_method
    def _duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    def cross_entropy(self, net, lstm_o, target_p, hand_slot_mask, seq_len):
        # target_p: [seq_len, batch, num_player, 5, 3]
        # hand_slot_mask: [seq_len, batch, num_player, 5]
        logit = net(lstm_o).view(target_p.size())
        q = nn.functional.softmax(logit, -1)
        logq = nn.functional.log_softmax(logit, -1)
        plogq = (target_p * logq).sum(-1)
        xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(min=1e-6)

        if xent.dim() == 3:
            # [seq, batch, num_player]
            xent = xent.mean(2)

        # save before sum out
        seq_xent = xent
        xent = xent.sum(0)
        assert xent.size() == seq_len.size()
        avg_xent = (xent / seq_len).mean().item()
        return xent, avg_xent, q, seq_xent.detach()

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return self.cross_entropy(self.pred, lstm_o, target, hand_slot_mask, seq_len)


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["vdn", "multi_step", "gamma", "eta", "uniform_priority"]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        symnet,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        uniform_priority,
    ):
        super().__init__()
        self.online_net = R2D2Net(
            device, symnet, in_dim, hid_dim, out_dim, num_lstm_layer, hand_size
        ).to(device)
        self.target_net = R2D2Net(
            device, symnet, in_dim, hid_dim, out_dim, num_lstm_layer, hand_size
        ).to(device)
        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.uniform_priority = uniform_priority

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.symnet,
            self.online_net.in_sizes,
            self.online_net.hid_sizes,
            self.online_net.out_sizes,
            self.online_net.num_lstm_layer,
            self.online_net.hand_size,
            self.uniform_priority
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        obsize, ibsize, num_player = 0, 0, 0
        if self.vdn:
            obsize, ibsize, num_player = obs["priv_s"].size()[:3]
            priv_s = obs["priv_s"].flatten(0, 2)
            legal_move = obs["legal_move"].flatten(0, 2)
            eps = obs["eps"].flatten(0, 2)
        else:
            obsize, ibsize = obs["priv_s"].size()[:2]
            num_player = 1
            priv_s = obs["priv_s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
            eps = obs["eps"].flatten(0, 1)

        hid = {
            "h0": obs["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": obs["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }

        greedy_action, new_hid = self.greedy_act(priv_s, legal_move, hid)

        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).detach().long()

        if self.vdn:
            action = action.view(obsize, ibsize, num_player)
            greedy_action = greedy_action.view(obsize, ibsize, num_player)
            rand = rand.view(obsize, ibsize, num_player)
        else:
            action = action.view(obsize, ibsize)
            greedy_action = greedy_action.view(obsize, ibsize)
            rand = rand.view(obsize, ibsize)

        hid_shape = (
            obsize,
            ibsize * num_player,
            self.online_net.num_lstm_layer,
            self.online_net.hid_dim
        )
        h0 = new_hid["h0"].transpose(0, 1).view(*hid_shape)
        c0 = new_hid["c0"].transpose(0, 1).view(*hid_shape)

        reply = {
            "a": action.detach().cpu(),
            "greedy_a": greedy_action.detach().cpu(),
            "h0": h0.contiguous().detach().cpu(),
            "c0": c0.contiguous().detach().cpu(),
        }
        return reply

    @torch.jit.script_method
    def compute_priority(self, input_: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        compute priority for one batch
        """
        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"]).detach().cpu()}

        obsize, ibsize, num_player = 0, 0, 0
        flatten_end = 0
        if self.vdn:
            obsize, ibsize, num_player = input_["priv_s"].size()[:3]
            flatten_end = 2
        else:
            obsize, ibsize = input_["priv_s"].size()[:2]
            num_player = 1
            flatten_end = 1

        priv_s = input_["priv_s"].flatten(0, flatten_end)
        legal_move = input_["legal_move"].flatten(0, flatten_end)
        online_a = input_["a"].flatten(0, flatten_end)

        next_priv_s = input_["next_priv_s"].flatten(0, flatten_end)
        next_legal_move = input_["next_legal_move"].flatten(0, flatten_end)
        temperature = input_["temperature"].flatten(0, flatten_end)

        hid = {
            "h0": input_["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        next_hid = {
            "h0": input_["next_h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["next_c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        reward = input_["reward"].flatten(0, 1)
        bootstrap = input_["bootstrap"].flatten(0, 1)

        online_qa = self.online_net(priv_s, legal_move, online_a, hid)[0]
        next_a, _ = self.greedy_act(
            next_priv_s, next_legal_move, next_hid)
        target_qa, _, _, _ = self.target_net(
            next_priv_s, next_legal_move, next_a, next_hid,
        )

        bsize = obsize * ibsize
        if self.vdn:
            # sum over action & player
            online_qa = online_qa.view(bsize, num_player).sum(1)
            target_qa = target_qa.view(bsize, num_player).sum(1)

        assert reward.size() == bootstrap.size()
        assert reward.size() == target_qa.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        priority = (target - online_qa).abs()
        priority = priority.view(obsize, ibsize).detach().cpu()
        return {"priority": priority}

    ############# python only functions #############
    def flat_4d(self, data):
        """
        rnn_hid: [num_layer, batch, num_player, dim] -> [num_player, batch, dim]
        seq_obs: [seq_len, batch, num_player, dim] -> [seq_len, batch, dim]
        """
        bsize = 0
        num_player = 0
        for k, v in data.items():
            if num_player == 0:
                bsize, num_player = v.size()[1:3]

            if v.dim() == 4:
                d0, d1, d2, d3 = v.size()
                data[k] = v.view(d0, d1 * d2, d3)
            elif v.dim() == 3:
                d0, d1, d2 = v.size()
                data[k] = v.view(d0, d1 * d2)
        return bsize, num_player

    def td_error(self, obs, hid, action, reward, terminal, bootstrap, seq_len, stat):
        max_seq_len = obs["priv_s"].size(0)

        bsize, num_player = 0, 1
        if self.vdn:
            bsize, num_player = self.flat_4d(obs)
            self.flat_4d(action)

        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        action = action["a"]

        hid = {}

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qa, greedy_a, _, lstm_o = self.online_net(
            priv_s, legal_move, action, hid)

        with torch.no_grad():
            target_qa, _, _, _ = self.target_net(priv_s, legal_move, greedy_a, hid)
            # assert target_q.size() == pa.size()
            # target_qe = (pa * target_q).sum(-1).detach()
            assert online_qa.size() == target_qa.size()

        if self.vdn:
            online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
            target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
            lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

        terminal = terminal.float()
        bootstrap = bootstrap.float()

        errs = []
        target_qa = torch.cat([target_qa[self.multi_step:], target_qa[:self.multi_step]], 0)
        target_qa[-self.multi_step:] = 0

        assert target_qa.size() == reward.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        return err, lstm_o

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        rotate = [num_player-1]
        rotate.extend(list(range(num_player-1)))
        partner_hand = own_hand[:, :, rotate, :, :]
        partner_hand_slot_mask = partner_hand.sum(4)
        partner_belief1 = belief1[:, :, rotate, :, :].detach()

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def loss(self, batch, pred_weight, stat):
        err, lstm_o = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
            stat
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()
        # priority = self.aggregate_priority(p, batch.seq_len)

        if pred_weight > 0:
            if self.vdn:
                pred_loss1  = self.aux_task_vdn(
                    lstm_o,
                    batch.obs["own_hand"],
                    batch.obs["temperature"],
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + pred_weight * pred_loss1
            else:
                pred_loss = self.aux_task_iql(
                    lstm_o,
                    batch.obs["own_hand"],
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + pred_weight * pred_loss
        else:
            loss = rl_loss
        return loss, priority
