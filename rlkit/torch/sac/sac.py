import os
import torch
import torch.optim as optim
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
from collections import OrderedDict
from itertools import product
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm, OMRLOnlineAdaptAlgorithm,OfflineMetaRLAlgorithmEnsemble,OMRLOnlineAdaptAlgorithmEnsemble
from rlkit.torch.brac import divergences
from rlkit.torch.brac import utils

import torch.nn.functional as F

class FOCALSoftActorCritic(OfflineMetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        self.recurrent                      = kwargs['recurrent']
        self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.use_information_bottleneck     = kwargs['use_information_bottleneck']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.use_brac                       = kwargs['use_brac']
        self.use_value_penalty              = kwargs['use_value_penalty']
        self.alpha_max                      = kwargs['alpha_max']
        self._c_iter                        = kwargs['c_iter']
        self.train_alpha                    = kwargs['train_alpha']
        self._target_divergence             = kwargs['target_divergence']
        self.alpha_init                     = kwargs['alpha_init']
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']
        self.z_loss_weight                  = kwargs['z_loss_weight']
        self.max_entropy                    = kwargs['max_entropy']
        self.allow_backward_z               = kwargs['allow_backward_z']
        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()

        self.qf1, self.qf2, self.vf, self.c = nets[1:]
        self.target_vf                      = self.vf.copy()

        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
        self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.context_optimizer              = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf, self.c]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context

    ##### Training #####
    def _do_training(self, indices, zloss=False):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context, zloss=zloss)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b # index in task * batch dim
            for j in range(i+1, len(indices)):
                idx_j = j * b # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)

    def _take_step(self, indices, context, zloss=False):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)
                #policy_outputs, task_z, task_z_vars, task_zp, task_zp_vars = self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z)
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z)
        self.loss["div_estimate"] = torch.mean(div_estimate).item()
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)

        self.loss["c_loss"] = c_loss.item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
                #target_v_values = self.target_vf(t, b, next_obs, task_z)
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)
        elif zloss:
            z_loss = self.z_loss_weight * self.z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()
        self.context_optimizer.step()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                        self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean

            #z_mean1 = ptu.get_numpy(self.agent.z_means[0][0])
            #z_mean2 = ptu.get_numpy(self.agent.z_means[0][1])
            #z_mean3 = ptu.get_numpy(self.agent.z_means[0][2])
            #z_mean4 = ptu.get_numpy(self.agent.z_means[0][3])
            #z_mean5 = ptu.get_numpy(self.agent.z_means[0][4])

            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            #self.eval_statistics['Z mean train1'] = z_mean1
            #self.eval_statistics['Z mean train2'] = z_mean2
            #self.eval_statistics['Z mean train3'] = z_mean3
            #self.eval_statistics['Z mean train4'] = z_mean4
            #self.eval_statistics['Z mean train5'] = z_mean5

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            if self.use_information_bottleneck:
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            elif zloss:
                self.eval_statistics['Z Loss'] = ptu.get_numpy(z_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions',  ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis',        ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha',          ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate',   ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict()
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        try:
            self.agent.context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder_itr_{}.pth'.format(epoch))))
            self.agent.policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            self.vf.load_state_dict(torch.load(os.path.join(path, 'vf_itr_{}.pth'.format(epoch))))
            self.target_vf.load_state_dict(torch.load(os.path.join(path, 'target_vf_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False


class FOCALSoftActorCriticModel(OfflineMetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            **kwargs
        )

        self.latent_dim = latent_dim
        self.soft_target_tau = kwargs['soft_target_tau']
        self.policy_mean_reg_weight = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight = kwargs['policy_pre_activation_weight']
        self.recurrent = kwargs['recurrent']
        self.kl_lambda = kwargs['kl_lambda']
        self._divergence_name = kwargs['divergence_name']
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        self.use_brac = kwargs['use_brac']
        self.use_value_penalty = kwargs['use_value_penalty']
        self.alpha_max = kwargs['alpha_max']
        self._c_iter = kwargs['c_iter']
        self.train_alpha = kwargs['train_alpha']
        self._target_divergence = kwargs['target_divergence']
        self.alpha_init = kwargs['alpha_init']
        self.alpha_lr = kwargs['alpha_lr']
        self.policy_lr = kwargs['policy_lr']
        self.qf_lr = kwargs['qf_lr']
        self.vf_lr = kwargs['vf_lr']
        self.c_lr = kwargs['c_lr']
        self.context_lr = kwargs['context_lr']
        self.z_loss_weight = kwargs['z_loss_weight']
        self.max_entropy = kwargs['max_entropy']
        self.allow_backward_z = kwargs['allow_backward_z']
        self.loss = {}
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        self.qf1, self.qf2, self.vf, self.c,self.reward_decoder,self.transition_decoder = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.reward_decoder_optimizer = optimizer_class(self.reward_decoder.parameters(), lr=self.qf_lr)
        self.transition_decoder_optimizer = optimizer_class(self.transition_decoder.parameters(), lr=self.qf_lr)
        self.vf_optimizer = optimizer_class(self.vf.parameters(), lr=self.vf_lr)
        self.c_optimizer = optimizer_class(self.c.parameters(), lr=self.c_lr)
        self.context_optimizer = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.pred_loss = nn.MSELoss()
        self._num_steps = 0
        self._visit_num_steps_train = 10
        self._alpha_var = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf, self.c,self.reward_decoder,self.transition_decoder]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        # print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in
                   indices]
        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for
                   idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in
                   context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context

    ##### Training #####
    def _do_training(self, indices, zloss=False):
        mb_size = self.embedding_mini_batch_size  # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context, zloss=zloss)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b  # index in task * batch dim
            for j in range(i + 1, len(indices)):
                idx_j = j * b  # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1 / (torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss / (pos_cnt + epsilon) + neg_z_loss / (neg_cnt + epsilon)

    def _take_step(self, indices, context, zloss=False):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(
            reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
            # policy_outputs, task_z, task_z_vars, task_zp, task_zp_vars = self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z)
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z)
        self.loss["div_estimate"] = torch.mean(div_estimate).item()
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)

        self.loss["c_loss"] = c_loss.item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
                # target_v_values = self.target_vf(t, b, next_obs, task_z)
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)
        elif zloss:
            z_loss = self.z_loss_weight * self.z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()
        self.context_optimizer.step()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        pred_rewardss = rewards.view(self.batch_size * num_tasks, -1)
        # print(task_z.shape,obs.shape,actions.shape)
        rew_pred = self.reward_decoder.forward(0,0,task_z.detach(), obs, actions)
        self.reward_decoder_optimizer.zero_grad()
        rew_loss = self.pred_loss(pred_rewardss, rew_pred) * 1
        rew_loss.backward()
        self.loss["reward_prediction_loss"] = torch.mean(rew_loss).item()
        # print('wwwwwwwwwwwwwwwwwww')
        self.reward_decoder_optimizer.step()

        self.transition_decoder_optimizer.zero_grad()
        trans_pred = self.transition_decoder.forward(0,0,task_z.detach(), obs, actions)
        trans_loss = self.pred_loss(next_obs, trans_pred) * 1
        trans_loss.backward()
        self.loss["transition_prediction_loss"] = torch.mean(trans_loss).item()
        self.transition_decoder_optimizer.step()

        self.train_prediction_loss = (rew_loss+trans_loss).detach().cpu().numpy()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                      self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean

            # z_mean1 = ptu.get_numpy(self.agent.z_means[0][0])
            # z_mean2 = ptu.get_numpy(self.agent.z_means[0][1])
            # z_mean3 = ptu.get_numpy(self.agent.z_means[0][2])
            # z_mean4 = ptu.get_numpy(self.agent.z_means[0][3])
            # z_mean5 = ptu.get_numpy(self.agent.z_means[0][4])

            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            # self.eval_statistics['Z mean train1'] = z_mean1
            # self.eval_statistics['Z mean train2'] = z_mean2
            # self.eval_statistics['Z mean train3'] = z_mean3
            # self.eval_statistics['Z mean train4'] = z_mean4
            # self.eval_statistics['Z mean train5'] = z_mean5

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            if self.use_information_bottleneck:
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            elif zloss:
                self.eval_statistics['Z Loss'] = ptu.get_numpy(z_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Reward Loss'] = np.mean(ptu.get_numpy(rew_loss))
            self.eval_statistics['Transition Loss'] = np.mean(ptu.get_numpy(trans_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions', ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha', ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate', ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict()
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        try:
            self.agent.context_encoder.load_state_dict(
                torch.load(os.path.join(path, 'context_encoder_itr_{}.pth'.format(epoch))))
            self.agent.policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            self.vf.load_state_dict(torch.load(os.path.join(path, 'vf_itr_{}.pth'.format(epoch))))
            self.target_vf.load_state_dict(torch.load(os.path.join(path, 'target_vf_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False


class FOCALSoftActorCriticModel2(OMRLOnlineAdaptAlgorithmEnsemble):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            **kwargs
        )

        self.latent_dim = latent_dim
        self.soft_target_tau = kwargs['soft_target_tau']
        self.policy_mean_reg_weight = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight = kwargs['policy_pre_activation_weight']
        self.recurrent = kwargs['recurrent']
        self.kl_lambda = kwargs['kl_lambda']
        self._divergence_name = kwargs['divergence_name']
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        self.use_brac = kwargs['use_brac']
        self.use_value_penalty = kwargs['use_value_penalty']
        self.alpha_max = kwargs['alpha_max']
        self._c_iter = kwargs['c_iter']
        self.train_alpha = kwargs['train_alpha']
        self._target_divergence = kwargs['target_divergence']
        self.alpha_init = kwargs['alpha_init']
        self.alpha_lr = kwargs['alpha_lr']
        self.policy_lr = kwargs['policy_lr']
        self.qf_lr = kwargs['qf_lr']
        self.vf_lr = kwargs['vf_lr']
        self.c_lr = kwargs['c_lr']
        self.context_lr = kwargs['context_lr']
        self.z_loss_weight = kwargs['z_loss_weight']
        self.max_entropy = kwargs['max_entropy']
        self.allow_backward_z = kwargs['allow_backward_z']
        self.num_ensemble = kwargs['num_ensemble']
        self.loss = {}
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        self.qf1, self.qf2, self.vf, self.c,self.reward_models, self.dynamic_models = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.reward_models_optimizer = optimizer_class(self.reward_models.parameters(), lr=self.qf_lr)
        self.dynamic_models_optimizer = optimizer_class(self.dynamic_models.parameters(), lr=self.qf_lr)
        self.vf_optimizer = optimizer_class(self.vf.parameters(), lr=self.vf_lr)
        self.c_optimizer = optimizer_class(self.c.parameters(), lr=self.c_lr)
        self.context_optimizer = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.pred_loss = nn.MSELoss()
        self._num_steps = 0
        self._visit_num_steps_train = 10
        self._alpha_var = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        net = self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf, self.c]
        for i in range(self.num_ensemble):
            net += [self.reward_models[i], self.dynamic_models[i]]
        return net

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        # print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in
                   indices]
        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for
                   idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in
                   context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context

    ##### Training #####
    def _do_training(self, indices, zloss=False):
        mb_size = self.embedding_mini_batch_size  # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context, zloss=zloss)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b  # index in task * batch dim
            for j in range(i + 1, len(indices)):
                idx_j = j * b  # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1 / (torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss / (pos_cnt + epsilon) + neg_z_loss / (neg_cnt + epsilon)

    def _take_step(self, indices, context, zloss=False):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(
            reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
            # policy_outputs, task_z, task_z_vars, task_zp, task_zp_vars = self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z)
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z)
        self.loss["div_estimate"] = torch.mean(div_estimate).item()
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)

        self.loss["c_loss"] = c_loss.item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
                # target_v_values = self.target_vf(t, b, next_obs, task_z)
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)
        elif zloss:
            z_loss = self.z_loss_weight * self.z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()
        self.context_optimizer.step()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        pred_rewardss = rewards.view(self.batch_size * num_tasks, -1)

        model_loss = None
        for i in range(self.num_ensemble):
            obs1, actions1, rewards1, next_obs1, terms1 = self.sample_sac(indices)

            t, b, _ = obs1.size()
            obs1 = obs1.view(t * b, -1)
            actions1 = actions1.view(t * b, -1)
            next_obs1 = next_obs1.view(t * b, -1)
            pred_rewardss1 = rewards1.view(self.batch_size * num_tasks, -1)

            rew_pred1 = self.reward_models[i].forward(0, 0, task_z.detach(), obs1, actions1)
            next_obs_pred1=self.dynamic_models[i].forward(0, 0, task_z.detach(), obs1, actions1)


            rew_loss = self.pred_loss(pred_rewardss1, rew_pred1) * 1
            dynamic_loss = self.pred_loss(next_obs1, next_obs_pred1) * 1

            if model_loss is None:
                model_loss = rew_loss
            else:
                model_loss = model_loss + rew_loss + dynamic_loss


        self.reward_models_optimizer.zero_grad()
        self.dynamic_models_optimizer.zero_grad()
        model_loss.backward()
        self.loss["reward_prediction_loss"] = torch.mean(model_loss).item()
        # print('wwwwwwwwwwwwwwwwwww')
        self.reward_models_optimizer.step()
        self.dynamic_models_optimizer.step()



        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                      self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean

            # z_mean1 = ptu.get_numpy(self.agent.z_means[0][0])
            # z_mean2 = ptu.get_numpy(self.agent.z_means[0][1])
            # z_mean3 = ptu.get_numpy(self.agent.z_means[0][2])
            # z_mean4 = ptu.get_numpy(self.agent.z_means[0][3])
            # z_mean5 = ptu.get_numpy(self.agent.z_means[0][4])

            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            # self.eval_statistics['Z mean train1'] = z_mean1
            # self.eval_statistics['Z mean train2'] = z_mean2
            # self.eval_statistics['Z mean train3'] = z_mean3
            # self.eval_statistics['Z mean train4'] = z_mean4
            # self.eval_statistics['Z mean train5'] = z_mean5

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            if self.use_information_bottleneck:
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            elif zloss:
                self.eval_statistics['Z Loss'] = ptu.get_numpy(z_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Reward Loss'] = np.mean(ptu.get_numpy(model_loss))
            # self.eval_statistics['Transition Loss'] = np.mean(ptu.get_numpy(trans_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions', ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha', ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate', ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            reward_model1=self.reward_models[0].state_dict(),
            reward_model2=self.reward_models[1].state_dict(),
            reward_model3=self.reward_models[2].state_dict(),
            reward_model4=self.reward_models[3].state_dict(),
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        try:
            self.agent.context_encoder.load_state_dict(
                torch.load(os.path.join(path, 'context_encoder_itr_{}.pth'.format(epoch))))
            self.agent.policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            self.vf.load_state_dict(torch.load(os.path.join(path, 'vf_itr_{}.pth'.format(epoch))))
            self.target_vf.load_state_dict(torch.load(os.path.join(path, 'target_vf_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False

class FOCALSoftActorCriticOnlineAdapt(OMRLOnlineAdaptAlgorithm):

    # is the same as FOCALSoftActorCriticModel except of OMRLOnlineAdaptAlgorithm

    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            **kwargs
        )

        self.latent_dim = latent_dim
        self.soft_target_tau = kwargs['soft_target_tau']
        self.policy_mean_reg_weight = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight = kwargs['policy_pre_activation_weight']
        self.recurrent = kwargs['recurrent']
        self.kl_lambda = kwargs['kl_lambda']
        self._divergence_name = kwargs['divergence_name']
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        self.use_brac = kwargs['use_brac']
        self.use_value_penalty = kwargs['use_value_penalty']
        self.alpha_max = kwargs['alpha_max']
        self._c_iter = kwargs['c_iter']
        self.train_alpha = kwargs['train_alpha']
        self._target_divergence = kwargs['target_divergence']
        self.alpha_init = kwargs['alpha_init']
        self.alpha_lr = kwargs['alpha_lr']
        self.policy_lr = kwargs['policy_lr']
        self.qf_lr = kwargs['qf_lr']
        self.vf_lr = kwargs['vf_lr']
        self.c_lr = kwargs['c_lr']
        self.context_lr = kwargs['context_lr']
        self.z_loss_weight = kwargs['z_loss_weight']
        self.max_entropy = kwargs['max_entropy']
        self.allow_backward_z = kwargs['allow_backward_z']
        self.loss = {}
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        self.qf1, self.qf2, self.vf, self.c,self.reward_decoder,self.transition_decoder = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.reward_decoder_optimizer = optimizer_class(self.reward_decoder.parameters(), lr=self.qf_lr)
        self.transition_decoder_optimizer = optimizer_class(self.transition_decoder.parameters(), lr=self.qf_lr)
        self.vf_optimizer = optimizer_class(self.vf.parameters(), lr=self.vf_lr)
        self.c_optimizer = optimizer_class(self.c.parameters(), lr=self.c_lr)
        self.context_optimizer = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.pred_loss = nn.MSELoss()
        self._num_steps = 0
        self._visit_num_steps_train = 10
        self._alpha_var = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf, self.c,self.reward_decoder,self.transition_decoder]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        # print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in
                   indices]
        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for
                   idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in
                   context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context

    ##### Training #####
    def _do_training(self, indices, zloss=False):
        mb_size = self.embedding_mini_batch_size  # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context, zloss=zloss)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b  # index in task * batch dim
            for j in range(i + 1, len(indices)):
                idx_j = j * b  # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1 / (torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss / (pos_cnt + epsilon) + neg_z_loss / (neg_cnt + epsilon)

    def _take_step(self, indices, context, zloss=False):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(
            reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
            # policy_outputs, task_z, task_z_vars, task_zp, task_zp_vars = self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z)
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z)
        self.loss["div_estimate"] = torch.mean(div_estimate).item()
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)

        self.loss["c_loss"] = c_loss.item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
                # target_v_values = self.target_vf(t, b, next_obs, task_z)
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)
        elif zloss:
            z_loss = self.z_loss_weight * self.z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()
        self.context_optimizer.step()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        pred_rewardss = rewards.view(self.batch_size * num_tasks, -1)
        # print(task_z.shape,obs.shape,actions.shape)
        rew_pred = self.reward_decoder.forward(0,0,task_z.detach(), obs, actions)
        self.reward_decoder_optimizer.zero_grad()
        rew_loss = self.pred_loss(pred_rewardss, rew_pred) * 1
        rew_loss.backward()
        self.loss["reward_prediction_loss"] = torch.mean(rew_loss).item()
        # print('wwwwwwwwwwwwwwwwwww')
        self.reward_decoder_optimizer.step()

        self.transition_decoder_optimizer.zero_grad()
        trans_pred = self.transition_decoder.forward(0,0,task_z.detach(), obs, actions)
        trans_loss = self.pred_loss(next_obs, trans_pred) * 1
        trans_loss.backward()
        self.loss["transition_prediction_loss"] = torch.mean(trans_loss).item()
        self.transition_decoder_optimizer.step()

        self.train_prediction_loss = (rew_loss+trans_loss).detach().cpu().numpy()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                      self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean

            # z_mean1 = ptu.get_numpy(self.agent.z_means[0][0])
            # z_mean2 = ptu.get_numpy(self.agent.z_means[0][1])
            # z_mean3 = ptu.get_numpy(self.agent.z_means[0][2])
            # z_mean4 = ptu.get_numpy(self.agent.z_means[0][3])
            # z_mean5 = ptu.get_numpy(self.agent.z_means[0][4])

            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            # self.eval_statistics['Z mean train1'] = z_mean1
            # self.eval_statistics['Z mean train2'] = z_mean2
            # self.eval_statistics['Z mean train3'] = z_mean3
            # self.eval_statistics['Z mean train4'] = z_mean4
            # self.eval_statistics['Z mean train5'] = z_mean5

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            if self.use_information_bottleneck:
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            elif zloss:
                self.eval_statistics['Z Loss'] = ptu.get_numpy(z_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Reward Loss'] = np.mean(ptu.get_numpy(rew_loss))
            self.eval_statistics['Transition Loss'] = np.mean(ptu.get_numpy(trans_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions', ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha', ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate', ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict()
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        try:
            self.agent.context_encoder.load_state_dict(
                torch.load(os.path.join(path, 'context_encoder_itr_{}.pth'.format(epoch))))
            self.agent.policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            self.vf.load_state_dict(torch.load(os.path.join(path, 'vf_itr_{}.pth'.format(epoch))))
            self.target_vf.load_state_dict(torch.load(os.path.join(path, 'target_vf_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False


class CPEARL(OMRLOnlineAdaptAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            wandb_project_name=None,
            wandb_run_name=None,
            csv_name=None,
            **kwargs
    ):

        # add self.task_id_decoder and modify the _take_step

        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            wandb_project_name=wandb_project_name,
            wandb_run_name=wandb_run_name,
            csv_name=csv_name,
            **kwargs
        )

        self.latent_dim = latent_dim
        self.soft_target_tau = kwargs['soft_target_tau']
        self.policy_mean_reg_weight = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight = kwargs['policy_pre_activation_weight']
        # True
        self.recurrent = kwargs['recurrent']
        self.kl_lambda = kwargs['kl_lambda']
        self._divergence_name = kwargs['divergence_name']
        self.use_information_bottleneck = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        self.use_brac = kwargs['use_brac']
        self.use_value_penalty = kwargs['use_value_penalty']
        self.alpha_max = kwargs['alpha_max']
        self._c_iter = kwargs['c_iter']
        self.train_alpha = kwargs['train_alpha']
        self._target_divergence = kwargs['target_divergence']
        self.alpha_init = kwargs['alpha_init']
        self.alpha_lr = kwargs['alpha_lr']
        self.policy_lr = kwargs['policy_lr']
        self.qf_lr = kwargs['qf_lr']
        self.vf_lr = kwargs['vf_lr']
        self.c_lr = kwargs['c_lr']
        self.context_lr = kwargs['context_lr']
        self.z_loss_weight = kwargs['z_loss_weight']
        self.max_entropy = kwargs['max_entropy']
        self.allow_backward_z = kwargs['allow_backward_z']
        self.is_predict_task_id = kwargs['is_predict_task_id']
        self.is_true_sparse_rewards = kwargs['is_true_sparse_rewards']
        self.loss = {}
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        self.qf1, self.qf2, self.vf, self.c, self.reward_decoder, self.transition_decoder, self.task_id_decoder = \
            nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.reward_decoder_optimizer = optimizer_class(self.reward_decoder.parameters(), lr=self.qf_lr)
        self.transition_decoder_optimizer = optimizer_class(self.transition_decoder.parameters(), lr=self.qf_lr)
        self.task_id_decoder_optimizer = optimizer_class(self.task_id_decoder.parameters(), lr=self.qf_lr)
        self.vf_optimizer = optimizer_class(self.vf.parameters(), lr=self.vf_lr)
        self.c_optimizer = optimizer_class(self.c.parameters(), lr=self.c_lr)
        self.context_optimizer = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)

        self.context_score_optimizer = optimizer_class(self.agent.context_score_encoder.parameters(), lr=self.context_lr)

        self.pred_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self._num_steps = 0
        self._visit_num_steps_train = 100
        self._alpha_var = torch.tensor(1.)

        self.l1_loss = nn.L1Loss()

        self.VideoRLCS_loss = nn.SmoothL1Loss()

        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.csv_name = csv_name

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf, self.c,
                                                     self.reward_decoder, self.transition_decoder, self.task_id_decoder]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        # print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def unpack_batch_sac(self, batch, sparse_reward=False, true_sparse_reward=False):
        ''' unpack a batch and return individual elements for sac '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        if true_sparse_reward:
            r_label = batch['sparse_rewards'][None, ...]
        else:
            r_label = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t, r_label]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in
                   indices]
        unpacked = [self.unpack_batch_sac(batch, true_sparse_reward=self.is_true_sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        # self.recurrent = True
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for
            idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in
                   context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context

    # For lap
    def sample_context_lap(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        # self.recurrent = True

        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent))
            for
            idx in indices]

        batches = []
        next_batches = []
        for idx in indices:
            batch, next_batch = self.enc_replay_buffer.random_batch_for_lap(idx,
                                                                            batch_size=self.embedding_batch_size,
                                                                            sequence=self.recurrent)
            batch = ptu.np_to_pytorch_batch(batch)
            next_batch = ptu.np_to_pytorch_batch(next_batch)

            batches.append(batch)
            next_batches.append(next_batch)

        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in
                   context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context

        # action_reward = torch.cat(context[1:-2], dim=2)
        # next_context = torch.cat([context[-2], action_reward], dim=2)

        next_context = [self.unpack_batch(next_batch, sparse_reward=self.sparse_rewards) for next_batch in
                        next_batches]
        # group like elements together
        next_context = [[x[i] for x in next_context] for i in range(len(next_context[0]))]
        next_context = [torch.cat(x, dim=0) for x in
                        next_context]  # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)

        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
            next_context = torch.cat(next_context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
            next_context = torch.cat(next_context[:-2], dim=2)
        # self.meta_batch * self.embedding_batch_size * sum_dim(o, a, r, no, t)
        return context, next_context

    ##### Training #####
    def _do_training(self, indices, zloss=False):
        # embedding_mini_batch_size = 1024
        mb_size = self.embedding_mini_batch_size  # NOTE: not meta batch!
        # embedding_batch_size = 1024
        # num_updates = 1
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        # num_updates = 1
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars, wandb_stat = self._take_step(indices, context, zloss=zloss)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars, wandb_stat

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms, _ = self.sample_sac(indices)

        # run inference in networks
        if self.use_information_bottleneck:
            policy_outputs, task_z = self.agent(obs, context, task_indices=indices)
        else:
            policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def z_loss(self, indices, task_z, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b  # index in task * batch dim
            for j in range(i + 1, len(indices)):
                idx_j = j * b  # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1 / (torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss / (pos_cnt + epsilon) + neg_z_loss / (neg_cnt + epsilon)

    def compute_fea_lap_loss(self, indices):
        context, next_context = self.sample_context_lap(indices)

        task_z = self.agent.only_infer_context_embeddings(context)
        next_task_z = self.agent.only_infer_context_embeddings(next_context)

        loss = (task_z - next_task_z).pow(2).mean()
        # Cov = torch.matmul(task_z, task_z.T)

        # Feature Lap
        Cov = torch.matmul(task_z.T, task_z)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        # Org Lap
        lap_Cov = torch.matmul(task_z, task_z.T)
        lap_I = torch.eye(*lap_Cov.size(), device=lap_Cov.device)
        lap_off_diag = ~lap_I.bool()
        lap_orth_loss_diag = - 2 * lap_Cov.diag().mean()
        lap_orth_loss_offdiag = lap_Cov[lap_off_diag].pow(2).mean()
        lap_orth_loss = lap_orth_loss_offdiag + lap_orth_loss_diag
        loss += lap_orth_loss

        return loss

    def sample_trajs(self, indices):
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size, sequence=self.recurrent)) for
                   idx in indices]

        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        if self.use_next_obs_in_context:
            context = torch.cat(unpacked[:-1], dim=2)
        else:
            context = torch.cat(unpacked[:-2], dim=2)

        return unpacked, context


    def TACO_loss(self, indices, K=4):
        trajs, trajs_context = self.sample_trajs(indices)

        context_embedding = self.agent.only_infer_context_embeddings_TACO(trajs_context)
        cross_entropy_loss = nn.CrossEntropyLoss()

        loss = 0.
        for i in range(context_embedding.shape[1]-K+1):
            # [task_num, 1, context_dim]
            last_context = context_embedding[:,i+K-1,:].view(context_embedding.shape[0], 1, context_embedding.shape[-1])
            # if i == 0:
            #     other_context = context_embedding[:,1:,:]
            # elif i == (context_embedding.shape[1] - 1):
            #     other_context = context_embedding[:,:-1, :]
            # else:
            #     other_context = torch.cat([context_embedding[:,:i,:],context_embedding[:,i+1:,:]], dim=1)
            # prior_context = torch.cat([context_embedding[:,i,:], context_embedding[:,i+1,:]], dim=1)
            prior_context = context_embedding[:, i:i+K-1,:]

            # [task_num, 1, context_dim]
            # other_context = torch.mean(other_context, dim=1, keepdim=True)
            prior_context = torch.mean(prior_context, dim=1, keepdim=True)

            # print("squeeze_prior_context.shape:", prior_context.shape)

            # other_context = other_context.squeeze()
            last_context = last_context.squeeze()
            prior_context = prior_context.squeeze()

            # print("prior_context.shape:", prior_context.shape)
            Wz = torch.matmul(self.agent.W, prior_context.T)
            logits = torch.matmul(last_context, Wz)
            logits = logits - torch.max(logits, 1)[0][:, None]

            labels = torch.arange(logits.shape[0]).long().cuda()
            loss += cross_entropy_loss(logits, labels)

        loss = loss / (context_embedding.shape[1]-K+1)

        #     I_ = torch.eye(context_embedding.shape[0]).cuda()
        #
        #     Wz = torch.matmul(self.agent.W, other_context.T.detach())
        #     logits = torch.matmul(last_context, Wz)
        #
        #     logits_I = logits * I_
        #     logits_I = torch.sum(logits_I, dim=-1)
        #     logits = torch.sum(logits, dim=-1)
        #
        #     loss += torch.sum(torch.log(logits_I/logits))
        #
        # loss = -loss/context_embedding.shape[1]

        return loss

    def compute_fea_lap_loss(self, indices):
        context, next_context = self.sample_context_lap(indices)

        task_z = self.agent.only_infer_context_embeddings(context)
        next_task_z = self.agent.only_infer_context_embeddings(next_context)

        loss = (task_z - next_task_z).pow(2).mean()
        # Cov = torch.matmul(task_z, task_z.T)

        # feature_lap
        Cov = torch.matmul(task_z.T, task_z)

        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        # org_lap
        lap_Cov = torch.matmul(task_z, task_z.T)
        lap_I = torch.eye(*lap_Cov.size(), device=lap_Cov.device)
        lap_off_diag = ~lap_I.bool()
        lap_orth_loss_diag = - 2 * lap_Cov.diag().mean()
        lap_orth_loss_offdiag = lap_Cov[lap_off_diag].pow(2).mean()
        lap_orth_loss = lap_orth_loss_offdiag + lap_orth_loss_diag
        loss += lap_orth_loss

        return loss


    def _take_step(self, indices, context, zloss=False):

        # 通过 wandb 进行记录训练过程中出现的loss 情况
        wandb_stat = {}

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(
            reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms, r_label = self.sample_sac(indices)
        # False
        if self.use_information_bottleneck:
            policy_outputs, task_z, masks, masks_sigmoid, reverse_task_z = self.agent(obs, context, task_indices=indices, for_update=True)
        # True
        else:
            policy_outputs, task_z, task_z_vars, masks, masks_sigmoid, reverse_task_z = self.agent(obs, context, task_indices=indices, for_update=True)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        masks = masks.view(masks.shape[0],masks.shape[1])

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()
        c_loss = self._divergence.dual_critic_loss(obs, new_actions, actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)

        self.loss["c_loss"] = c_loss.item()
        wandb_stat["c_loss"] = c_loss.item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
                # target_v_values = self.target_vf(t, b, next_obs, task_z)
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()
        wandb_stat["target_v_values"] = torch.mean(target_v_values).item()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        self.context_score_optimizer.zero_grad()

        self.reward_decoder_optimizer.zero_grad()

        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)
            self.loss["kl_loss"] = kl_loss.item()
            wandb_stat["kl_loss"] = kl_loss.item()

        if zloss:
            z_loss = self.z_loss_weight * self.z_loss(indices=indices, task_z=task_z, b=b)
            z_loss.backward(retain_graph=True)
            self.loss["z_loss"] = z_loss.item()
            wandb_stat["z_loss"] = z_loss.item()


        Ort_loss_matrix = torch.abs(masks.mm(masks.t()))
        Ort_loss_matrix = (torch.ones_like(Ort_loss_matrix).cuda() - torch.eye(Ort_loss_matrix.size(0)).cuda()) * Ort_loss_matrix
        orthogonal_loss = torch.mean(torch.triu(Ort_loss_matrix, diagonal=1))
        # orthogonal_weight = 5e-4
        orthogonal_loss = 1.0 * orthogonal_loss
        self.loss["orthogonal_loss"] = torch.mean(orthogonal_loss).item()
        wandb_stat["orthogonal_loss"] = torch.mean(orthogonal_loss).item()

        # l1_weight = 1e-2
        # loss_l1 = 1.0 * torch.linalg.norm(masks, ord=1)
        # loss_l1 = 1.0 * F.normalize(masks_sigmoid, p=1, dim=-1)
        # loss_l1 = 1.0 * self.l1_loss(masks_sigmoid)
        loss_l1 = 1.0 * torch.norm(masks_sigmoid, 1)
        self.loss["loss_l1"] = torch.mean(loss_l1).item()
        wandb_stat["loss_l1"] = torch.mean(loss_l1).item()
        loss_l1 = torch.mean(loss_l1)

        pred_rewardss = r_label.view(self.batch_size * num_tasks, -1)
        rew_pred = self.reward_decoder.forward(0, 0, task_z, obs, actions)
        # rew_loss = self.pred_loss(pred_rewardss, rew_pred) * 1
        rew_loss = self.VideoRLCS_loss(pred_rewardss, rew_pred) * 1.0
        # rew_loss = self.pred_loss(pred_rewardss, rew_pred) * 1.0
        # rew_loss.backward(retain_graph=True)
        self.loss["reward_prediction_loss"] = torch.mean(rew_loss).item()
        wandb_stat["reward_prediction_loss"] = torch.mean(rew_loss).item()

        # 原本的 reward_decoder 的对应的 loss
        rew_pred_ = self.reward_decoder.forward(0, 0, task_z.detach(), obs, actions)
        rew_loss_ = self.pred_loss(pred_rewardss, rew_pred_) * 1.0
        rew_loss_.backward(retain_graph=True)
        self.loss["rew_loss_"] = torch.mean(rew_loss_).item()
        wandb_stat["rew_loss_"] = torch.mean(rew_loss_).item()


        # # use min reward for negative
        # # [task_num, trajectory_len, 1] -> [task_num, trajectory_len]
        # rewards_ = r_label.view(num_tasks, -1)
        # # [task_num]
        # min_rewards = torch.min(rewards_, dim=-1)[0]
        # # [task_num] -> [task_num, 1]
        # min_rewards = min_rewards.unsqueeze(-1)
        # # [task_num, 1] -> [task_num, 1, 1] -> [task_num, trajectory_len, 1]
        # min_rewards = min_rewards.unsqueeze(-1)
        # min_rewards = min_rewards.repeat(1, self.batch_size, 1).view(self.batch_size * num_tasks, 1)
        # reverse_rew_pred = self.reward_decoder.forward(0, 0, reverse_task_z, obs, actions)
        # reverse_rew_loss = self.VideoRLCS_loss(min_rewards, reverse_rew_pred) * 1.0
        # # reverse_rew_loss = self.pred_loss(min_rewards, reverse_rew_pred) * 1.0
        # self.loss["reverse_rew_pred"] = torch.mean(reverse_rew_pred).item()
        # wandb_stat["reverse_rew_pred"] = torch.mean(reverse_rew_pred).item()

        # # use max reward for negative
        # # [task_num, trajectory_len, 1] -> [task_num, trajectory_len]
        # rewards_ = r_label.view(num_tasks, -1)
        # # [task_num]
        # min_rewards = torch.max(rewards_, dim=-1)[0]
        # # [task_num] -> [task_num, 1]
        # min_rewards = min_rewards.unsqueeze(-1)
        # # [task_num, 1] -> [task_num, 1, 1] -> [task_num, trajectory_len, 1]
        # min_rewards = min_rewards.unsqueeze(-1)
        # min_rewards = min_rewards.repeat(1, self.batch_size, 1).view(self.batch_size * num_tasks, 1)
        # reverse_rew_pred = self.reward_decoder.forward(0, 0, reverse_task_z, obs, actions)
        # reverse_rew_loss = self.VideoRLCS_loss(min_rewards, reverse_rew_pred) * 1.0
        # # reverse_rew_loss = self.pred_loss(min_rewards, reverse_rew_pred) * 1.0
        # self.loss["reverse_rew_pred"] = torch.mean(reverse_rew_pred).item()
        # wandb_stat["reverse_rew_pred"] = torch.mean(reverse_rew_pred).item()


        # use random reward for negative
        # [task_num, trajectory_len, 1] -> [task_num, trajectory_len]
        rewards_ = r_label.view(num_tasks, -1)
        sp = rewards_.shape
        noise = torch.normal(mean=0., std=0.5, size=sp).cuda()
        min_rewards = noise + rewards_
        min_rewards = min_rewards.view(self.batch_size * num_tasks, 1)
        # # [task_num]
        # min_rewards = torch.min(rewards_, dim=-1)[0]
        # # [task_num] -> [task_num, 1]
        # min_rewards = min_rewards.unsqueeze(-1)
        # # [task_num, 1] -> [task_num, 1, 1] -> [task_num, trajectory_len, 1]
        # min_rewards = min_rewards.unsqueeze(-1)
        # min_rewards = min_rewards.repeat(1, self.batch_size, 1).view(self.batch_size * num_tasks, 1)
        reverse_rew_pred = self.reward_decoder.forward(0, 0, reverse_task_z, obs, actions)
        reverse_rew_loss = self.VideoRLCS_loss(min_rewards, reverse_rew_pred) * 1.0
        # reverse_rew_loss = self.pred_loss(min_rewards, reverse_rew_pred) * 1.0
        self.loss["reverse_rew_pred"] = torch.mean(reverse_rew_pred).item()
        wandb_stat["reverse_rew_pred"] = torch.mean(reverse_rew_pred).item()

        # For M1
        # 四种全部
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-4 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # HC
        # VideoRLCS_loss = orthogonal_loss * 1.0 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e4 + \
        #                  reverse_rew_loss * 1e2

        #

        # assembly-v2
        # VideoRLCS_loss = orthogonal_loss * 0.0 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 1e2 + \
        #                  reverse_rew_loss * 1e-2

        # basketball-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # bin-picking-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # box-close-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # button-press
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # button-press-wall
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # button-press-topdown-wall-v2
        # VideoRLCS_loss = orthogonal_loss * 1e-4 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        # coffee-button-v2
        # VideoRLCS_loss = orthogonal_loss * 1e2 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        # coffee-pull-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # coffee-push-v2
        # VideoRLCS_loss = orthogonal_loss * 0.0 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 1.0 + \
        #                  reverse_rew_loss * 1e-3

        # dial-turn-v2
        # VideoRLCS_loss = orthogonal_loss * 1.0 + \
        #                  loss_l1 * 1e-4 + \
        #                  rew_loss * 1.0 + \
        #                  reverse_rew_loss * 1e-4

        # disassemble-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # door-close-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # door-lock-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # door-unlock-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # door-open-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # drawer-close-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # drawer-open-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # faucet-close-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # faucet-open-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # hammer-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # hand-insert-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # handle-press-side-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # handle-press-v2
        # VideoRLCS_loss = orthogonal_loss * 0.0 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 1e-1 + \
        #                  reverse_rew_loss * 0.0

        # handle-pull-side-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # handle-pull-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # lever-pull-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # peg-unplug-side-v2
        # VideoRLCS_loss = orthogonal_loss * 0.0 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        # pick-out-of-hole-v2
        # VideoRLCS_loss = orthogonal_loss * 1e-3 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        # pick-place-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # pick-place-wall-v2
        # VideoRLCS_loss = orthogonal_loss * 0.0 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 1e-2 + \
        #                  reverse_rew_loss * 0.0

        # plate-slide-back-side-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # plate-slide-back-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # plate-slide-side-v2
        # VideoRLCS_loss = orthogonal_loss * 1e-1 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        # plate-slide-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # push-back-v2
        # VideoRLCS_loss = orthogonal_loss * 1e-4 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        # push-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # push-wall-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # reach-v2
        VideoRLCS_loss = orthogonal_loss * 1e1 + \
                         loss_l1 * 1e-2 + \
                         rew_loss * 1e1 + \
                         reverse_rew_loss * 1e2

        # shelf-place-v2
        # VideoRLCS_loss = orthogonal_loss * 1e-4 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        # soccer-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # stick-pull-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # stick-push-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # sweep-into-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # sweep-v2
        # VideoRLCS_loss = orthogonal_loss * 1e1 + \
        #                  loss_l1 * 1e-2 + \
        #                  rew_loss * 1e1 + \
        #                  reverse_rew_loss * 1e2

        # window-close-v2
        # VideoRLCS_loss = orthogonal_loss * 0.0 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 1.0 + \
        #                  reverse_rew_loss * 1e-3

        # window-open-v2
        # VideoRLCS_loss = orthogonal_loss * 1e-2 + \
        #                  loss_l1 * 0.0 + \
        #                  rew_loss * 0.0 + \
        #                  reverse_rew_loss * 0.0

        #

        # 单种
        # VideoRLCS_loss = orthogonal_loss * 1e4

        # VideoRLCS_loss = loss_l1 * 1e-4

        # VideoRLCS_loss = rew_loss * 1e-4

        # VideoRLCS_loss = reverse_rew_loss * 1e-4

        # 三种组合
        # VideoRLCS_loss = loss_l1 * 0.01 + \
        #                  rew_loss * 1e4 + \
        #                  reverse_rew_loss * 1e2

        # VideoRLCS_loss = orthogonal_loss * 10.0 + \
        #                  rew_loss * 1e4 + \
        #                  reverse_rew_loss * 1e2

        # VideoRLCS_loss = orthogonal_loss * 10.0 + \
        #                  loss_l1 * 0.01 + \
        #                  reverse_rew_loss * 1e2

        # VideoRLCS_loss = orthogonal_loss * 10.0 + \
        #                  loss_l1 * 0.01 + \
        #                  rew_loss * 1e4

        # 两种组合
        # VideoRLCS_loss = orthogonal_loss * 10.0 + \
        #                  loss_l1 * 0.01

        # VideoRLCS_loss = orthogonal_loss * 1e-3 + \
        #                  rew_loss * 1e-4

        # VideoRLCS_loss = orthogonal_loss * 10.0 + \
        #                  reverse_rew_loss * 1e2

        # VideoRLCS_loss = loss_l1 * 0.01 + \
        #                  rew_loss * 1e4

        # VideoRLCS_loss = loss_l1 * 0.01 + \
        #                  reverse_rew_loss * 1e2

        # VideoRLCS_loss = rew_loss * 1e-1 + \
        #                  reverse_rew_loss * 1e-4

        VideoRLCS_loss.backward(retain_graph=True)
        self.loss["VideoRLCS_loss"] = torch.mean(VideoRLCS_loss).item()
        wandb_stat["VideoRLCS_loss"] = torch.mean(VideoRLCS_loss).item()

        # lap_loss = self.compute_fea_lap_loss(indices)
        # lap_loss = lap_loss * 1e-2
        # lap_loss.backward(retain_graph=True)
        # self.loss["lap_loss"] = lap_loss.item()
        # wandb_stat["lap_loss"] = lap_loss.item()

        taco_loss = self.TACO_loss(indices)
        self.loss["taco_loss"] = taco_loss.item()
        wandb_stat["taco_loss"] = taco_loss.item()
        taco_loss = taco_loss * 1.0
        taco_loss.backward(retain_graph=True)

        if self.is_predict_task_id:
            task_id = indices.reshape(-1, 1).repeat(self.batch_size, axis=1).reshape(-1,)
            task_id = torch.tensor(task_id, device=ptu.device, dtype=torch.long)
            pred_task_id = self.task_id_decoder.forward(0, 0, task_z)
            self.task_id_decoder_optimizer.zero_grad()
            task_id_loss = self.ce_loss(pred_task_id, task_id)
            task_id_loss.backward()
            self.loss["task_id_prediction_loss"] = torch.mean(task_id_loss).item()
            wandb_stat["task_id_prediction_loss"] = torch.mean(task_id_loss).item()

            self.task_id_decoder_optimizer.step()
        if not self.allow_backward_z:
            self.context_optimizer.step()
            self.context_score_optimizer.step()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        wandb_stat["qf_loss"] = qf_loss.item()
        wandb_stat["q_target"] = torch.mean(q_target).item()
        wandb_stat["q1_pred"] = torch.mean(q1_pred).item()
        wandb_stat["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        if self.allow_backward_z:
            self.context_optimizer.step()
            self.context_score_optimizer.step()

        # pred_rewardss = r_label.view(self.batch_size * num_tasks, -1)
        # # print(task_z.shape,obs.shape,actions.shape)
        # rew_pred = self.reward_decoder.forward(0,0,task_z.detach(), obs, actions)
        # self.reward_decoder_optimizer.zero_grad()
        # rew_loss = self.pred_loss(pred_rewardss, rew_pred) * 1
        # # rew_loss.backward()
        # rew_loss.backward(retain_graph=True)
        # self.loss["reward_prediction_loss"] = torch.mean(rew_loss).item()
        # wandb_stat["reward_prediction_loss"] = torch.mean(rew_loss).item()
        #
        # self.reward_decoder_optimizer.step()
        self.reward_decoder_optimizer.step()

        self.transition_decoder_optimizer.zero_grad()
        trans_pred = self.transition_decoder.forward(0,0,task_z.detach(), obs, actions)
        trans_loss = self.pred_loss(next_obs, trans_pred) * 1
        trans_loss.backward()
        self.loss["transition_prediction_loss"] = torch.mean(trans_loss).item()
        wandb_stat["transition_prediction_loss"] = torch.mean(trans_loss).item()

        self.transition_decoder_optimizer.step()

        self.train_prediction_loss = (rew_loss+trans_loss).detach().cpu().numpy()
        wandb_stat["train_prediction_loss"] = torch.mean(rew_loss).item() + torch.mean(trans_loss).item()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                      self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        wandb_stat["vf_loss"] = vf_loss.item()
        wandb_stat["v_target"] = torch.mean(v_target).item()
        wandb_stat["v_pred"] = torch.mean(v_pred).item()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        wandb_stat["policy_loss"] = policy_loss.item()

        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        wandb_stat["a_loss"] = a_loss.item()

        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean

            # z_mean1 = ptu.get_numpy(self.agent.z_means[0][0])
            # z_mean2 = ptu.get_numpy(self.agent.z_means[0][1])
            # z_mean3 = ptu.get_numpy(self.agent.z_means[0][2])
            # z_mean4 = ptu.get_numpy(self.agent.z_means[0][3])
            # z_mean5 = ptu.get_numpy(self.agent.z_means[0][4])

            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            # self.eval_statistics['Z mean train1'] = z_mean1
            # self.eval_statistics['Z mean train2'] = z_mean2
            # self.eval_statistics['Z mean train3'] = z_mean3
            # self.eval_statistics['Z mean train4'] = z_mean4
            # self.eval_statistics['Z mean train5'] = z_mean5

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            if self.use_information_bottleneck:
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            elif zloss:
                self.eval_statistics['Z Loss'] = ptu.get_numpy(z_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Reward Loss'] = np.mean(ptu.get_numpy(rew_loss))
            self.eval_statistics['Transition Loss'] = np.mean(ptu.get_numpy(trans_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions', ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha', ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate', ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars), wandb_stat

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict()
        )
        return snapshot

    def load_epoch_model(self, epoch, log_dir):
        path = log_dir
        print(epoch)
        try:
            self.agent.context_encoder.load_state_dict(
                torch.load(os.path.join(path, 'context_encoder_itr_{}.pth'.format(epoch))))
            self.agent.policy.load_state_dict(torch.load(os.path.join(path, 'policy_itr_{}.pth'.format(epoch))))
            self.qf1.load_state_dict(torch.load(os.path.join(path, 'qf1_itr_{}.pth'.format(epoch))))
            self.qf2.load_state_dict(torch.load(os.path.join(path, 'qf2_itr_{}.pth'.format(epoch))))
            self.vf.load_state_dict(torch.load(os.path.join(path, 'vf_itr_{}.pth'.format(epoch))))
            self.target_vf.load_state_dict(torch.load(os.path.join(path, 'target_vf_itr_{}.pth'.format(epoch))))
            return True
        except:
            print("epoch: {} is not ready".format(epoch))
            return False
