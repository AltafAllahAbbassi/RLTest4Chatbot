import torch
import torch.nn.functional as F
import numpy as np
from RLTest4Chatbot.environments.dialogue_simulator import DialogueSimulator
from RLTest4Chatbot.agents.utils.utils import get_random_actions, get_actions
import argparse
from RLTest4Chatbot.agents.pdqn import QActor, ParamActor, PDQNAgent


class MultiPDQN(PDQNAgent):
    def __init__(self,
                 observation_space,
                 action_space,
                 top_k,
                 actor_class=QActor,
                 actor_kwargs={'hidden_layers': (128,),
                               'action_input_layer': 0},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={'hidden_layers': (128,),
                                     'squashing_function': False,
                                     'output_layer_init_std': 0.0001},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.9,
                 tau_actor=0.01,
                 tau_actor_param=0.001,
                 replay_memory_size=10000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.0001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,
                 loss_func=F.mse_loss,
                 clip_grad=10,
                 inverting_gradients=True,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device='cpu',
                 seed=1):
        super().__init__(observation_space,
                         action_space,
                         actor_class=actor_class,
                         actor_kwargs=actor_kwargs,
                         actor_param_class=actor_param_class,
                         actor_param_kwargs=actor_param_kwargs,
                         epsilon_initial=epsilon_initial,
                         epsilon_final=epsilon_final,
                         epsilon_steps=epsilon_steps,
                         batch_size=batch_size,
                         gamma=gamma,
                         tau_actor=tau_actor,
                         tau_actor_param=tau_actor_param,
                         replay_memory_size=replay_memory_size,
                         learning_rate_actor=learning_rate_actor,
                         learning_rate_actor_param=learning_rate_actor_param,
                         initial_memory_threshold=initial_memory_threshold,
                         use_ornstein_noise=use_ornstein_noise,
                         loss_func=loss_func,
                         clip_grad=clip_grad,
                         inverting_gradients=inverting_gradients,
                         zero_index_gradients=zero_index_gradients,
                         indexed=indexed,
                         weighted=weighted,
                         average=average,
                         random_weighted=random_weighted,
                         device=device,
                         seed=seed)

        self.top_k = top_k

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)
            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()
            # if rnd < self.epsilon: # this is the correct form
            if rnd < self.epsilon:
                actions = get_random_actions(self.num_actions, self.top_k)
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                               self.action_parameter_max_numpy))
            else:
                Q_a = self.actor.forward(state.unsqueeze(
                    0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                actions = get_actions(Q_a[0], self.top_k)

            all_action_parameters = all_action_parameters.cpu().data.numpy()
        return actions, all_action_parameters

    def step(self, state, action, reward, next_state, next_action, terminal):
        self._step += 1
        self._add_sample(state, action, reward,
                         next_state, next_action, terminal)
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        # assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(
            state, action, reward, next_state, terminal=terminal)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, help='Random seed.', type=int)
    parser.add_argument('--evaluation_episodes', default=10,
                        help='Episodes over which to evaluate after training.', type=int)  # episodes = 1000
    parser.add_argument('--episodes', default=10,
                        help='Number of epsiodes.', type=int)  # 20000
    parser.add_argument('--batch_size', default=1,
                        help='Minibatch size.', type=int)  # 128
    parser.add_argument('--gamma', default=0.9,
                        help='Discount factor.', type=float)
    parser.add_argument('--inverting_gradients', default=True,
                        help='Use inverting gradients scheme instead of squashing function.', type=bool)
    parser.add_argument('--initial-memory-threshold', default=2, help='Number of transitions required to start learning.',
                        type=int)
    parser.add_argument('--use_ornstein_noise', default=True,
                        help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
    parser.add_argument('--replay_memory_size', default=1000,
                        help='Replay memory size in transitions.', type=int)
    parser.add_argument('--epsilon_steps', default=1000,
                        help='Number of episodes over which to linearly anneal epsilon.', type=int)
    parser.add_argument('--epsilon_final', default=0.01,
                        help='Final epsilon value.', type=float)
    parser.add_argument('--tau_actor', default=0.1,
                        help='Soft target network update averaging factor.', type=float)
    parser.add_argument('--tau-actor_param', default=0.001,
                        help='Soft target network update averaging factor.', type=float)  # 0.001
    # 0.001/0.0001 learns faster but tableaus faster too
    parser.add_argument('--learning_rate_actor', default=0.001,
                        help="Actor network learning rate.", type=float)
    parser.add_argument('--learning_rate_actor_param', default=0.0001,
                        help="Critic network learning rate.", type=float)  # 0.00001
    parser.add_argument('--initialise_params', default=True,
                        help='Initialise action parameters.', type=bool)
    parser.add_argument('--clip_grad', default=10.,
                        help="Parameter gradient clipping limit.", type=float)
    parser.add_argument('--indexed', default=False,
                        help='Indexed loss function.', type=bool)
    parser.add_argument('--weighted', default=False,
                        help='Naive weighted loss function.', type=bool)
    parser.add_argument('--average', default=False,
                        help='Average weighted loss function.', type=bool)
    parser.add_argument('--random_weighted', default=False,
                        help='Randomly weighted loss function.', type=bool)
    parser.add_argument('--zero_index_gradients', default=False,
                        help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
    parser.add_argument('--action_input_layer', default=0,
                        help='Which layer to input action parameters.', type=int)
    parser.add_argument('--layers', default=(128,),
                        help='Duplicate action-parameter inputs.')
    parser.add_argument('--save_freq', default=0,
                        help='How often to save models (0 = never).', type=int)
    parser.add_argument('--save_dir', default="RLTest4chatbot/results/",
                        help='Output directory.', type=str)
    parser.add_argument('--render_freq', default=100,
                        help='How often to render / save frames of an episode.', type=int)
    parser.add_argument('--title', default="PDDQN",
                        help="Prefix of output files", type=str)
    parser.add_argument('--disp_freq', default=5,
                        help="When to display results", type=int)  # display results

    parser.add_argument('--top-k_freq', default=3,
                        help="the number of discrete actions", type=int)  # display results

    args = parser.parse_args()
    env = DialogueSimulator()
    agent = MultiPDQN(env.observation_space.spaces[0], env.action_space,
                      actor_kwargs={'hidden_layers': args.layers,
                                    'action_input_layer': args.action_input_layer, },
                      actor_param_kwargs={'hidden_layers': args.layers,
                                          'squashing_function': False,
                                          'output_layer_init_std': 0.0001, },)

    print(env.state)
    state = np.array(env.state, dtype=np.float32, copy=False)
    action = agent.act(state)
    ret = env.step(action)
    next_state, reward, terminal, _ = ret
    print(next_state)
