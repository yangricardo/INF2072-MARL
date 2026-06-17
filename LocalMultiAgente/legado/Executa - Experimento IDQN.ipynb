import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
import imageio
from tqdm import tqdm
import os
from datetime import datetime

# Configurações
ENV_NAME = "rware-medium-6ag-hard-v2"
NUM_AGENTS = 4  # Usaremos apenas 4 dos 6 agentes disponíveis
EPISODES = 5000
MAX_STEPS = 500
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 50000  # Decaimento lento é crucial para recompensas esparsas [citation:5]
BATCH_SIZE = 256
BUFFER_SIZE = 200000
TARGET_UPDATE = 1000
HIDDEN_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Usando dispositivo: {DEVICE}")

# Rede Neural para DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, agent_id):
        self.agent_id = agent_id
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0
        self.action_dim = action_dim
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + GAMMA * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Função para processar observação
def process_observation(obs):
    """Converte observação multiagente em vetor flat para cada agente"""
    processed = []
    for agent_obs in obs:
        if isinstance(agent_obs, dict):
            # Se for dicionário, concatenar valores
            flat = np.concatenate([v.flatten() if hasattr(v, 'flatten') else np.array([v]) 
                                   for v in agent_obs.values()])
        else:
            flat = agent_obs.flatten()
        processed.append(flat[:128])  # Limitar tamanho
    return processed

# Função para coletar estatísticas
def collect_stats(env, agents, num_episodes=10):
    """Avalia agentes treinados"""
    episode_returns = []
    episode_deliveries = []
    agent_returns = {i: [] for i in range(NUM_AGENTS)}
    agent_deliveries = {i: [] for i in range(NUM_AGENTS)}
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = process_observation(obs)
        episode_return = 0
        deliveries = 0
        agent_ep_returns = [0] * NUM_AGENTS
        agent_ep_deliveries = [0] * NUM_AGENTS
        
        for step in range(MAX_STEPS):
            actions = []
            for i in range(NUM_AGENTS):
                action = agents[i].select_action(obs[i], 0.0)  # Sem exploração
                actions.append(action)
            
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_obs = process_observation(next_obs)
            
            episode_return += sum(rewards)
            for i in range(NUM_AGENTS):
                agent_ep_returns[i] += rewards[i]
                # Detectar entregas (recompensa positiva)
                if rewards[i] > 0:
                    agent_ep_deliveries[i] += 1
                    deliveries += 1
            
            if terminated or truncated:
                break
            
            obs = next_obs
        
        episode_returns.append(episode_return)
        episode_deliveries.append(deliveries)
        for i in range(NUM_AGENTS):
            agent_returns[i].append(agent_ep_returns[i])
            agent_deliveries[i].append(agent_ep_deliveries[i])
    
    return {
        'returns': episode_returns,
        'deliveries': episode_deliveries,
        'agent_returns': agent_returns,
        'agent_deliveries': agent_deliveries
    }

# Função para salvar vídeo
def save_video(env, agents, filename, num_steps=500):
    """Gera vídeo da execução dos agentes"""
    frames = []
    obs, _ = env.reset()
    obs = process_observation(obs)
    
    for step in range(num_steps):
        # Renderizar frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Selecionar ações
        actions = []
        for i in range(NUM_AGENTS):
            action = agents[i].select_action(obs[i], 0.0)
            actions.append(action)
        
        obs, rewards, terminated, truncated, _ = env.step(actions)
        obs = process_observation(obs)
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Salvar vídeo
    with imageio.get_writer(filename, fps=10, codec='libx264'):
        for frame in frames:
            imageio.imwrite(filename, frame)
            writer.append_data(frame)
    
    print(f"Vídeo salvo: {filename}")

# Função para plotar gráficos
def plot_results(episode_returns, epsilon_history, step_counts, 
                 episode_deliveries, success_rates, agent_returns, 
                 agent_deliveries):
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Reward Total por Episódio
    axes[0, 0].plot(episode_returns, alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Reward Total por Episódio')
    axes[0, 0].set_xlabel('Episódio')
    axes[0, 0].set_ylabel('Reward Total')
    # Média móvel
    if len(episode_returns) > 100:
        moving_avg = np.convolve(episode_returns, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(episode_returns)), moving_avg, 'r-', linewidth=2, label='Média móvel 100 ep.')
        axes[0, 0].legend()
    
    # 2. Gráfico de Exploração (Epsilon)
    axes[0, 1].plot(epsilon_history)
    axes[0, 1].set_title('Decaimento da Exploração (ε)')
    axes[0, 1].set_xlabel('Passos de Treinamento')
    axes[0, 1].set_ylabel('ε')
    axes[0, 1].set_yscale('log')
    
    # 3. Steps por Episódio
    axes[0, 2].plot(step_counts)
    axes[0, 2].set_title('Steps por Episódio')
    axes[0, 2].set_xlabel('Episódio')
    axes[0, 2].set_ylabel('Steps')
    
    # 4. Número de Entregas
    axes[1, 0].plot(episode_deliveries)
    axes[1, 0].set_title('Entregas por Episódio')
    axes[1, 0].set_xlabel('Episódio')
    axes[1, 0].set_ylabel('Número de Entregas')
    if len(episode_deliveries) > 100:
        moving_avg_del = np.convolve(episode_deliveries, np.ones(100)/100, mode='valid')
        axes[1, 0].plot(range(99, len(episode_deliveries)), moving_avg_del, 'r-', linewidth=2)
    
    # 5. Taxa de Sucesso
    axes[1, 1].plot(success_rates)
    axes[1, 1].set_title('Taxa de Sucesso (≥1 entrega)')
    axes[1, 1].set_xlabel('Episódio')
    axes[1, 1].set_ylabel('Taxa de Sucesso')
    axes[1, 1].set_ylim([0, 1])
    
    # 6. Recompensa por Robô
    ax6 = axes[1, 2]
    agent_returns_array = np.array(list(agent_returns.values()))
    agent_means = agent_returns_array.mean(axis=1)
    agent_stds = agent_returns_array.std(axis=1)
    x_pos = np.arange(NUM_AGENTS)
    ax6.bar(x_pos, agent_means, yerr=agent_stds, capsize=5, alpha=0.7)
    ax6.set_title('Recompensa Média por Robô')
    ax6.set_xlabel('Robô')
    ax6.set_ylabel('Recompensa Média')
    ax6.set_xticks(x_pos)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    
    # Gráfico adicional: Entregas por Robô
    plt.figure(figsize=(10, 6))
    agent_del_array = np.array(list(agent_deliveries.values()))
    del_means = agent_del_array.mean(axis=1)
    del_stds = agent_del_array.std(axis=1)
    plt.bar(x_pos, del_means, yerr=del_stds, capsize=5, alpha=0.7, color='green')
    plt.title('Entregas Médias por Robô')
    plt.xlabel('Robô')
    plt.ylabel('Entregas por Episódio')
    plt.xticks(x_pos)
    plt.tight_layout()
    plt.savefig('deliveries_per_robot.png', dpi=150)
    plt.show()

# Função principal de treinamento
def train():
    print(f"Inicializando ambiente {ENV_NAME}...")
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    
    # Obter dimensões do espaço de observação/ação
    sample_obs, _ = env.reset()
    sample_obs = process_observation(sample_obs)
    state_dim = len(sample_obs[0])
    action_dim = env.action_space[0].n
    
    print(f"Dimensão do estado: {state_dim}")
    print(f"Dimensão da ação: {action_dim}")
    print(f"Número de agentes: {NUM_AGENTS}")
    
    # Criar agentes
    agents = [DQNAgent(state_dim, action_dim, i) for i in range(NUM_AGENTS)]
    
    # Métricas
    episode_returns = []
    epsilon_history = []
    step_counts = []
    episode_deliveries = []
    success_rates = []
    agent_returns_history = {i: [] for i in range(NUM_AGENTS)}
    agent_deliveries_history = {i: [] for i in range(NUM_AGENTS)}
    
    epsilon = EPSILON_START
    total_steps = 0
    
    print("\nIniciando treinamento...")
    
    for episode in tqdm(range(EPISODES), desc="Treinamento"):
        obs, _ = env.reset()
        obs = process_observation(obs)
        episode_return = 0
        step_count = 0
        deliveries = 0
        agent_ep_returns = [0] * NUM_AGENTS
        agent_ep_deliveries = [0] * NUM_AGENTS
        
        for step in range(MAX_STEPS):
            # Selecionar ações
            actions = []
            for i in range(NUM_AGENTS):
                action = agents[i].select_action(obs[i], epsilon)
                actions.append(action)
            
            # Executar ações
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_obs = process_observation(next_obs)
            
            # Armazenar experiências
            for i in range(NUM_AGENTS):
                agents[i].memory.push(obs[i], actions[i], rewards[i], 
                                      next_obs[i], terminated or truncated)
                episode_return += rewards[i]
                agent_ep_returns[i] += rewards[i]
                if rewards[i] > 0:
                    agent_ep_deliveries[i] += 1
                    deliveries += 1
            
            # Otimizar agentes
            for i in range(NUM_AGENTS):
                agents[i].optimize()
            
            step_count += 1
            total_steps += 1
            
            # Decair epsilon
            if total_steps < EPSILON_DECAY:
                epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * total_steps / EPSILON_DECAY
            else:
                epsilon = EPSILON_END
            
            if terminated or truncated:
                break
            
            obs = next_obs
        
        # Atualizar target networks periodicamente
        if episode % TARGET_UPDATE == 0:
            for i in range(NUM_AGENTS):
                agents[i].update_target()
        
        # Registrar métricas
        episode_returns.append(episode_return)
        epsilon_history.append(epsilon)
        step_counts.append(step_count)
        episode_deliveries.append(deliveries)
        success_rate = sum(1 for d in episode_deliveries[-100:] if d > 0) / min(100, len(episode_deliveries))
        success_rates.append(success_rate)
        
        for i in range(NUM_AGENTS):
            agent_returns_history[i].append(agent_ep_returns[i])
            agent_deliveries_history[i].append(agent_ep_deliveries[i])
        
        # Logging periódico
        if (episode + 1) % 100 == 0:
            avg_return = np.mean(episode_returns[-100:])
            avg_deliveries = np.mean(episode_deliveries[-100:])
            print(f"\nEpisódio {episode+1}: Retorno médio={avg_return:.2f}, "
                  f"Entregas médias={avg_deliveries:.2f}, ε={epsilon:.3f}, "
                  f"Steps={total_steps}")
    
    env.close()
    
    print("\nTreinamento concluído! Gerando gráficos...")
    
    # Gerar gráficos
    plot_results(episode_returns, epsilon_history, step_counts, 
                 episode_deliveries, success_rates, agent_returns_history, 
                 agent_deliveries_history)
    
    print("Salvando modelo...")
    for i, agent in enumerate(agents):
        torch.save(agent.policy_net.state_dict(), f'dqn_agent_{i}.pth')
    
    # Avaliação final
    print("\nAvaliando agentes treinados...")
    eval_results = collect_stats(env, agents, num_episodes=50)
    print(f"Retorno médio (avaliação): {np.mean(eval_results['returns']):.2f} ± {np.std(eval_results['returns']):.2f}")
    print(f"Entregas médias (avaliação): {np.mean(eval_results['deliveries']):.2f} ± {np.std(eval_results['deliveries']):.2f}")
    
    # Gerar vídeo
    print("\nGerando vídeo da solução...")
    # Recriar ambiente com renderização
    video_env = gym.make(ENV_NAME, render_mode='rgb_array')
    save_video(video_env, agents, 'solution_video.mp4', num_steps=MAX_STEPS)
    
    return agents, episode_returns, episode_deliveries

# Executar treinamento
if __name__ == "__main__":
    agents, returns, deliveries = train()