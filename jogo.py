import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodios, is_training=True, render=False):
    # Configura o ambiente
    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    # Inicializa ou carrega a tabela Q
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    # Hiperparâmetros
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay = 0.0001
    rng = np.random.default_rng()

    # Array para armazenar as recompensas por episódio
    rewards_per_episode = np.zeros(episodios)

    for episode in range(episodios):
        state = env.reset()[0]
        terminated = False
        truncated = False

        total_rewards = 0
        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            total_rewards += reward

            if is_training:
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay, 0)

        if epsilon == 0:
            learning_rate = 0.0001

        rewards_per_episode[episode] = total_rewards

    env.close()

    # Gráfico das recompensas acumuladas
    rolling_rewards = np.zeros(episodios)
    for t in range(episodios):
        rolling_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(rolling_rewards)
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Desempenho do Agente')
    plt.savefig('taxi.png')

    # Salva a tabela Q treinada
    if is_training:
        with open("taxi.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(15000)  # Treinamento

    run(10, is_training=False, render=True)  # Execução com renderização
