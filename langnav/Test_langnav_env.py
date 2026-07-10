from env import HuskyTerrainEnv

env = HuskyTerrainEnv(render_mode="human")
obs, info = env.reset()
print("Reset OK. Obs shape:", obs.shape)

for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Ran {i+1} steps without crashing.")
env.close()