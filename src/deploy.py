import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env import BuildingEnv
from settings import PROJECT_ROOT

with open("config.json", "r") as f:
    config = json.load(f)

config["start_time"] = 0
config["stop_time"] = 288 * 67 * 300  # 67 days

env = BuildingEnv(config)

# Load the trained agent
model = PPO.load("ppo_simplehouse_model")

obs, _ = env.reset()

# Data storage for plotting
times = []
zone_temps = []
outdoor_temps = []
occupancy = []
heating_signals = []
solar_rads = []
energy_prices = []

terminated = False
truncated = False

print("Starting deployment simulation...")

while not (terminated or truncated):

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    current_time = env.current_time

    # Save only the last week
    if current_time > 288 * 60 * 300:
        times.append(current_time / 3600.0)

        zone_temps.append(info["zone_temp"])
        outdoor_temps.append(info["weaBus.TDryBul"] - 273.15)
        occupancy.append(info["booltoOcc.y"])
        solar_rads.append(info.get("weaBus.HDirNor", 0.0))
        energy_prices.append(info["energy_price"])
        heating_signals.append(action[0])

env.close()
print("Simulation finished. Generating plots...")

time_array = np.array(times)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1, 2, 1]})

ax1.plot(time_array, zone_temps, label="Indoor Temp (°C)", color="tab:blue", linewidth=2)
ax1.fill_between(time_array, 19, 21, color="green", alpha=0.15, label="Comfort Band (Occupied)")
ax1.axhline(y=17, color="red", linestyle=":", alpha=0.5, label="Min Temp (Unoccupied)")
ax1_occ = ax1.twinx()
ax1_occ.fill_between(time_array, 0, occupancy, color="orange", alpha=0.15, step="post", label="Occupancy")
ax1_occ.set_yticks([])
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("Indoor Temperature vs Comfort Constraints")
ax1.grid(True, alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_occ.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")


ax2.plot(time_array, heating_signals, label="Heating Signal (u)", color="tab:red", alpha=0.8, linewidth=1.5)
ax2.set_ylabel("Signal (0-1)")
ax2.set_title("HVAC Control Action")
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

ax3.plot(time_array, outdoor_temps, label="Outdoor Temp (°C)", color="tab:gray", linestyle="-", linewidth=2)
ax3.set_ylabel("Outdoor Temp (°C)", color="tab:gray")
ax3.tick_params(axis='y', labelcolor="tab:gray")
ax3_rad = ax3.twinx()
ax3_rad.fill_between(time_array, 0, solar_rads, color="gold", alpha=0.3, label="Solar Irradiance (W/m²)")
ax3_rad.set_ylabel("Irradiance (W/m²)", color="goldenrod")
ax3_rad.tick_params(axis='y', labelcolor="goldenrod")

ax3.set_title("Environmental Conditions")
ax3.grid(True, alpha=0.3)
lines3, labels3 = ax3.get_legend_handles_labels()
lines3r, labels3r = ax3_rad.get_legend_handles_labels()
ax3.legend(lines3 + lines3r, labels3 + labels3r, loc="upper left")

ax4.plot(time_array, energy_prices, label="Energy Price (€/kWh)", color="tab:green", linewidth=2,
         drawstyle="steps-post")
ax4.fill_between(time_array, 0, energy_prices, color="tab:green", alpha=0.1, step="post")
ax4.set_ylabel("Price (€/kWh)")
ax4.set_xlabel("Time (Hours)")
ax4.set_title("Energy Cost Signal")
ax4.set_ylim(0, max(energy_prices) * 1.2)
ax4.grid(True, alpha=0.3)
ax4.legend(loc="upper left")

plt.xlim(time_array[0], time_array[-1])
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "results", "deployment_results.png"))
plt.show()
