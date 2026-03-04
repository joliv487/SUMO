import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---- SUMO / TraCI setup ----
SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    raise RuntimeError(
        "SUMO_HOME is not set. Set it to your SUMO folder (contains 'bin' and 'tools')."
    )

tools_path = os.path.join(SUMO_HOME, "tools")
if tools_path not in sys.path:
    sys.path.append(tools_path)

import traci


class SumoEnv(gym.Env):
    """
    Minimal SUMO RL environment (single traffic light control).

    - Loads a SUMO config (.sumocfg)
    - Chooses a TLS that has >1 phase (if possible)
    - Observation: queue (halting vehicles) on up to obs_lanes controlled lanes (padded)
    - Action: phase index for chosen TLS (Discrete(num_phases))
    - Reward: negative total queue
    - Steps the sim forward decision_interval seconds each action
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(self, sumocfg: str, gui: bool = False, obs_lanes: int = 10, decision_interval: int = 5):
        super().__init__()

        self.sumocfg = os.path.abspath(sumocfg)
        self.gui = bool(gui)
        self.obs_lanes = int(obs_lanes)
        self.decision_interval = int(decision_interval)

        self.sumoBinary = "sumo-gui" if self.gui else "sumo"

        # Discovered on reset()
        self.tls_id: str | None = None
        self.controlled_lanes: list[str] = []
        self.num_phases: int = 1

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(self.obs_lanes,), dtype=np.float32
        )
        # Placeholder; updated in reset() once we know phase count
        self.action_space = spaces.Discrete(2)

    def _start_sumo(self):
        # --no-step-log keeps console clean
        traci.start([self.sumoBinary, "-c", self.sumocfg, "--no-step-log", "true"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        self._start_sumo()

        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            traci.close()
            raise RuntimeError("No traffic lights found in this SUMO scenario.")

        # Pick a TLS with >1 phase if possible
        chosen = None
        chosen_phases = 0
        for tid in tls_list:
            try:
                logic = traci.trafficlight.getAllProgramLogics(tid)[0]
                nph = len(logic.phases)
            except Exception:
                continue
            if nph > 1:
                chosen = tid
                chosen_phases = nph
                break

        # Fallback: use first TLS even if it has 1 phase
        if chosen is None:
            chosen = tls_list[0]
            logic = traci.trafficlight.getAllProgramLogics(chosen)[0]
            chosen_phases = len(logic.phases)

        self.tls_id = chosen
        self.num_phases = max(1, int(chosen_phases))

        # Controlled lanes for observations
        self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        # Set correct action space
        self.action_space = spaces.Discrete(self.num_phases)

        # Helpful debug print (safe to leave on while you're setting up)
        print(f"Using TLS: {self.tls_id} phases: {self.num_phases} lanes: {len(self.controlled_lanes)}")

        obs = self._get_state()
        return obs, {}

    def _lane_queue(self, lane_id: str) -> float:
        # "halting" vehicles is a robust queue proxy
        return float(traci.lane.getLastStepHaltingNumber(lane_id))

    def _get_state(self) -> np.ndarray:
        lanes = self.controlled_lanes or []
        queues = [self._lane_queue(l) for l in lanes[: self.obs_lanes]]

        # Pad to fixed length
        if len(queues) < self.obs_lanes:
            queues.extend([0.0] * (self.obs_lanes - len(queues)))

        return np.array(queues, dtype=np.float32)

    def step(self, action):
        if self.tls_id is None:
            raise RuntimeError("Environment not reset() before step().")

        # Clamp action to valid range (prevents TraCI crash)
        a = int(action)
        if a < 0:
            a = 0
        elif a >= self.num_phases:
            a = self.num_phases - 1

        # Apply phase
        traci.trafficlight.setPhase(self.tls_id, a)

        # Advance simulation
        for _ in range(self.decision_interval):
            traci.simulationStep()

        state = self._get_state()
        reward = -float(np.sum(state))  # minimize queue
        done = traci.simulation.getMinExpectedNumber() == 0

        return state, reward, done, False, {}

    def close(self):
        if traci.isLoaded():
            traci.close()