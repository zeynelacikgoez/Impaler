# Impaler/ml/qmix.py
"""
Implementierung des QMIX-Algorithmus für kooperatives Multi-Agent Reinforcement Learning.

QMIX lernt eine globale, monotonische Action-Value Funktion Q_tot, die sich aus den
individuellen Q-Werten der einzelnen Agenten zusammensetzt. Dies ermöglicht
zentralisiertes Training bei dezentraler Ausführung. Geeignet für Szenarien,
in denen Agenten kooperieren müssen, um ein gemeinsames Ziel zu erreichen,
z.B. Ressourcen effizient zu nutzen oder einen globalen Plan zu erfüllen.

Die Reward-Funktion wird extern definiert und sollte das gemeinsame Ziel widerspiegeln.
"""

import random
import logging
import numpy as np
from collections import deque
from typing import Dict, Any, List, Tuple, Optional, Deque

# Prüfe und importiere PyTorch
_TORCH_ERROR_MSG = ("PyTorch nicht gefunden. QMIX erfordert PyTorch. "
                    "Installieren Sie PyTorch (pytorch.org).")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
    # Typ-Aliase für Lesbarkeit
    Tensor = torch.Tensor
    Module = nn.Module
    Optimizer = optim.Optimizer
except ImportError:
    TORCH_AVAILABLE = False
    # Platzhalter-Typen für Type Hints
    Tensor = Any
    Module = Any
    Optimizer = Any
    # Fehler wird erst bei Instanziierung geworfen, wenn Torch fehlt

# Logger für dieses Modul
logger = logging.getLogger(__name__)
if not TORCH_AVAILABLE:
     logger.warning(_TORCH_ERROR_MSG + " QMIX-Klassen sind nicht verfügbar.")


# --- QMIX Mixer Netzwerk ---
if TORCH_AVAILABLE:
    class QMIXNetwork(Module):
        """
        Das QMIX Mixing-Netzwerk.

        Kombiniert die individuellen Q-Werte der Agenten (agent_q_values) monoton
        zu einem globalen Q-Wert (q_total), unter Berücksichtigung des globalen
        Zustands (states). Nutzt Hypernetzwerke, um die Mischgewichte und Biases
        zustandsabhängig zu generieren.

        Stellt Monotonie sicher: d(Q_tot) / d(Q_i) >= 0 für alle Agenten i.

        Attributes:
            num_agents (int): Anzahl der Agenten.
            state_dim (int): Dimension des globalen Zustandsvektors.
            mixing_embed_dim (int): Dimension des internen Embedding-Raums für die Mischung.
            hyper_w1 (Module): Hypernetzwerk zur Generierung der Gewichte der ersten Mischschicht.
            hyper_w2 (Module): Hypernetzwerk zur Generierung der Gewichte der zweiten Mischschicht.
            hyper_b1 (Module): Hypernetzwerk zur Generierung des Bias der ersten Mischschicht.
            hyper_b2 (Module): Hypernetzwerk zur Generierung des Bias der zweiten Mischschicht.
        """
        num_agents: int
        state_dim: int
        mixing_embed_dim: int
        hyper_w1: Module
        hyper_w2: Module
        hyper_b1: Module
        hyper_b2: Module

        def __init__(self,
                     num_agents: int,
                     state_dim: int,
                     mixing_embed_dim: int = 64, # Erhöht für mehr Kapazität
                     hypernet_hidden_dim: int = 64): # Größe der Hidden Layer in Hypernetzen
            """
            Initialisiert das QMIX Mixing-Netzwerk.

            Args:
                num_agents: Anzahl der kooperierenden Agenten.
                state_dim: Dimension des globalen Zustandsvektors.
                mixing_embed_dim: Dimension des internen Embeddings für die Mischung.
                hypernet_hidden_dim: Größe der versteckten Schichten in den Hypernetzwerken.
            """
            super(QMIXNetwork, self).__init__()
            self.num_agents = num_agents
            self.state_dim = state_dim
            self.mixing_embed_dim = mixing_embed_dim
            self.hypernet_hidden_dim = hypernet_hidden_dim

            # Hypernetzwerke zur Generierung der Mischgewichte und Biases
            # W1: state_dim -> (num_agents * mixing_embed_dim)
            self.hyper_w1 = nn.Sequential(
                nn.Linear(self.state_dim, self.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hypernet_hidden_dim, self.num_agents * self.mixing_embed_dim)
            )
            # B1: state_dim -> mixing_embed_dim
            self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_embed_dim)

            # W2: state_dim -> mixing_embed_dim
            self.hyper_w2 = nn.Sequential(
                nn.Linear(self.state_dim, self.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hypernet_hidden_dim, self.mixing_embed_dim)
            )
            # B2: state_dim -> 1 (globaler Bias)
            # Separate Schicht für B2, da keine Aktivierungsfunktion danach folgt
            self.hyper_b2_layer = nn.Sequential(
                nn.Linear(self.state_dim, self.hypernet_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hypernet_hidden_dim, 1)
            )


        def forward(self, agent_q_values: Tensor, states: Tensor) -> Tensor:
            """
            Führt den Forward-Pass des Mixers durch.

            Args:
                agent_q_values: Individuelle Q-Werte der Agenten für die gewählten Aktionen.
                                Shape: (batch_size, num_agents).
                states: Globaler Zustand. Shape: (batch_size, state_dim).

            Returns:
                Globaler Q_tot-Wert. Shape: (batch_size, 1).
            """
            batch_size = agent_q_values.size(0)
            # Reshape Q-Werte für Batch-Matrixmultiplikation: (batch_size, 1, num_agents)
            agent_qs_reshaped = agent_q_values.view(batch_size, 1, self.num_agents)

            # --- Erste Mischschicht ---
            # Generiere Gewichte W1 (zustandsabhängig)
            w1 = self.hyper_w1(states)
            # Monotonie erzwingen: Gewichte müssen nicht-negativ sein -> Absolutwert
            w1 = torch.abs(w1)
            # Reshape W1: (batch_size, num_agents, mixing_embed_dim)
            w1 = w1.view(batch_size, self.num_agents, self.mixing_embed_dim)

            # Generiere Bias B1 (zustandsabhängig)
            b1 = self.hyper_b1(states)
            # Reshape B1: (batch_size, 1, mixing_embed_dim)
            b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

            # Erste Mischung: Q_i * W1 + B1
            # Ergebnis Shape: (batch_size, 1, mixing_embed_dim)
            hidden = torch.bmm(agent_qs_reshaped, w1) + b1
            # Aktivierungsfunktion (z.B. ELU oder ReLU)
            hidden = F.elu(hidden)

            # --- Zweite Mischschicht ---
            # Generiere Gewichte W2 (zustandsabhängig)
            w2 = self.hyper_w2(states)
             # Monotonie erzwingen
            w2 = torch.abs(w2)
            # Reshape W2: (batch_size, mixing_embed_dim, 1)
            w2 = w2.view(batch_size, self.mixing_embed_dim, 1)

            # Generiere Bias B2 (zustandsabhängig)
            # b2 = self.hyper_b2(states) # Veraltet, da state-independent bias
            b2 = self.hyper_b2_layer(states) # Verwende Layer für Bias
            # Reshape B2: (batch_size, 1, 1)
            b2 = b2.view(batch_size, 1, 1)

            # Zweite Mischung: hidden * W2 + B2
            # Ergebnis Shape: (batch_size, 1, 1)
            q_total = torch.bmm(hidden, w2) + b2

            # Reshape zu (batch_size, 1) für Konsistenz mit Loss-Funktion
            return q_total.view(batch_size, 1)


# --- QMIX Agent Koordinator ---
if TORCH_AVAILABLE:
    class QMIXAgent:
        """
        Koordinator für das QMIX-basierte Multi-Agent RL-Setup.

        Verwaltet die individuellen DQN-Agenten-Netzwerke und das zentrale
        QMIX Mixing-Netzwerk zur Berechnung eines gemeinsamen Q-Wertes.

        Attributes:
            num_agents (int): Anzahl der Agenten.
            state_dim (int): Dimension des globalen Zustands.
            obs_dim (int): Dimension der lokalen Beobachtung pro Agent.
            action_dim (int): Anzahl diskreter Aktionen pro Agent.
            hidden_dim (int): Größe der versteckten Layer in Agenten-Netzwerken.
            mixing_embed_dim (int): Embedding-Dimension im Mixer.
            learning_rate (float): Lernrate für alle Optimierer.
            gamma (float): Diskontfaktor für Bellman-Updates.
            batch_size (int): Größe der Mini-Batches für das Training.
            buffer_size (int): Maximale Größe des Replay Buffers.
            target_update_freq (int): Frequenz (in Lernschritten) für Target-Netzwerk-Updates.
            update_counter (int): Zähler für Lernschritte seit letztem Target-Update.
            device (torch.device): CPU oder GPU.
            buffer (Deque): Replay Buffer für Transitionen.
            agent_networks (List[Module]): Liste der Policy-Netzwerke der Agenten.
            target_networks (List[Module]): Liste der Target-Netzwerke der Agenten.
            mixer (QMIXNetwork): Das trainierbare Mixing-Netzwerk.
            target_mixer (QMIXNetwork): Das Ziel-Mixing-Netzwerk.
            agent_optimizers (List[Optimizer]): Liste der Optimierer für die Agenten-Netzwerke.
            mixer_optimizer (Optimizer): Optimierer für das Mixing-Netzwerk.
            loss_fn (Module): Verlustfunktion (typischerweise MSELoss).
            reward_history (Deque[float]): Historie der globalen Rewards.
            loss_history (Deque[float]): Historie der Mixer-Verlustwerte.
        """
        num_agents: int
        state_dim: int
        obs_dim: int
        action_dim: int
        hidden_dim: int
        mixing_embed_dim: int
        learning_rate: float
        gamma: float
        batch_size: int
        buffer_size: int
        target_update_freq: int
        update_counter: int
        device: torch.device
        buffer: Deque[Tuple[np.ndarray, List[np.ndarray], List[int], float, np.ndarray, List[np.ndarray], bool]]
        agent_networks: List[Module]
        target_networks: List[Module]
        mixer: QMIXNetwork
        target_mixer: QMIXNetwork
        agent_optimizers: List[Optimizer]
        mixer_optimizer: Optimizer
        loss_fn: Module # Typischerweise MSE Loss
        reward_history: Deque[float]
        loss_history: Deque[float]


        def __init__(self,
                     num_agents: int,
                     state_dim: int,
                     agent_obs_dim: int,
                     agent_action_dim: int,
                     agent_hidden_dim: int = 64,
                     mixing_embed_dim: int = 64, # Angepasst
                     hypernet_hidden_dim: int = 64, # Hinzugefügt
                     learning_rate: float = 0.0005, # Angepasst
                     gamma: float = 0.99,
                     batch_size: int = 128, # Angepasst
                     buffer_size: int = 10000, # Angepasst
                     target_update_freq: int = 200, # Angepasst
                     history_length: int = 1000):
            """
            Initialisiert den QMIXAgent Koordinator.

            Args:
                num_agents: Anzahl der Agenten.
                state_dim: Dimension des globalen Zustandsvektors.
                agent_obs_dim: Dimension der lokalen Beobachtung jedes Agenten.
                agent_action_dim: Anzahl der diskreten Aktionen jedes Agenten.
                agent_hidden_dim: Größe der Hidden Layer in den Agenten-Netzwerken.
                mixing_embed_dim: Embedding-Dimension im Mixer-Netzwerk.
                hypernet_hidden_dim: Größe der Hidden Layer in den Hypernetzwerken des Mixers.
                learning_rate: Lernrate für alle Optimierer.
                gamma: Diskontfaktor.
                batch_size: Größe der Trainings-Batches.
                buffer_size: Maximale Größe des Replay Buffers.
                target_update_freq: Frequenz der Target-Netzwerk Updates.
                history_length: Länge der gespeicherten Historien (Reward, Loss).

            Raises:
                ImportError: Wenn PyTorch nicht installiert ist.
            """
            if not TORCH_AVAILABLE:
                raise ImportError(_TORCH_ERROR_MSG)

            # Parameter validieren (Beispiele)
            if num_agents <= 0 or state_dim <= 0 or agent_obs_dim <= 0 or agent_action_dim <= 0:
                raise ValueError("Dimensionen und Agentenanzahl müssen positiv sein.")
            # ... weitere Validierungen ...

            self.num_agents = num_agents
            self.state_dim = state_dim
            self.obs_dim = agent_obs_dim
            self.action_dim = agent_action_dim
            self.hidden_dim = agent_hidden_dim
            self.mixing_embed_dim = mixing_embed_dim
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.batch_size = batch_size
            self.buffer_size = buffer_size
            self.target_update_freq = target_update_freq
            self.update_counter = 0

            # Gerät wählen
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"QMIX Agent verwendet Gerät: {self.device}")

            # Replay Buffer
            self.buffer = deque(maxlen=buffer_size)

            # Agenten-Netzwerke und Target-Netzwerke erstellen
            self.agent_networks = [self._build_agent_network().to(self.device) for _ in range(num_agents)]
            self.target_networks = [self._build_agent_network().to(self.device) for _ in range(num_agents)]
            self._update_target_networks() # Initial synchronisieren
            for net in self.target_networks: net.eval() # Target Nets im Eval-Modus

            # Mixer-Netzwerk und Target-Mixer erstellen
            self.mixer = QMIXNetwork(num_agents, state_dim, mixing_embed_dim, hypernet_hidden_dim).to(self.device)
            self.target_mixer = QMIXNetwork(num_agents, state_dim, mixing_embed_dim, hypernet_hidden_dim).to(self.device)
            self._update_target_mixer() # Initial synchronisieren
            self.target_mixer.eval() # Target Mixer im Eval-Modus

            # Optimierer erstellen
            self.agent_optimizers = [
                optim.Adam(net.parameters(), lr=learning_rate)
                for net in self.agent_networks
            ]
            self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=learning_rate)

            # Verlustfunktion
            self.loss_fn = nn.MSELoss()

            # Historien
            self.reward_history = deque(maxlen=history_length)
            self.loss_history = deque(maxlen=history_length)

            logger.info(f"QMIXAgent initialisiert für {num_agents} Agenten.")

        def _build_agent_network(self) -> Module:
            """Erstellt das Netzwerk für einen einzelnen Agenten (DRQN oder MLP)."""
            # Hier einfaches MLP, könnte durch Rekurrentes Netz (z.B. GRU) ersetzt werden für Historie
            return nn.Sequential(
                nn.Linear(self.obs_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim)
            )

        def _update_target_networks(self) -> None:
            """Kopiert Gewichte von Policy- zu Target-Netzwerken für alle Agenten."""
            logger.debug("Aktualisiere Agenten Target-Netzwerke...")
            for i in range(self.num_agents):
                self.target_networks[i].load_state_dict(self.agent_networks[i].state_dict())

        def _update_target_mixer(self) -> None:
            """Kopiert Gewichte vom Policy-Mixer zum Target-Mixer."""
            logger.debug("Aktualisiere Target-Mixer Netzwerk...")
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        @torch.no_grad() # Keine Gradienten für Aktionsauswahl nötig
        def select_actions(self, observations: List[np.ndarray], epsilon: float) -> List[int]:
            """
            Wählt Aktionen für alle Agenten ε-greedy.

            Args:
                observations: Liste der lokalen Beobachtungen für jeden Agenten.
                epsilon: Aktuelle Explorationsrate.

            Returns:
                Liste der gewählten Aktions-Indizes.
            """
            actions = []
            self.eval_mode() # Netzwerke in Eval-Modus schalten
            for i, obs in enumerate(observations):
                if random.random() < epsilon:
                    actions.append(random.randrange(self.action_dim))
                else:
                    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
                    q_values = self.agent_networks[i](obs_tensor)
                    action = q_values.argmax(dim=1).item()
                    actions.append(action)
            self.train_mode() # Zurück in Trainingsmodus
            return actions

        def store_transition(self,
                             state: np.ndarray,        # Globaler Zustand
                             observations: List[np.ndarray], # Lokale Beobachtungen pro Agent
                             actions: List[int],       # Gewählte Aktionen pro Agent
                             reward: float,            # Globaler Reward
                             next_state: np.ndarray,   # Globaler Folgezustand
                             next_observations: List[np.ndarray], # Lokale Folgebeobachtungen
                             done: bool) -> None:      # Globales Done-Signal
            """Speichert eine komplette Transition im Replay Buffer."""
            state = np.asarray(state, dtype=np.float32)
            next_state = np.asarray(next_state, dtype=np.float32)
            obs = [np.asarray(o, dtype=np.float32) for o in observations]
            next_obs = [np.asarray(no, dtype=np.float32) for no in next_observations]
            acts = list(actions) # Sicherstellen, dass es eine Liste ist
            rew = float(reward)
            dn = bool(done)

            self.buffer.append((state, obs, acts, rew, next_state, next_obs, dn))
            self.reward_history.append(rew) # Track global reward

        def _sample_batch(self) -> Optional[Tuple]:
             """Samplet einen Mini-Batch aus dem Replay Buffer."""
             if len(self.buffer) < self.batch_size:
                 logger.debug(f"Nicht genug Samples im Buffer ({len(self.buffer)}/{self.batch_size}).")
                 return None
             # Effizienteres Sampling
             batch_indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
             batch = [self.buffer[idx] for idx in batch_indices]
             # Entpacken
             states, observations, actions, rewards, next_states, next_observations, dones = zip(*batch)
             return states, observations, actions, rewards, next_states, next_observations, dones

        def _prepare_batch_tensors(self, batch_data: Tuple) -> Tuple:
             """Konvertiert einen Batch von NumPy Arrays/Listen zu PyTorch Tensoren."""
             states, observations, actions, rewards, next_states, next_observations, dones = batch_data

             state_tensor = torch.from_numpy(np.stack(states)).float().to(self.device)
             next_state_tensor = torch.from_numpy(np.stack(next_states)).float().to(self.device)
             reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
             done_tensor = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)

             # Agenten-spezifische Daten zu Batches zusammenfassen
             obs_tensors = []
             next_obs_tensors = []
             action_tensors = []
             for i in range(self.num_agents):
                  obs_tensors.append(torch.from_numpy(np.stack([obs[i] for obs in observations])).float().to(self.device))
                  next_obs_tensors.append(torch.from_numpy(np.stack([n_obs[i] for n_obs in next_observations])).float().to(self.device))
                  action_tensors.append(torch.LongTensor([act[i] for act in actions]).unsqueeze(1).to(self.device))

             return (state_tensor, obs_tensors, action_tensors, reward_tensor,
                     next_state_tensor, next_obs_tensors, done_tensor)

        def _get_chosen_action_qvals(self, obs_tensors: List[Tensor], action_tensors: List[Tensor]) -> Tensor:
            """Berechnet die Q-Werte der gewählten Aktionen für jeden Agenten mit den Policy-Netzen."""
            chosen_qvals = []
            for i in range(self.num_agents):
                 q_all = self.agent_networks[i](obs_tensors[i]) # Q-Werte für alle Aktionen
                 q_chosen = q_all.gather(1, action_tensors[i])  # Q-Wert für die tatsächlich gewählte Aktion
                 chosen_qvals.append(q_chosen)
            # Stack zu (batch_size, num_agents)
            return torch.cat(chosen_qvals, dim=1)

        @torch.no_grad() # Keine Gradienten für Target-Berechnung
        def _get_target_max_qvals(self, next_obs_tensors: List[Tensor]) -> Tensor:
            """Berechnet die maximalen Q-Werte im Folgezustand für jeden Agenten mit den Target-Netzen."""
            target_max_q = []
            for i in range(self.num_agents):
                 next_q_target_net = self.target_networks[i](next_obs_tensors[i])
                 # Wähle die Aktion mit dem höchsten Q-Wert aus dem Target-Netzwerk
                 q_max = next_q_target_net.max(dim=1, keepdim=True)[0]
                 target_max_q.append(q_max)
            # Stack zu (batch_size, num_agents)
            return torch.cat(target_max_q, dim=1)


        def train(self) -> Optional[float]:
            """
            Führt einen kompletten Trainingsschritt durch.

            Returns:
                Der berechnete Verlust des Mixers für diesen Schritt, oder None wenn kein Training stattfand.
            """
            batch_data = self._sample_batch()
            if batch_data is None:
                return None

            # Tensoren vorbereiten
            (state_batch, obs_batch, action_batch, reward_batch,
             next_state_batch, next_obs_batch, done_batch) = self._prepare_batch_tensors(batch_data)

            # --- Training ---
            self.train_mode() # Sicherstellen, dass Netze im Trainingsmodus sind

            # 1. Berechne Q-Werte für die im Batch ausgeführten Aktionen (mit Policy-Netzen)
            chosen_agent_qvals = self._get_chosen_action_qvals(obs_batch, action_batch)

            # 2. Berechne maximale Q-Werte für die Folgezustände (mit Target-Netzen)
            target_max_agent_qvals = self._get_target_max_qvals(next_obs_batch)

            # 3. Berechne Q_tot für aktuelle und nächste Zustände mit den Mixern
            # Q_tot(s, u) mit Policy-Mixer
            current_q_total = self.mixer(chosen_agent_qvals, state_batch)

            # Q_tot(s', u') mit Target-Mixer (detach, da dies das Ziel ist)
            target_q_total = self.target_mixer(target_max_agent_qvals, next_state_batch).detach()

            # 4. Berechne das Bellman-Target: r + gamma * Q_tot'(s', u') * (1 - done)
            targets = reward_batch + self.gamma * target_q_total * (1.0 - done_batch)

            # 5. Berechne den TD-Error (Verlust) für den Mixer
            loss = self.loss_fn(current_q_total, targets)
            loss_value = loss.item()
            self.loss_history.append(loss_value)

            # 6. Backpropagation: Aktualisiere Mixer und Agenten-Netzwerke
            #    Der Gradient fließt vom Mixer-Loss zurück durch die Agenten-Netzwerke.
            #    Alle Optimierer zurücksetzen
            self.mixer_optimizer.zero_grad()
            for opt in self.agent_optimizers:
                opt.zero_grad()

            # Gradienten berechnen (für Mixer und implizit für Agenten-Netze)
            loss.backward()

            # Gradient Clipping (optional, aber empfohlen)
            torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10.0) # Clip Mixer
            for net in self.agent_networks:
                 torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0) # Clip Agenten

            # Optimierungsschritt
            self.mixer_optimizer.step()
            for opt in self.agent_optimizers:
                opt.step()

            # 7. Target-Netzwerke periodisch aktualisieren
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self._update_target_networks()
                self._update_target_mixer()

            logger.debug(f"QMIX Trainingsschritt: Loss={loss_value:.4f}")
            return loss_value

        def eval_mode(self):
            """Schaltet alle Netzwerke in den Evaluationsmodus."""
            for net in self.agent_networks: net.eval()
            self.mixer.eval()
            # Target-Netze sollten immer im Eval-Modus sein, aber zur Sicherheit:
            for net in self.target_networks: net.eval()
            self.target_mixer.eval()

        def train_mode(self):
            """Schaltet Policy-Netzwerke und Mixer in den Trainingsmodus."""
            for net in self.agent_networks: net.train()
            self.mixer.train()
            # Target-Netze bleiben im Eval-Modus

        def save_model(self, directory: str) -> None:
            """Speichert den Zustand des QMIXAgent (Netzwerke, Optimierer)."""
            if not TORCH_AVAILABLE: return
            try:
                os.makedirs(directory, exist_ok=True)
                # Speichere Mixer
                torch.save(self.mixer.state_dict(), os.path.join(directory, "mixer.pth"))
                torch.save(self.mixer_optimizer.state_dict(), os.path.join(directory, "mixer_optimizer.pth"))
                # Speichere Target Mixer (optional, kann auch synchronisiert werden)
                torch.save(self.target_mixer.state_dict(), os.path.join(directory, "target_mixer.pth"))

                # Speichere Agenten-Netzwerke
                for i, net in enumerate(self.agent_networks):
                    torch.save(net.state_dict(), os.path.join(directory, f"agent_{i}_network.pth"))
                    torch.save(self.agent_optimizers[i].state_dict(), os.path.join(directory, f"agent_{i}_optimizer.pth"))
                    torch.save(self.target_networks[i].state_dict(), os.path.join(directory, f"target_agent_{i}_network.pth"))

                logger.info(f"QMIX-Modell gespeichert im Verzeichnis: {directory}")
            except Exception as e:
                 logger.error(f"Fehler beim Speichern des QMIX-Modells: {e}", exc_info=True)

        def load_model(self, directory: str) -> None:
            """Lädt den Zustand des QMIXAgent."""
            if not TORCH_AVAILABLE: return
            logger.info(f"Lade QMIX-Modell aus Verzeichnis: {directory}")
            try:
                # Lade Mixer
                self.mixer.load_state_dict(torch.load(os.path.join(directory, "mixer.pth"), map_location=self.device))
                self.mixer_optimizer.load_state_dict(torch.load(os.path.join(directory, "mixer_optimizer.pth"), map_location=self.device))
                self.target_mixer.load_state_dict(torch.load(os.path.join(directory, "target_mixer.pth"), map_location=self.device))

                # Lade Agenten-Netzwerke
                for i, net in enumerate(self.agent_networks):
                    net.load_state_dict(torch.load(os.path.join(directory, f"agent_{i}_network.pth"), map_location=self.device))
                    self.agent_optimizers[i].load_state_dict(torch.load(os.path.join(directory, f"agent_{i}_optimizer.pth"), map_location=self.device))
                    self.target_networks[i].load_state_dict(torch.load(os.path.join(directory, f"target_agent_{i}_network.pth"), map_location=self.device))

                # Setze Netzwerke in korrekten Modus
                self.train_mode() # Standardmäßig zum Weitertrainieren bereit
                for net in self.target_networks: net.eval()
                self.target_mixer.eval()

                logger.info("QMIX-Modell erfolgreich geladen.")
            except FileNotFoundError:
                 logger.error(f"Fehler beim Laden des QMIX-Modells: Mindestens eine Datei in '{directory}' nicht gefunden.")
            except Exception as e:
                 logger.error(f"Fehler beim Laden des QMIX-Modells: {e}", exc_info=True)

else:
    # Definiere Platzhalter-Klassen, wenn PyTorch nicht verfügbar ist
    # Damit können andere Module importieren, ohne sofort zu crashen
    class QMIXNetwork:
        def __init__(self, *args, **kwargs):
            logger.error(_TORCH_ERROR_MSG)
            raise ImportError(_TORCH_ERROR_MSG)

    class QMIXAgent:
         def __init__(self, *args, **kwargs):
            logger.error(_TORCH_ERROR_MSG)
            raise ImportError(_TORCH_ERROR_MSG)