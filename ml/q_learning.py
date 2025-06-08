# Impaler/ml/q_learning.py
"""
Implementierungen für tabellarisches Q-Learning und Deep Q-Learning (DQN).

Angepasst für Szenarien ohne klassische Markt-/Profit-Logik, wie z.B.
in einer Planwirtschaftssimulation. Reward-Signale können auf
Planerfüllung, Ressourceneffizienz, Fairness oder Emissionsreduktion basieren.
"""

import random
import numpy as np
import logging
from collections import deque
from typing import Dict, Any, List, Tuple, Optional, Deque, Union

# Logger für dieses Modul
logger = logging.getLogger(__name__)

# Optional PyTorch für Deep Q-Learning:
_TORCH_ERROR_MSG = ("PyTorch nicht gefunden. DeepQLearningAgent ist nicht verfügbar. "
                    "Installieren Sie PyTorch (pytorch.org) für DQN-Funktionalität.")
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    Tensor = torch.Tensor # Typ-Alias für Lesbarkeit
    Module = nn.Module
    Optimizer = optim.Optimizer
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(_TORCH_ERROR_MSG)
    # Platzhalter-Typen für Type Hints, wenn Torch nicht da ist
    Tensor = Any
    Module = Any
    Optimizer = Any


# --- Tabellarisches Q-Learning ---

class QLearningAgent:
    """
    Einfache tabellarische Q-Learning-Implementierung für diskrete Zustände und Aktionen.

    Geeignet für Probleme mit einem überschaubaren, diskreten Zustandsraum.
    In einer Planwirtschaft könnte dies z.B. die Auswahl diskreter Investitionslevel
    oder Produktionsstrategien sein.

    Attributes:
        state_dim (int): Dimension des Zustandsraums (wird zur Diskretisierung verwendet,
                         falls state_bounds gesetzt sind).
        action_dim (int): Anzahl diskreter Aktionen.
        q_table (Dict[Tuple, np.ndarray]): Q-Werte-Tabelle. Der Schlüssel ist der
                                           diskretisierte Zustand (als Tuple).
        learning_rate (float): Lernrate (alpha) für Q-Wert-Updates.
        discount_factor (float): Diskontfaktor (gamma) für zukünftige Belohnungen.
        epsilon (float): Aktuelle Explorationsrate (Wahrscheinlichkeit für Zufallsaktion).
        epsilon_decay (float): Faktor zur Reduzierung der Explorationsrate pro Lernschritt.
        min_epsilon (float): Untergrenze für die Explorationsrate.
        state_bounds (Optional[List[Tuple[float, float]]]): Grenzen für jede Dimension
                                                             des Zustandsraums zur Diskretisierung.
                                                             Wenn None, wird angenommen, dass
                                                             Zustände bereits diskret (hashbar) sind.
        bins_per_dimension (int): Anzahl der Bins pro Dimension für die Diskretisierung.
        action_history (Deque[int]): Historie der gewählten Aktionen.
        reward_history (Deque[float]): Historie der erhaltenen Belohnungen.
    """
    state_dim: int
    action_dim: int
    q_table: Dict[Tuple, np.ndarray] # Zustand (Tuple) -> Q-Werte (Array)
    learning_rate: float
    discount_factor: float
    epsilon: float
    epsilon_decay: float
    min_epsilon: float
    state_bounds: Optional[List[Tuple[float, float]]]
    bins_per_dimension: int
    action_history: Deque[int]
    reward_history: Deque[float]

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 initial_epsilon: float = 1.0, # Start mit hoher Exploration
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.05,  # Angepasst auf 5%
                 state_bounds: Optional[List[Tuple[float, float]]] = None,
                 bins_per_dimension: int = 10,
                 history_length: int = 1000):
        """
        Initialisiert den tabellarischen Q-Learning Agenten.

        Args:
            state_dim: Dimension des (ggf. kontinuierlichen) Zustandsvektors.
            action_dim: Anzahl der möglichen diskreten Aktionen.
            learning_rate: Lernrate alpha (0 bis 1).
            discount_factor: Diskontfaktor gamma (0 bis 1).
            initial_epsilon: Startwert für die Explorationsrate (0 bis 1).
            epsilon_decay: Faktor zur Reduzierung von epsilon (typ. < 1).
            min_epsilon: Minimaler Wert für epsilon.
            state_bounds: Liste von Tupeln (min, max) für jede Zustandsdimension,
                          wird zur Diskretisierung benötigt. Wenn None, wird
                          angenommen, dass die übergebenen Zustände bereits
                          diskret und hashbar (z.B. Tupel) sind.
            bins_per_dimension: Anzahl der Bins für die Diskretisierung.
            history_length: Maximale Länge der gespeicherten Historien.
        """
        if state_dim <= 0 or action_dim <= 0:
             raise ValueError("Zustands- und Aktionsdimension müssen positiv sein.")
        if not (0.0 <= learning_rate <= 1.0):
             raise ValueError("Lernrate muss zwischen 0 und 1 liegen.")
        if not (0.0 <= discount_factor <= 1.0):
             raise ValueError("Diskontfaktor muss zwischen 0 und 1 liegen.")
        if not (0.0 <= initial_epsilon <= 1.0) or not (0.0 <= min_epsilon <= initial_epsilon):
             raise ValueError("Epsilon-Werte müssen gültige Wahrscheinlichkeiten sein (min <= initial <= 1).")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.state_bounds = state_bounds
        self.bins_per_dimension = max(1, bins_per_dimension) # Mindestens 1 Bin

        # Deques für effizientes Anhängen/Entfernen
        self.action_history = deque(maxlen=history_length)
        # self.state_history = deque(maxlen=history_length) # Zustandshistorie oft Speicherintensiv
        self.reward_history = deque(maxlen=history_length)

        logger.info(f"Tabular QLearningAgent initialisiert: state_dim={state_dim}, action_dim={action_dim}, lr={learning_rate}, gamma={discount_factor}, epsilon_init={initial_epsilon}")

    def _discretize_state(self, state: Union[np.ndarray, Tuple]) -> Tuple:
        """
        Wandelt einen Zustand in einen diskreten, hashbaren Zustand (Tuple) um.

        Wenn state_bounds gesetzt sind, wird ein kontinuierlicher np.ndarray erwartet
        und in Bins eingeteilt. Ansonsten wird angenommen, der Zustand ist bereits
        hashbar (z.B. ein Tuple) und wird direkt zurückgegeben.

        Args:
            state: Der Zustand (als NumPy Array oder bereits hashbares Objekt).

        Returns:
            Ein Tuple, das den diskretisierten Zustand repräsentiert.

        Raises:
            ValueError: Wenn state_bounds benötigt werden (state ist ndarray) aber nicht gesetzt sind,
                        oder wenn die Dimension des Zustands nicht zu state_bounds passt.
            TypeError: Wenn der Zustand weder ndarray noch hashbar ist.
        """
        if self.state_bounds:
            if not isinstance(state, np.ndarray):
                raise TypeError(f"Zur Diskretisierung wird ein NumPy Array erwartet, aber {type(state)} erhalten.")
            if len(state) != len(self.state_bounds):
                raise ValueError(f"Dimension des Zustands ({len(state)}) passt nicht zur Anzahl der state_bounds ({len(self.state_bounds)}).")

            discrete_state = []
            for i, s_val in enumerate(state):
                low, high = self.state_bounds[i]
                if high <= low: # Ungültige Bounds
                     bin_index = 0
                else:
                    # Wert auf Grenzen clippen
                    s_clipped = np.clip(s_val, low, high)
                    # Bin-Index berechnen
                    # Robust gegen high == low
                    normalized_val = (s_clipped - low) / max(high - low, 1e-9) # Skaliert auf 0-1
                    bin_index = int(normalized_val * self.bins_per_dimension)
                    # Sicherstellen, dass der Index im Bereich [0, bins-1] liegt
                    bin_index = min(bin_index, self.bins_per_dimension - 1)
                discrete_state.append(bin_index)
            return tuple(discrete_state)
        else:
            # Annahme: Zustand ist bereits hashbar (z.B. Tuple)
            if not isinstance(state, tuple):
                 logger.warning(f"Zustand ist Typ {type(state)}, kein Tuple, und keine state_bounds gesetzt. Versuche Hashing, kann fehlschlagen.")
                 # Versuche es trotzdem, Python's dict kann viele Typen hashen
            try:
                 # Teste Hashing
                 _ = {state: 1}
                 return state # Gib unverändert zurück
            except TypeError:
                 raise TypeError(f"Zustand vom Typ {type(state)} ist nicht hashbar und state_bounds sind nicht gesetzt.")


    def get_q_value(self, state: Union[np.ndarray, Tuple], action: int) -> float:
        """Gibt den Q-Wert für ein Zustands-Aktions-Paar zurück."""
        if not (0 <= action < self.action_dim):
             raise ValueError(f"Ungültige Aktion {action}. Muss zwischen 0 und {self.action_dim - 1} liegen.")

        disc_state = self._discretize_state(state)
        # Initialisiere Q-Werte für neuen Zustand mit Nullen
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(self.action_dim)
        return float(self.q_table[disc_state][action])

    def update_q_value(self,
                       state: Union[np.ndarray, Tuple],
                       action: int,
                       reward: float,
                       next_state: Union[np.ndarray, Tuple],
                       done: bool) -> None:
        """
        Führt das Q-Learning Update (Bellman-Gleichung) durch.
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a'(Q(s', a')) - Q(s, a)]
        """
        if not (0 <= action < self.action_dim):
             logger.error(f"Ungültige Aktion {action} beim Q-Update. Update übersprungen.")
             return

        disc_state = self._discretize_state(state)
        disc_next_state = self._discretize_state(next_state)

        # Initialisiere Q-Werte für (neue) Zustände
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(self.action_dim)
        if disc_next_state not in self.q_table:
            self.q_table[disc_next_state] = np.zeros(self.action_dim)

        # Finde besten Q-Wert im Folgezustand
        best_next_q = np.max(self.q_table[disc_next_state])

        # TD-Ziel berechnen (Temporal Difference Target)
        # Wenn done=True, gibt es keinen Folgezustandswert
        td_target = reward + (0.0 if done else self.discount_factor * best_next_q)

        # TD-Fehler berechnen
        current_q = self.q_table[disc_state][action]
        td_error = td_target - current_q

        # Q-Wert aktualisieren
        self.q_table[disc_state][action] += self.learning_rate * td_error

    def choose_action(self, state: Union[np.ndarray, Tuple], explore: bool = True) -> int:
        """
        Wählt eine Aktion mittels Epsilon-Greedy-Strategie.

        Args:
            state: Der aktuelle Zustand (diskret oder kontinuierlich, wenn bounds gesetzt).
            explore: Ob Exploration (epsilon > 0) angewendet werden soll.

        Returns:
            Der Index der gewählten Aktion (int).
        """
        disc_state = self._discretize_state(state)

        # Initialisiere Q-Werte für neuen Zustand
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(self.action_dim)

        # Epsilon-Greedy
        if explore and random.random() < self.epsilon:
            # Exploration: Zufällige Aktion wählen
            action = random.randrange(self.action_dim)
            logger.debug(f"QLearningAgent wählt zufällige Aktion: {action} (Epsilon: {self.epsilon:.3f})")
        else:
            # Exploitation: Beste bekannte Aktion wählen
            q_values = self.q_table[disc_state]
            # Wähle Aktion mit maximalem Q-Wert. Bei Gleichstand zufällig auswählen.
            max_q = np.max(q_values)
            best_actions = np.where(q_values >= max_q - EPSILON)[0] # Finde alle Aktionen nahe am Maximum
            action = random.choice(best_actions)
            logger.debug(f"QLearningAgent wählt beste Aktion: {action} (Q-Werte: {q_values})")

        return int(action)

    def decay_exploration(self) -> None:
        """Reduziert die Explorationsrate Epsilon."""
        old_epsilon = self.epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        if self.epsilon < old_epsilon:
             logger.debug(f"Epsilon reduziert auf {self.epsilon:.4f}")

    def learn(self,
              state: Union[np.ndarray, Tuple],
              action: int,
              reward: float,
              next_state: Union[np.ndarray, Tuple],
              done: bool) -> None:
        """
        Führt einen kompletten Lernschritt durch: Q-Update, Historie, Epsilon-Decay.

        Args:
            state: Zustand vor der Aktion.
            action: Ausgeführte Aktion.
            reward: Erhaltene Belohnung.
            next_state: Zustand nach der Aktion.
            done: True, wenn der Zustand ein Endzustand ist.
        """
        # 1. Q-Wert aktualisieren
        self.update_q_value(state, action, reward, next_state, done)

        # 2. Historie speichern
        self.action_history.append(action)
        self.reward_history.append(reward)
        # self.state_history.append(state) # Zustandshistorie oft sehr groß

        # 3. Exploration reduzieren
        self.decay_exploration()

    def get_recent_average_reward(self, window: int = 100) -> float:
        """Berechnet den Durchschnitts-Reward der letzten 'window' Schritte."""
        if not self.reward_history:
            return 0.0
        # Nehme min(window, Länge der Historie)
        actual_window = min(window, len(self.reward_history))
        # Berechne Durchschnitt der letzten Einträge
        recent_rewards = list(self.reward_history)[-actual_window:]
        return sum(recent_rewards) / actual_window if actual_window > 0 else 0.0


# --- Deep Q-Learning (DQN) ---

class DeepQLearningAgent:
    """
    Ein Deep Q-Network (DQN) Agent auf PyTorch-Basis.

    Verwendet neuronale Netze zur Approximation der Q-Funktion, geeignet für
    kontinuierliche oder hochdimensionale Zustandsräume. Nutzt Techniken wie
    Experience Replay und Target Networks für stabileres Lernen.

    Reward-Funktion sollte im Anwendungskontext (z.B. Planwirtschaft) definiert werden.

    Attributes:
        state_dim (int): Dimension des Zustandsraums.
        action_dim (int): Anzahl diskreter Aktionen.
        hidden_dim (int): Größe der versteckten Layer im Netzwerk.
        learning_rate (float): Lernrate für den Optimierer.
        discount_factor (float): Diskontfaktor gamma.
        epsilon (float): Aktuelle Explorationsrate.
        epsilon_decay (float): Faktor zur Reduzierung von epsilon.
        min_epsilon (float): Untergrenze für epsilon.
        batch_size (int): Größe der Mini-Batches für das Training.
        memory_size (int): Maximale Größe des Replay-Speichers.
        target_update_freq (int): Frequenz (in Lernschritten), mit der das Target-Netzwerk aktualisiert wird.
        update_counter (int): Zähler für Lernschritte seit dem letzten Target-Update.
        policy_net (Module): Das trainierte Q-Netzwerk.
        target_net (Module): Das Ziel-Netzwerk (verzögerte Kopie des Policy-Netzwerks).
        optimizer (Optimizer): PyTorch-Optimierer (z.B. Adam).
        loss_fn (Module): Verlustfunktion (z.B. MSELoss).
        memory (Deque): Replay-Speicher für Transitionen (s, a, r, s', done).
        loss_history (Deque[float]): Historie der Verlustwerte.
        reward_history (Deque[float]): Historie der erhaltenen Belohnungen.
        epsilon_history (Deque[float]): Historie der Epsilon-Werte.
    """
    state_dim: int
    action_dim: int
    hidden_dim: int
    learning_rate: float
    discount_factor: float
    epsilon: float
    epsilon_decay: float
    min_epsilon: float
    batch_size: int
    target_update_freq: int
    update_counter: int
    policy_net: Module
    target_net: Module
    optimizer: Optimizer
    loss_fn: Module
    memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]]
    loss_history: Deque[float]
    reward_history: Deque[float]
    epsilon_history: Deque[float]


    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128, # Erhöht für mehr Kapazität
                 learning_rate: float = 0.0005, # Oft kleiner als bei tabular
                 discount_factor: float = 0.99,
                 initial_epsilon: float = 1.0,
                 epsilon_decay: float = 0.999, # Langsameres Decay
                 min_epsilon: float = 0.05,
                 batch_size: int = 128, # Größerer Batch
                 memory_size: int = 20000, # Größerer Speicher
                 target_update_freq: int = 500, # Selteneres Update
                 history_length: int = 1000):
        """
        Initialisiert den DQN Agenten.

        Args:
            state_dim: Dimension des Zustandsvektors.
            action_dim: Anzahl der diskreten Aktionen.
            hidden_dim: Anzahl Neuronen in den versteckten Layern.
            learning_rate: Lernrate für den Adam-Optimierer.
            discount_factor: Diskontfaktor gamma.
            initial_epsilon: Startwert für Epsilon.
            epsilon_decay: Decay-Faktor für Epsilon.
            min_epsilon: Minimalwert für Epsilon.
            batch_size: Größe der Trainings-Batches.
            memory_size: Maximale Größe des Replay Buffers.
            target_update_freq: Frequenz der Target-Netzwerk-Updates in Lernschritten.
            history_length: Länge der gespeicherten Historien (Loss, Reward, Epsilon).

        Raises:
            ImportError: Wenn PyTorch nicht installiert ist.
        """
        if not TORCH_AVAILABLE:
            # Fehler werfen statt nur zu loggen, da die Klasse ohne Torch nicht nutzbar ist
            raise ImportError(_TORCH_ERROR_MSG)

        # Validierung der Parameter
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
             raise ValueError("Dimensionen (state, action, hidden) müssen positiv sein.")
        if not (0.0 < learning_rate <= 1.0):
             raise ValueError("Lernrate muss im Bereich (0, 1] liegen.")
        # ... (weitere Validierungen für gamma, epsilon, batch_size, etc.) ...

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Geräteerkennung (CPU oder GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DQN Agent verwendet Gerät: {self.device}")

        # Netzwerke erstellen und auf Gerät verschieben
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        # Initial Gewichte synchronisieren
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target-Netzwerk ist nur zur Evaluation

        # Optimierer und Verlustfunktion
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss() # Mean Squared Error für Q-Wert Differenz

        # Replay Memory
        self.memory = deque(maxlen=memory_size)

        # Historien
        self.loss_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        self.epsilon_history = deque(maxlen=history_length)

        logger.info(f"DeepQLearningAgent initialisiert (State: {state_dim}, Action: {action_dim}, Hidden: {hidden_dim}, LR: {learning_rate}, Gamma: {discount_factor}, Batch: {batch_size})")

    def _build_network(self) -> Module:
        """
        Erstellt das neuronale Netzwerk für die Q-Funktion (MLP).

        Kann für spezifische Probleme angepasst werden (z.B. CNNs für Bilddaten).

        Returns:
            Ein PyTorch nn.Module Objekt.
        """
        # Einfaches Multi-Layer Perceptron (MLP)
        # Input -> Hidden1 -> ReLU -> Hidden2 -> ReLU -> Output (Q-Werte pro Aktion)
        model = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        return model

    def store_transition(self,
                         state: np.ndarray,
                         action: int,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool) -> None:
        """
        Speichert eine (State, Action, Reward, Next_State, Done)-Transition
        im Replay Memory.

        Args:
            state: Zustand vor der Aktion.
            action: Ausgeführte Aktion (Index).
            reward: Erhaltene Belohnung.
            next_state: Zustand nach der Aktion.
            done: True, wenn der Folgezustand ein Endzustand ist.
        """
        # Stelle sicher, dass Inputs korrekte Typen haben (für spätere Tensor-Konvertierung)
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.append((state, action, float(reward), next_state, bool(done)))
        # Direkte Speicherung des Rewards für einfache Analyse
        self.reward_history.append(float(reward))


    def choose_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Wählt eine Aktion: Epsilon-Greedy (Exploration vs. Exploitation).

        Args:
            state: Der aktuelle Zustand als NumPy Array.
            explore: Ob Exploration (zufällige Aktion) erlaubt ist.

        Returns:
            Der Index der gewählten Aktion.
        """
        if explore and random.random() < self.epsilon:
            # Exploration
            action = random.randrange(self.action_dim)
            logger.debug(f"DQN Agent wählt zufällige Aktion: {action} (Epsilon: {self.epsilon:.3f})")
        else:
            # Exploitation: Wähle Aktion mit höchstem Q-Wert vom Policy Network
            try:
                state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                # Wichtig: Netzwerk in Eval-Modus schalten, kein Gradient berechnen
                self.policy_net.eval()
                with torch.no_grad():
                    q_values = self.policy_net(state_t)
                self.policy_net.train() # Zurück in Trainingsmodus

                action = q_values.argmax(dim=1).item() # Aktion mit höchstem Q-Wert
                logger.debug(f"DQN Agent wählt beste Aktion: {action} (Q-Werte: {q_values.cpu().numpy()})")
            except Exception as e:
                 logger.error(f"Fehler bei der Aktionsauswahl im DQN: {e}. Wähle zufällige Aktion.", exc_info=True)
                 action = random.randrange(self.action_dim)

        return int(action)


    def learn(self) -> Optional[float]:
        """
        Führt einen Lernschritt durch: Samplen aus dem Memory, Q-Update, Target-Net Update.

        Returns:
            Der berechnete Verlust (Loss) für diesen Batch, oder None wenn nicht genug Daten.
        """
        # Nur lernen, wenn genug Erfahrungen im Speicher sind
        if len(self.memory) < self.batch_size:
            logger.debug(f"Nicht genug Samples im Memory ({len(self.memory)}/{self.batch_size}) zum Lernen.")
            return None

        # Mini-Batch aus dem Speicher ziehen
        batch = random.sample(self.memory, self.batch_size)
        # Entpacke den Batch in separate Listen/Arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        # Konvertiere zu PyTorch Tensoren und sende auf das richtige Gerät (CPU/GPU)
        state_batch = torch.from_numpy(np.stack(states)).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device) # Shape: [batch_size, 1]
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # Shape: [batch_size, 1]
        next_state_batch = torch.from_numpy(np.stack(next_states)).to(self.device)
        done_batch = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device) # Shape: [batch_size, 1] (0.0 oder 1.0)

        # --- Q-Learning Update ---
        # 1. Berechne Q(s, a) für die ausgeführten Aktionen mit dem Policy-Netzwerk
        #    q_values hat Shape [batch_size, action_dim]
        q_values = self.policy_net(state_batch)
        #    Nutze gather(), um die Q-Werte der tatsächlich gewählten Aktionen zu selektieren
        #    current_q_values hat Shape [batch_size, 1]
        current_q_values = q_values.gather(1, action_batch)

        # 2. Berechne max Q'(s', a') für die Folgezustände mit dem Target-Netzwerk
        #    (Double DQN Verbesserung: Aktionen vom Policy-Net, Werte vom Target-Net)
        with torch.no_grad(): # Keine Gradienten für Target-Berechnung nötig
             # Finde die besten Aktionen im Folgezustand gemäß Policy-Netzwerk
             next_policy_q = self.policy_net(next_state_batch)
             best_next_actions = next_policy_q.argmax(dim=1, keepdim=True) # Shape: [batch_size, 1]

             # Hole die Q-Werte für diese Aktionen vom Target-Netzwerk
             next_target_q = self.target_net(next_state_batch)
             max_next_q_values = next_target_q.gather(1, best_next_actions).detach() # Shape: [batch_size, 1]

        # 3. Berechne das TD-Target: R + gamma * max Q'(s', a') * (1 - done)
        #    (1 - done_batch) stellt sicher, dass der zukünftige Wert 0 ist, wenn es ein Endzustand war
        td_target = reward_batch + (1.0 - done_batch) * self.discount_factor * max_next_q_values

        # 4. Berechne den Verlust (Loss) zwischen aktuellen Q-Werten und TD-Target
        loss = self.loss_fn(current_q_values, td_target)
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        # 5. Optimiere das Policy-Netzwerk (Backpropagation)
        self.optimizer.zero_grad() # Gradienten zurücksetzen
        loss.backward() # Gradienten berechnen
        # Optional: Gradient Clipping zur Stabilisierung
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0) # Beispiel: Clip bei Wert 1.0
        self.optimizer.step() # Gewichte aktualisieren

        # 6. Epsilon-Decay
        self.decay_exploration()
        self.epsilon_history.append(self.epsilon)

        # 7. Target-Netzwerk periodisch aktualisieren
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()

        logger.debug(f"DQN Lernschritt: Batch Size={self.batch_size}, Loss={loss_value:.4f}")
        return loss_value

    def _update_target_network(self) -> None:
        """Kopiert die Gewichte vom Policy-Netzwerk zum Target-Netzwerk."""
        logger.info(f"Aktualisiere Target-Netzwerk nach {self.update_counter} Lernschritten.")
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_recent_average_reward(self, window: int = 100) -> float:
        """Berechnet den Durchschnitts-Reward der letzten 'window' gespeicherten Transitions."""
        if not self.reward_history: return 0.0
        actual_window = min(window, len(self.reward_history))
        recent_rewards = list(self.reward_history)[-actual_window:]
        return sum(recent_rewards) / actual_window if actual_window > 0 else 0.0

    def get_average_loss(self, window: int = 100) -> float:
        """Berechnet den durchschnittlichen Loss der letzten 'window' Lernschritte."""
        if not self.loss_history: return 0.0
        actual_window = min(window, len(self.loss_history))
        recent_losses = list(self.loss_history)[-actual_window:]
        return sum(recent_losses) / actual_window if actual_window > 0 else 0.0

    def save_model(self, policy_path: str, target_path: Optional[str] = None) -> None:
        """Speichert die Gewichte des Policy- (und optional Target-) Netzwerks."""
        if not TORCH_AVAILABLE:
            logger.error("Kann DQN-Modell nicht speichern, da PyTorch nicht verfügbar ist.")
            return
        try:
            torch.save(self.policy_net.state_dict(), policy_path)
            logger.info(f"Policy-Netzwerk gespeichert unter: {policy_path}")
            if target_path:
                 torch.save(self.target_net.state_dict(), target_path)
                 logger.info(f"Target-Netzwerk gespeichert unter: {target_path}")
        except Exception as e:
             logger.error(f"Fehler beim Speichern des DQN-Modells: {e}", exc_info=True)


    def load_model(self, policy_path: str, target_path: Optional[str] = None) -> None:
        """Lädt die Gewichte für das Policy- (und optional Target-) Netzwerk."""
        if not TORCH_AVAILABLE:
            logger.error("Kann DQN-Modell nicht laden, da PyTorch nicht verfügbar ist.")
            return
        try:
            self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))
            self.policy_net.eval() # In Eval-Modus setzen nach Laden
            logger.info(f"Policy-Netzwerk geladen von: {policy_path}")
            # Target-Netzwerk ebenfalls laden oder synchronisieren
            if target_path:
                 self.target_net.load_state_dict(torch.load(target_path, map_location=self.device))
                 self.target_net.eval()
                 logger.info(f"Target-Netzwerk geladen von: {target_path}")
            else:
                 # Wenn kein separates Target-File, synchronisiere mit Policy-Net
                 self._update_target_network()
                 logger.info("Target-Netzwerk mit geladenem Policy-Netzwerk synchronisiert.")

            # Setze Policy-Net wieder in Trainingsmodus, falls weiter trainiert werden soll
            self.policy_net.train()

        except FileNotFoundError:
             logger.error(f"Fehler beim Laden des DQN-Modells: Datei(en) nicht gefunden ({policy_path}{' / ' + target_path if target_path else ''}).")
        except Exception as e:
             logger.error(f"Fehler beim Laden des DQN-Modells: {e}", exc_info=True)