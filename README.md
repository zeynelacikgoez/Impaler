# Impaler

**Impaler** ist eine hochentwickelte, agentenbasierte Simulationsplattform zur Generierung und Evaluierung neuartiger Planungs- und Wirtschaftsmodelle. Der Fokus liegt auf der Untersuchung komplexer, nicht-marktbasierter Koordinationsmechanismen unter Verwendung von Konzepten des **Viable System Model (VSM)**, Multi-Agent Reinforcement Learning (MARL) und evolutionären Algorithmen.

## Kernkonzepte

Impaler kombiniert mehrere fortschrittliche Paradigmen, um ein tiefes Verständnis für komplexe Wirtschaftssysteme zu ermöglichen:

-   **Agentenbasierte Simulation:** Das Herzstück des Systems sind autonome Agenten (`Producer`, `Consumer`, `Resource`), die mit dynamischen Bedürfnissen, Lernfähigkeit und komplexen Produktionsprozessen ausgestattet sind.

-   **Viable System Model (VSM):** Die Simulation ist nach den Prinzipien des VSM strukturiert, einem kybernetischen Managementmodell. Verschiedene Systemebenen (S2, S3, S4, S5) übernehmen spezifische Aufgaben von der operativen Koordination bis zur strategischen Politikgestaltung.

-   **Dynamische Krisensimulation:** Der `CrisisManager` kann geplante oder zufällige Krisen wie Ressourcenknappheit oder Naturkatastrophen simulieren und deren Auswirkungen auf die Systemstabilität und Anpassungsfähigkeit testen.

-   **Evolutionäre Modellsuche:** Impaler kann genetische Algorithmen verwenden, um den riesigen Parameterraum von Wirtschaftsmodellen zu durchsuchen. Durch parallele Ausführung von Simulationen mit unterschiedlichen "Genomen" (Konfigurationen) und einer Fitnessfunktion (`welfare`, `sustainability`, `stability`) werden robustere und leistungsfähigere Modellkonfigurationen evolutiv gefunden.

-   **Multi-Agent Reinforcement Learning (MARL):** Agenten, insbesondere `ProducerAgent`, können im RL-Modus betrieben werden, um optimale Verhaltensstrategien (z.B. für Investitionen) zu erlernen. Das Framework nutzt **Ray RLlib** und stellt eine **PettingZoo**-Umgebung für standardisierte Trainingsprozesse bereit.

-   **Flexible Pydantic-Konfiguration:** Die gesamte Simulation wird durch eine stark typisierte und validierte Konfigurationsstruktur (`core/config.py`) gesteuert, die eine einfache Anpassung und Reproduzierbarkeit von Experimenten ermöglicht.

## Architektur-Überblick

Das Projekt ist modular aufgebaut:

-   **/core:** Enthält das zentrale `EconomicModel`, den `StageManager`, den `CrisisManager`, den `DataCollector` und die Pydantic-`SimulationConfig`.
-   **/agents:** Definiert die Verhaltenslogik der Wirtschaftsakteure (`ProducerAgent`, `ConsumerAgent`, `ResourceAgent`).
-   **/vsm:** Implementiert die verschiedenen Ebenen des Viable System Model (`System2Coordinator`, `System3Manager`, `System4Planner`).
-   **/governance:** Beinhaltet System 5 (`GovernmentAgent`), das für die übergeordnete Politik und Zielsetzung zuständig ist.
-   **/evolution:** Stellt Werkzeuge für die evolutionäre Suche bereit, einschließlich Fitnessfunktion und Parameterraum-Definition.
-   **/marl:** Enthält den Code für das Training von Multi-Agent Reinforcement Learning Modellen, inklusive der PettingZoo-Umgebung (`envs`) und Trainingsskripten.
-   **/utils:** Bietet mathematische und ökonomische Hilfsfunktionen.

## Installation

### Voraussetzungen

-   Python 3.7 oder höher
-   Einige Funktionalitäten (z.B. Optimierung) profitieren von der Installation optionaler Bibliotheken wie SciPy.

### Schritte zur Installation

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/deinusername/impaler.git
    cd Impaler-main
    ```

2.  **Virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    ```

3.  **Abhängigkeiten installieren:**
    Alle notwendigen Pakete, einschließlich `ray[rllib]`, `pettingzoo`, `pydantic` und `numpy`, werden über die `requirements.txt`-Datei installiert.
    ```bash
    pip install -r requirements.txt
    ```

## Anwendung

Impaler kann auf verschiedene Weisen ausgeführt werden, je nach Untersuchungsziel.

### 1. Einzelne Simulation ausführen

Dies ist nützlich, um ein spezifisches Wirtschaftsmodell mit einer festen Konfiguration zu testen und zu analysieren.

```python
# Beispiel-Skript: run_single_sim.py

from impaler.core.model import EconomicModel
from impaler.core.config import create_default_config

if __name__ == "__main__":
    # Erstelle eine Standardkonfiguration und passe sie bei Bedarf an
    config = create_default_config()
    config.simulation_steps = 50
    config.logging_config.log_level = "INFO"
    config.agent_populations["default_producers"].count = 10
    config.agent_populations["default_consumers"].count = 40
    
    # Initialisiere und starte das Modell
    model = EconomicModel(config=config)
    print("Starte einzelne Simulation...")
    model.run_simulation()
    
    # Speichere die Ergebnisse
    model.save_results(path="single_run_results.json", format="json")
    print("Simulation abgeschlossen. Ergebnisse in 'single_run_results.json' gespeichert.")

```

### 2. Evolutionäre Modellsuche

Dies ist der leistungsstärkste Modus, um robuste und leistungsfähige Modellparameter zu finden. Das Skript `run_evolution.py` führt parallel viele Simulationen aus und "züchtet" die besten Konfigurationen über mehrere Generationen.

**Starten der evolutionären Suche:**
```bash
python run_evolution.py
```
-   Die Konfiguration des genetischen Algorithmus (Populationsgröße, Mutationsrate etc.) befindet sich direkt in `run_evolution.py`.
-   Der zu durchsuchende Parameterraum (`PARAMETER_SPACE`) ist in `evolution/parameter_space.py` definiert.
-   Die Fitness eines jeden Modells wird durch `evolution/fitness.py` bestimmt.
-   Die Ergebnisse der besten Genome jeder Generation werden im Verzeichnis `evolution_results/` gespeichert.

### 3. Training von RL-Agenten (MARL)

Das `marl`-Verzeichnis ist für das Training der Agenten mit Reinforcement Learning vorbereitet. Es nutzt Ray RLlib und die definierte PettingZoo-Umgebung.

**Starten eines Trainingslaufs (z.B. mit QMIX):**
```bash
python marl/train.py --config marl/config/qmix.yaml
```
-   Füge `--wandb` hinzu, um das Training mit [Weights & Biases](https://wandb.ai/) zu loggen (erfordert `wandb login`).
-   Die Konfiguration des RL-Algorithmus (z.B. `qmix.yaml`) steuert Hyperparameter wie Lernrate, Netzwerkarchitektur und Umgebungseinstellungen.
-   Die Reward-Funktion wird durch die Konfiguration in `marl/config/reward.yml` gewichtet.

## Konfiguration

Die gesamte Simulation wird durch das Pydantic-Modell `SimulationConfig` in `core/config.py` gesteuert. Dies ermöglicht eine robuste, validierte und leicht erweiterbare Konfiguration über Python-Dictionaries oder JSON-Dateien.

-   **Struktur:** Die Konfiguration ist in logische Blöcke wie `regional_config`, `environment_config`, `planning_priorities` oder `agent_populations` unterteilt.
-   **Anpassung:** Sie können eine Standardkonfiguration mit `create_default_config()` erstellen und diese programmatisch anpassen oder eine komplette Konfiguration als Dictionary oder JSON-Datei laden.
-   **Agenten-Definition:** Agenten können als Populationen (`agent_populations`) mit randomisierten Parametern oder als spezifische, individuelle Instanzen (`specific_agents`) definiert werden.

## Tests

Das Projekt verfügt über eine umfassende Testsuite mit `pytest`.

**Tests ausführen:**
```bash
pytest
```
Dies führt alle Unit- und Integrationstests im `tests/`-Verzeichnis aus und stellt die korrekte Funktionalität der Kernkomponenten sicher.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Weitere Details finden Sie in der `LICENSE`-Datei.