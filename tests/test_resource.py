# tests/test_resource.py

import pytest
from unittest.mock import Mock
from impaler.agents.resource import ResourceAgent

@pytest.fixture
def mock_model():
    """
    Ein einfacher Mock für das Modell, das 'resource_capacity' und 
    'resource_regeneration' bereitstellt.
    Wir können hier ggf. weitere Attribute ergänzen, 
    falls ResourceAgent sie erwartet.
    """
    model = Mock()
    model.resource_capacity = {
        "iron_ore": 1000.0,
        "wood": 1500.0,
        "energy": 2000.0
    }
    model.resource_regeneration = {
        "iron_ore": 0.05,
        "wood": 0.1,
        "energy": 0.15
    }
    return model


def test_resource_agent_initialization(mock_model):
    """
    Prüft, ob der ResourceAgent korrekt initialisiert wird:
    - Inventory, Capacity, Regeneration
    - Zufallswerte für environment_impact, depletion_rate, resource_quality
    """
    agent = ResourceAgent(unique_id="test_resource", model=mock_model)

    assert agent.unique_id == "test_resource"
    # Die Standard-Inventory sollte iron_ore/wood/energy = 0.0 haben
    assert agent.inventory["iron_ore"] == 0.0
    assert agent.inventory["wood"] == 0.0
    assert agent.inventory["energy"] == 0.0

    # Kapazitäten kommen vom Mock-Model
    assert agent.resource_capacity["iron_ore"] == 1000.0
    assert agent.resource_capacity["wood"] == 1500.0

    # Check random attributes (environment_impact, depletion_rate, resource_quality)
    # Wir prüfen hier nur, ob sie existieren und in validem Bereich liegen
    for res in agent.resource_capacity:
        assert 0.1 <= agent.environmental_impact[res] <= 0.3
        assert 0.01 <= agent.depletion_rate[res] <= 0.05
        assert 0.7 <= agent.resource_quality[res] <= 1.0


def test_update_resource_regeneration(mock_model):
    """
    Testet, ob die ResourceAgent.update_resource_regeneration() 
    die regenerationsraten korrekt modifiziert, 
    abhängig von Beständen, environment_impact & sustainability_index.
    """
    agent = ResourceAgent("test_resource", mock_model)
    # Wir modifizieren inventory, cumulative_impact, etc.
    agent.inventory["iron_ore"] = 100.0
    agent.cumulative_impact = 10.0  # moderate Umweltbelastung
    old_regen = agent.resource_regeneration["iron_ore"]

    agent.update_resource_regeneration()
    new_regen = agent.resource_regeneration["iron_ore"]

    # Regeneration sollte tendenziell < old_regen sein, 
    # weil cumulative_impact die Rate senkt
    assert new_regen <= old_regen


def test_extract_resources(mock_model):
    """
    Testet extract_resources(). Dabei prüfen wir, ob:
    - Der korrekte Teil entnommen wird.
    - environmental_impact & cumulative_impact steigen.
    - Der extracted Wert nicht die vorhandene Menge übersteigt.
    """
    agent = ResourceAgent("test_resource", mock_model)
    agent.inventory["iron_ore"] = 200.0
    # Extraktionseffizienz default: 0.8
    # environment_impact[iron_ore] ~ random(0.1 .. 0.3)

    old_cumulative = agent.cumulative_impact
    request = {"iron_ore": 50.0}
    result = agent.extract_resources(request)

    # Da efficiency=0.8 => real_extracted ~ 40
    assert pytest.approx(result["iron_ore"], 0.001) == 40.0
    # Inventory sinkt um 50 (weil "actual_extractable")
    assert agent.inventory["iron_ore"] == 150.0
    # cumulative_impact sollte sich erhöhen 
    # => agent.environmental_impact["iron_ore"] * extracted_amount 
    # z.B. ~ 0.1..0.3 * 50
    assert agent.cumulative_impact > old_cumulative


def test_invest_in_sustainability(mock_model):
    """
    Prüft, ob invest_in_sustainability() die Parameter 
    (environmental_impact, extraction_efficiency, sustainability_index) 
    erwartungsgemäß anpasst.
    """
    agent = ResourceAgent("test_resource", mock_model)
    old_sustain = agent.sustainability_index
    old_eff = agent.extraction_efficiency
    old_impact = agent.environmental_impact["iron_ore"]

    agent.invest_in_sustainability(invest_factor=0.05)

    # sustainability_index sollte steigen
    assert agent.sustainability_index > old_sustain
    # extraction_efficiency sollte steigen
    assert agent.extraction_efficiency > old_eff
    # environmental_impact sollte sinken (0.05 => minimal)
    assert agent.environmental_impact["iron_ore"] < old_impact


def test_resource_depletion_cycle(mock_model):
    """
    Testet den resource_depletion_cycle(). 
    Hier wird net regeneration (regen_rate * current_amt) - depletion (depletion_rate * current_amt) berechnet
    und der neue Bestand geprüft.
    """
    agent = ResourceAgent("test_resource", mock_model)
    agent.inventory["iron_ore"] = 200.0
    # z.B. regen_rate=0.05 => 10.0 add
    # depletion_rate ~ random(0.01..0.05). 
    # => net_change = 10 - (x * 200)

    # Vorher
    old_amt = agent.inventory["iron_ore"]
    agent.resource_depletion_cycle()
    new_amt = agent.inventory["iron_ore"]

    # Check plausibel
    assert new_amt >= old_amt  * 0.9  # (z.B. worst case: depletion=0.05 => 10 - 10=0 net => ~200)
    assert new_amt <= 1000.0  # capacity


def test_step_stage_resource_regen(mock_model):
    """
    Prüft das step_stage("resource_regen"). 
    Dabei wird:
     1) update_resource_regeneration()
     2) resource_depletion_cycle()
     3) ggf. invest_in_sustainability() aufgerufen.
    """
    agent = ResourceAgent("test_resource", mock_model)
    agent.inventory["iron_ore"] = 50.0
    agent.sustainability_index = 0.55  # < 0.6 => invest in sustainability

    old_index = agent.sustainability_index
    agent.step_stage("resource_regen")

    # => Nach "resource_regen" sollte sustainability_index minimal angehoben werden.
    assert agent.sustainability_index > old_index


def test_step_stage_ignored_stage(mock_model):
    """
    Prüft, ob ein stage-Call mit z.B. 'resource_bookkeeping' 
    nicht abstürzt (default: pass).
    """
    agent = ResourceAgent("test_resource", mock_model)
    # Ruft stage auf, das nichts tut
    agent.step_stage("resource_bookkeeping")
    # Keine Exception => Test ok
    assert True
