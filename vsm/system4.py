# Impaler/vsm/system4.py
"""
System 4: Strategische Planung und Intelligenz im VSM.

Verantwortlich für die langfristige Ausrichtung, die Festlegung globaler Ziele
und die Anpassung der Wirtschaftsstrategie basierend auf internem Feedback
und externen Bedingungen. Nutzt Modelle wie ADMM zur Koordination und
Optimierung unter Berücksichtigung von Ressourcen, Fairness und anderen Zielen,
anstelle klassischer Markt-/Profitlogik. Enthält auch die 'Intelligence'-Funktion
zur Analyse und Prognose.
"""

from __future__ import annotations

import random
import logging
import numpy as np
import math  # Added for math.sqrt in ADMM residuals
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple, Union, Deque, DefaultDict, TYPE_CHECKING, Callable

# Typ-Prüfung Imports
if TYPE_CHECKING:
    from ..core.model import EconomicModel
    from ..agents.producer import ProducerAgent
    from .communication_contracts import (
        OperationalReportS3,
        StrategicDirectiveS4,
        ConflictResolutionDirectiveS2,
        SystemFeedback,
    )
    # For parallel execution if used (conceptual for now)
    # from joblib import Parallel, delayed

# Importiere Konfigurationsmodelle
from ..core.config import ADMMConfig, PlanningPriorities 

# Utils
from ..utils.math_utils import gini_coefficient

# Laufzeit-Import für verwendete Nachrichtenklassen
from .communication_contracts import StrategicDirectiveS4

class System4Planner:
    """
    System4Planner - Strategische Planungs- und Intelligenzinstanz im VSM.
    """
    model: 'EconomicModel'
    logger: logging.Logger
    config: Dict[str, Any] 
    admm_config: ADMMConfig
    priority_config: PlanningPriorities
    admm_on: bool
    rho: float
    tolerance: float
    max_iterations: int
    intelligence_system: 'PlannerIntelligence'
    z_consensus: Dict[str, float]
    u_duals: DefaultDict[str, DefaultDict[str, float]]
    admm_convergence_history: List[Dict[str, Any]]
    plan_performance_history: List[Dict[str, Any]]
    current_strategic_plan: Optional['StrategicDirectiveS4'] 
    last_operational_report: Optional['OperationalReportS3'] 
    pending_feedback: Deque[Union['ConflictResolutionDirectiveS2', 'SystemFeedback']] 
    active_planning_method: str
    plan_relevant_goods: List[str]

    # Flags for intelligence system recommendations
    _needs_parameter_review: bool
    _needs_admm_rho_adjustment: Optional[str]
    _recommended_priority_increase_good: Optional[str]
    
    # Scenario Planning Config (mirrored from self.config for direct access if needed)
    _enable_scenario_planning: bool


    def __init__(self, unique_id: str, model: 'EconomicModel'):
        self.unique_id = unique_id
        self.model = model
        self.logger = model.logger.getChild('System4Planner') # Create a child logger for S4
        self.config = getattr(model.config, "system4_params", {})

        self.admm_config = model.config.admm_config
        self.priority_config = model.config.planning_priorities
        self.admm_on = model.config.admm_on

        # ADMM parameter shortcuts
        self.rho = self.admm_config.rho
        self.tolerance = self.admm_config.tolerance
        self.max_iterations = self.admm_config.max_iterations

        admm_params = getattr(model.config, "admm_parameters", {})
        self.lambdas: Dict[str, float] = admm_params.get("lambdas", {}).copy()

        self.z_consensus = {}
        self.u_duals = defaultdict(lambda: defaultdict(float))
        self.admm_convergence_history = []
        self.plan_performance_history = []
        # Vector of shared model parameters for federated updates
        self.global_params = np.zeros(3, dtype=np.float32)
        self._prev_x_locals: Dict[str, Dict[str, float]] = defaultdict(dict)

        self.current_strategic_plan = None
        self.last_operational_report = None
        self.pending_feedback = deque(maxlen=50) 
        self.active_planning_method = self.config.get("initial_planning_method", "comprehensive")
        # Some configs may not define 'plan_relevant_goods'
        self.plan_relevant_goods = getattr(model.config, "plan_relevant_goods", model.config.goods)

        # Initialize flags for intelligence recommendations
        self._needs_parameter_review = False
        self._needs_admm_rho_adjustment = None
        self._recommended_priority_increase_good = None

        # Scenario Planning Config
        self._enable_scenario_planning = self.config.get("enable_scenario_planning", False) # Default to False
        self.logger.info(f"Scenario-based planning feature in System4Planner is {'ENABLED' if self._enable_scenario_planning else 'DISABLED'} via config.")
        
        # Intelligence Subsystem - crucially, pass self.config to PlannerIntelligence
        self.intelligence_system = PlannerIntelligence(self)

        self.planning_methods: Dict[str, Callable] = {
            "comprehensive": self.comprehensive_planning,
            # Add other planning methods here if they exist
            "iterative": self.iterative_planning,
            "hierarchical": self.hierarchical_planning,
            "distributed": self.distributed_planning,
            "optimization_based": self.optimization_based_planning,
        }
        self.logger.info(
            f"System4Planner initialisiert (ADMM: {self.admm_on}, Rho: {self.rho}, Tol: {self.tolerance}, Methode: {self.active_planning_method})."
        )

    def handle_crisis_start(self, crisis_type: str, effects: Dict[str, Any]) -> None:
        """Placeholder to react to crisis start notifications."""
        self.logger.info(f"Received crisis start notification: {crisis_type}")

    def handle_crisis_end(self, crisis_type: str) -> None:
        """Placeholder to react to crisis end notifications."""
        self.logger.info(f"Received crisis end notification: {crisis_type}")

    # Convenience accessors for ADMM parameter
    @property
    def rho(self) -> float:
        return self.admm_config.rho

    @rho.setter
    def rho(self, value: float) -> None:
        self.admm_config.rho = value

    def step_stage(self, stage: str) -> None:
        self.logger.debug(f"Executing stage: {stage} at S4 step {self.model.current_step}")

        if stage == "system4_strategic_planning" or stage == "system4_tactical_planning":
            self._process_pending_feedback()
            self.adapt_planning_strategy() 
            
            plan_func = self.planning_methods.get(self.active_planning_method, self.comprehensive_planning)
            new_plan_directives_obj = plan_func() 
            
            if new_plan_directives_obj:
                self.current_strategic_plan = new_plan_directives_obj
                self.logger.info(f"Neuer strategischer Plan für S4 Step {self.model.current_step + 1} erstellt (Methode: {self.active_planning_method}).")
                # Example: Record a summary of the plan as a "decision"
                # More detailed decisions (like specific investments) should be recorded where they are made.
                plan_summary_details = {
                    "target_goods_count": len(new_plan_directives_obj.production_targets),
                    "autonomy_level_to_s3": new_plan_directives_obj.system3_autonomy_level 
                }
                self.intelligence_system.record_planning_decision(
                    decision_type="overall_strategic_plan_generated",
                    details=plan_summary_details,
                    step=self.model.current_step # Record with current S4 step
                )

            # Intelligenz-System aktualisieren (runs its own internal logic for intervals)
            self.intelligence_system.update()

        elif stage == "admm_update" and self.admm_on:
            self.run_admm_iteration()

    def receive_operational_report(self, report: 'OperationalReportS3') -> None: # type: ignore
        self.logger.info(f"System 4 empfängt operativen Bericht von System 3 (S3 Step {report.step}, S4 Step {self.model.current_step})")
        self.last_operational_report = report

        # --- Record outcomes for Causal Inference ---
        # This is a good place to record outcomes based on S3's report
        outcome_details = {
            "coordination_effectiveness_s3": report.coordination_effectiveness,
            "average_regional_gini_s3": report.regional_status.get('average_gini', np.nan), # Use nan if missing
        }
        for good, fulfillment in report.plan_fulfillment_summary.items():
            outcome_details[f"fulfillment_rate_{good}"] = fulfillment
        if report.bottlenecks:
            outcome_details["bottleneck_goods_count_s3"] = len(report.bottlenecks)
        
        self.intelligence_system.record_system_outcome(
            outcome_data=outcome_details, 
            step=self.model.current_step # Record with current S4 step
        )
        # (Rest of the existing S3 report processing logic follows)
        effectiveness = report.coordination_effectiveness
        if effectiveness < 0.65:
            old_rho = self.admm_config.rho
            self.admm_config.rho = min(old_rho * 1.05, 5.0)
            self.logger.warning(f"S3 Koordinationseffektivität niedrig ({effectiveness:.2f}). Erhöhe ADMM Rho auf {self.admm_config.rho:.3f}.")
        elif effectiveness > 0.85:
             old_rho = self.admm_config.rho
             self.admm_config.rho = max(old_rho * 0.98, 0.05) 
             self.logger.info(f"S3 Koordinationseffektivität hoch ({effectiveness:.2f}). Reduziere ADMM Rho auf {self.admm_config.rho:.3f}.")

        low_fulfillment_goods = {g: f for g, f in report.plan_fulfillment_summary.items() if f < 0.8}
        if low_fulfillment_goods:
            self.logger.warning(f"Niedrige Planerfüllung für: {low_fulfillment_goods}. Passe Prioritäten an.")
            # Potentially use causal insights here: if low fulfillment for good X, and S4 knows
            # that investment Y usually helps good X, it might trigger such an investment.
            # For now, existing logic is kept:
            for good, fulfillment in low_fulfillment_goods.items():
                 current_prio = self.priority_config.goods_priority.get(good, 0.5)
                 increase_factor = 1.0 + (0.8 - fulfillment) * 0.5 
                 self.priority_config.goods_priority[good] = min(2.0, current_prio * increase_factor)

        if report.bottlenecks:
             self.logger.warning(f"Von System 3 gemeldete Engpässe: {list(report.bottlenecks.keys())}. Berücksichtige in nächster Planung.")

        regional_gini = report.regional_status.get('average_gini', 0.0)
        fairness_threshold = getattr(self.priority_config, 'fairness_threshold', 0.8)
        if regional_gini > fairness_threshold * 1.1:
            old_weight = self.priority_config.fairness_weight
            self.priority_config.fairness_weight = min(3.0, old_weight * 1.1)
            self.logger.warning(f"Regionale Ungleichheit hoch (Gini: {regional_gini:.3f}). Erhöhe Fairness-Gewicht auf {self.priority_config.fairness_weight:.2f}")

    def receive_directive_or_feedback(self, message: Union['ConflictResolutionDirectiveS2', 'SystemFeedback']) -> None: # type: ignore
        source = getattr(message, 'source_system', 'Unknown')
        self.logger.info(f"System 4 empfängt Nachricht von {source}: Typ='{message.action_type if hasattr(message, 'action_type') else type(message).__name__}'")
        self.pending_feedback.append(message)

    def _process_pending_feedback(self) -> None:
         processed_count = 0
         while self.pending_feedback:
              message = self.pending_feedback.popleft()
              action_type = getattr(message, 'action_type', None)
              payload = getattr(message, 'payload', {})
              try:
                   if action_type == "request_plan_adjustment":
                        self.logger.warning(f"Empfange 'request_plan_adjustment' von S2 für Regionen {payload.get('regions')}. Konflikt: {payload.get('conflict_type')}")
                        self._flag_for_plan_review(payload.get('regions'), payload.get('conflict_type'))
                   elif action_type == "request_intervention":
                        self.logger.error(f"Empfange Eskalation 'request_intervention' von S2 für Regionen {payload.get('regions')}. Problem: {payload.get('conflict_type')}. Leite ggf. an System 5 weiter.")
                        if self.model.system5policy and hasattr(self.model.system5policy, 'handle_escalation'):
                             self.model.system5policy.handle_escalation(message) # type: ignore
                   else:
                        self.logger.debug(f"Verarbeite generisches Feedback: {message}")
                   processed_count += 1
              except Exception as e:
                   self.logger.error(f"Fehler bei der Verarbeitung von Feedback/Direktive {message}: {e}", exc_info=True)
         if processed_count > 0:
              self.logger.info(f"{processed_count} anstehende Feedbacks/Direktiven verarbeitet.")

    def get_strategic_directives(self) -> Optional['StrategicDirectiveS4']: # type: ignore
        if self.current_strategic_plan:
             self.logger.debug(f"Gebe strategische Direktiven (für S4 Step {self.current_strategic_plan.step}) an System 3 weiter.")
             return self.current_strategic_plan
        else:
             self.logger.warning("Keine aktuellen strategischen Direktiven verfügbar für System 3.")
             return None

    def update_policy_parameters(self, policy_updates: Dict[str, Any]) -> None:
        """Apply policy updates sent from System 5 at runtime."""
        self.logger.info("System 4 empfängt Politik-Parameter-Updates von System 5...")

        if "lambdas" in policy_updates and isinstance(policy_updates["lambdas"], dict):
            self.lambdas.update(policy_updates["lambdas"])
            self.logger.debug(
                f"ADMM Lambdas aktualisiert. Neue/geänderte Keys: {list(policy_updates['lambdas'].keys())}"
            )

        if "priorities" in policy_updates and isinstance(policy_updates["priorities"], dict):
            for key, value in policy_updates["priorities"].items():
                if hasattr(self.priority_config, key):
                    setattr(self.priority_config, key, value)
                    self.logger.debug(f"Planungspriorität '{key}' auf {value} aktualisiert.")
                else:
                    self.logger.warning(f"Versuch, unbekannte Priorität '{key}' zu aktualisieren, wird ignoriert.")
    
    def adapt_planning_strategy(self) -> None:
        """Passt die Planungsstrategie oder Parameter basierend auf Intelligence-Empfehlungen an."""
        self.logger.debug("Prüfe Notwendigkeit zur Anpassung der Planungsstrategie...")
        if self._needs_parameter_review:
            self.logger.info("PlannerIntelligence empfiehlt Parameter-Review. (Logik hierfür noch zu implementieren)")
            # TODO: Implement logic to adjust planning parameters based on this flag
            self._needs_parameter_review = False # Reset flag

        if self._recommended_priority_increase_good:
            good_to_prioritize = self._recommended_priority_increase_good
            if good_to_prioritize in self.priority_config.goods_priority:
                self.priority_config.goods_priority[good_to_prioritize] *= 1.15 # Increase by 15%
                self.logger.info(f"Priorität für Gut '{good_to_prioritize}' aufgrund von Intelligence-Empfehlung auf {self.priority_config.goods_priority[good_to_prioritize]:.2f} erhöht.")
            else:
                self.priority_config.goods_priority[good_to_prioritize] = 1.15 # Set initial higher priority
                self.logger.info(f"Priorität für Gut '{good_to_prioritize}' aufgrund von Intelligence-Empfehlung initial auf 1.15 gesetzt.")
            self._recommended_priority_increase_good = None # Reset flag
        
        if self._needs_admm_rho_adjustment:
            adjustment_direction = self._needs_admm_rho_adjustment
            current_rho = self.admm_config.rho
            if adjustment_direction == "increase":
                self.admm_config.rho = min(current_rho * 1.1, 10.0) # Cap at 10.0
            elif adjustment_direction == "decrease":
                self.admm_config.rho = max(current_rho * 0.9, 0.01) # Floor at 0.01
            self.logger.info(f"ADMM Rho aufgrund von Intelligence-Empfehlung von {current_rho:.3f} auf {self.admm_config.rho:.3f} angepasst.")
            self._needs_admm_rho_adjustment = None # Reset flag
        
        # --- Integration von Kausalen Einsichten ---
        causal_insights = self.intelligence_system.get_causal_insights()
        if causal_insights:
            self.logger.info(f"Verwende kausale Einsichten in der Planungsstrategie: {causal_insights}")
            # Beispiel: Wenn eine Investition in R&D für Sektor X positiv mit Tech-Level korreliert,
            # und das aktuelle Ziel ist, Tech-Level in Sektor X zu erhöhen,
            # dann könnte die Priorität für R&D Investitionen in Sektor X erhöht werden.
            # This logic would be quite complex and depend on current strategic goals.
            # For now, just logging the insights. More detailed integration is a TODO.
            for insight_key, details in causal_insights.items():
                if details.get("type") == "positive" and details.get("correlation",0) > 0.5 and "invest_rnd" in insight_key and "tech_change" in insight_key:
                    # e.g. insight_key = "invest_rnd_electronics_to_tech_change_lag3"
                    # Potentially increase propensity to invest in R&D for 'electronics'
                    self.logger.debug(f"Positive Korrelation '{insight_key}' könnte R&D Priorisierung beeinflussen.")
                    # Actual modification of planning parameters based on this is complex and not yet implemented.


    def comprehensive_planning(self) -> Optional['StrategicDirectiveS4']: # type: ignore
        self.logger.info(f"Starte umfassenden Planungszyklus für S4 Step {self.model.current_step}...")
        max_iterations = self.config.get("planning_max_iterations", 3) # Reduced for brevity

        demand_estimate = self._estimate_global_demand()
        if not demand_estimate:
             self.logger.error("Keine Nachfrageschätzung verfügbar. Planung abgebrochen.")
             return None

        current_plan_targets = self._basic_capacity_allocation(demand_estimate)
        
        # --- Record Investment Decision Example ---
        # This is a simplified example. Real investment logic would be more nuanced.
        # Assume we decide to invest a portion of surplus or a fixed amount into R&D for all sectors.
        investment_allocations_details: Dict[str, Dict[str, float]] = {}
        for good_name in self.plan_relevant_goods: # Assuming goods map to sectors or have associated sectors
            # Simplified: find a sector for this good. In reality, goods-to-sector mapping is needed.
            example_sector_name = good_name # This is too simple, needs proper mapping
            
            # Use Causal Insights: if R&D for this sector is known to be effective, invest more.
            invest_more_rnd = False
            for insight_key, details in self.intelligence_system.get_causal_insights().items():
                if f"invest_rnd_{example_sector_name}" in insight_key and details.get("correlation",0) > 0.5:
                    invest_more_rnd = True
                    self.logger.info(f"Insight '{insight_key}' (Corr: {details['correlation']}) motiviert höhere R&D für {example_sector_name}.")
                    break
            
            base_rnd_investment = self.config.get("base_rnd_investment_per_sector", 10)
            final_rnd_investment = base_rnd_investment * 1.5 if invest_more_rnd else base_rnd_investment
            
            investment_allocations_details[f"sector_{example_sector_name}"] = {
                "technology_research": final_rnd_investment,
                "capacity_expansion": self.config.get("base_capacity_investment_per_sector", 20) # Example fixed amount
            }
        if investment_allocations_details:
            self.intelligence_system.record_planning_decision(
                decision_type="investment_allocation",
                details={"allocations": investment_allocations_details}, # Nested dict for clarity
                step=self.model.current_step
            )
        # --- End Record Investment Decision Example ---

        # Main planning loop (iterations of refinement)
        for iteration in range(max_iterations):
            self.logger.debug(f"Planungsiteration {iteration + 1}/{max_iterations}")
            temp_plan_targets = current_plan_targets # Use a temporary variable for revisions within an iteration
            
            # Estimate impacts of the current temporary plan
            estimated_usage = self._estimate_resource_usage(temp_plan_targets)
            estimated_emissions = self._estimate_emissions(temp_plan_targets)
            estimated_fairness_gini = self._estimate_fairness_gini(temp_plan_targets)
            violations = self._check_plan_violations(estimated_usage, estimated_emissions, estimated_fairness_gini)

            # Revise plan based on violations, feedback, ADMM, etc.
            if self.admm_on:
                 revised_plan_targets_iter = self.run_admm_iteration(initial_targets=temp_plan_targets, n_iter=2)
            else:
                 revised_plan_targets_iter = self._revise_plan_heuristically(temp_plan_targets, violations, self.last_operational_report)

            if self._has_plan_converged(current_plan_targets, revised_plan_targets_iter):
                self.logger.info(f"Planungskonvergenz nach Iteration {iteration + 1}.")
                current_plan_targets = revised_plan_targets_iter # Lock in the converged plan
                break 
            current_plan_targets = revised_plan_targets_iter # Continue with revised plan
        else: # If loop finishes without break (no convergence)
            self.logger.warning(f"Planung erreichte maximale Iterationszahl ({max_iterations}) ohne formale Konvergenz.")
        
        # --- Scenario-Based Robustness Check (after iterative refinement) ---
        if self._enable_scenario_planning: # Check S4's config flag
            self.logger.info(f"[S4 Step {self.model.current_step}] Initiating scenario-based robustness evaluation for the current plan draft.")
            # Assuming evaluate_candidate_plan_robustness is now part of PlannerIntelligence
            robustness_report = self.intelligence_system.evaluate_candidate_plan_robustness(current_plan_targets)
            self.logger.info(f"[S4 Step {self.model.current_step}] Plan Robustness Report: {robustness_report}")
            
            # Simplified adjustment based on robustness report
            avg_fulfillment = robustness_report.get("average_projected_fulfillment_all_scenarios", 1.0) # Default to 1.0 if key missing
            worst_deficits = robustness_report.get("worst_case_resource_deficit_summary_percentage", {})

            if avg_fulfillment < self.config.get("scenario_min_avg_fulfillment_threshold", 0.75): 
                reduction_factor = self.config.get("scenario_target_reduction_factor_low_fulfillment", 0.95) # e.g. 5% reduction
                self.logger.warning(f"Low average plan fulfillment ({avg_fulfillment:.2f}) under simulated shocks. Reducing overall targets by {(1-reduction_factor)*100:.1f}%.")
                for good in current_plan_targets:
                    current_plan_targets[good] *= reduction_factor
            elif any(def_rate > self.config.get("scenario_max_resource_deficit_threshold", 0.25) for def_rate in worst_deficits.values()):
                reduction_factor = self.config.get("scenario_target_reduction_factor_high_deficit", 0.97) # e.g. 3% reduction
                self.logger.warning(f"High resource deficit observed in worst-case scenarios ({worst_deficits}). Reducing overall targets by {(1-reduction_factor)*100:.1f}%.")
                for good in current_plan_targets:
                    current_plan_targets[good] *= reduction_factor
            else:
                self.logger.info("Current plan draft deemed sufficiently robust or no major vulnerabilities identified by scenario simulation.")
        # --- End Scenario-Based Robustness Check ---

        final_plan_targets = self._finalize_plan(current_plan_targets) 
        self._track_plan_performance(final_plan_targets, demand_estimate)

        directives = StrategicDirectiveS4( # type: ignore
            step=self.model.current_step + 1, 
            production_targets=final_plan_targets,
            resource_allocation_guidelines={"fairness_target": self.priority_config.fairness_weight},
            system3_autonomy_level=getattr(self.model.system3manager, 'autonomy_level', 0.7) if self.model.system3manager else 0.7 # type: ignore
        )
        return directives

    def iterative_planning(self, *_, **__) -> Optional['StrategicDirectiveS4']: # type: ignore
         self.logger.warning("Iterative Planung noch nicht implementiert.")
         return self.comprehensive_planning() # Fallback to comprehensive for now
    def hierarchical_planning(self, *_, **__) -> Optional['StrategicDirectiveS4']: # type: ignore
         self.logger.warning("Hierarchische Planung noch nicht implementiert.")
         return self.comprehensive_planning() # Fallback
    def distributed_planning(self, *_, **__) -> Optional['StrategicDirectiveS4']: # type: ignore
         self.logger.warning("Verteilte Planung noch nicht implementiert.")
         return self.comprehensive_planning() # Fallback
    def optimization_based_planning(self, *_, **__) -> Optional['StrategicDirectiveS4']: # type: ignore
         self.logger.warning("Optimierungsbasierte Planung noch nicht implementiert.")
         return self.comprehensive_planning() # Fallback

    def _estimate_global_demand(self) -> Dict[str, float]:
         if hasattr(self.model, 'aggregated_consumer_demand'):
             return self.model.aggregated_consumer_demand.copy() # type: ignore
         self.logger.warning("Keine aggregierten Konsumentendaten verfügbar für Nachfrageschätzung.")
         return {good: 100.0 for good in self.plan_relevant_goods} 

    def _basic_capacity_allocation(self, demand: Dict[str, float]) -> Dict[str, float]:
         initial_targets = {}
         total_capacity_per_good = defaultdict(float)
         for p in self.model.producers: # type: ignore
             cap = getattr(p, 'productive_capacity', 0)
             for good in getattr(p, 'can_produce', set()):
                  if good in self.plan_relevant_goods:
                       total_capacity_per_good[good] += cap
         for good, dem_val in demand.items():
             total_cap = total_capacity_per_good.get(good, 0)
             initial_targets[good] = min(dem_val, total_cap) if total_cap > 0 else 0.0
         self.logger.debug(f"Initialer Kapazitätsbasierter Plan (Gesamtziele): {initial_targets}")
         return initial_targets

    def _estimate_resource_usage(self, production_targets: Dict[str, float]) -> Dict[str, float]:
         self.logger.debug(f"Placeholder _estimate_resource_usage für Ziele: {production_targets}")
         return defaultdict(float)
    def _estimate_emissions(self, production_targets: Dict[str, float]) -> Dict[str, float]:
         self.logger.debug(f"Placeholder _estimate_emissions für Ziele: {production_targets}")
         return defaultdict(float)
    def _estimate_fairness_gini(self, production_targets: Dict[str, float]) -> float:
         self.logger.debug(f"Placeholder _estimate_fairness_gini für Ziele: {production_targets}")
         return 0.2 
    def _check_plan_violations(self, usage: Dict, emissions: Dict, fairness_gini: float) -> Dict[str, float]:
         self.logger.debug(f"Placeholder _check_plan_violations. Usage: {usage}, Emissions: {emissions}, Gini: {fairness_gini}")
         return {}
    def _revise_plan_heuristically(self, current_targets: Dict, violations: Dict, feedback: Optional['OperationalReportS3']) -> Dict: # type: ignore
         self.logger.debug(f"Placeholder _revise_plan_heuristically. Targets: {current_targets}, Violations: {violations}")
         return current_targets 
    def _has_plan_converged(self, old_targets: Dict, new_targets: Dict) -> bool:
         diff = sum(abs(new_targets.get(g, 0) - old_targets.get(g, 0)) for g in set(old_targets) | set(new_targets))
         max_total = max(sum(old_targets.values()), sum(new_targets.values()), 1.0) # Avoid division by zero
         relative_diff = diff / max_total
         return relative_diff < self.config.get("planning_convergence_threshold", 0.01)

    def _finalize_plan(self, targets: Dict[str, float]) -> Dict[str, float]:
         return {g: max(0, round(t, 2)) for g, t in targets.items()}
    def _track_plan_performance(self, final_plan: Dict, demand: Dict) -> None:
        """Speichert einfache Kennzahlen zur Planerfüllung für Analysezwecke."""
        step = self.model.current_step + 1
        entry: Dict[str, Any] = {"s4_step": step}
        total_demand = 0.0
        total_planned = 0.0

        for good, demand_val in demand.items():
            plan_val = final_plan.get(good, 0.0)
            total_demand += demand_val
            total_planned += plan_val
            coverage = plan_val / demand_val if demand_val > 0 else 1.0
            entry[f"coverage_{good}"] = round(coverage, 3)

        overall = total_planned / total_demand if total_demand > 0 else 1.0
        entry["overall_coverage"] = round(overall, 3)

        self.plan_performance_history.append(entry)
        # Auch im Intelligence-System hinterlegen
        self.intelligence_system.record_system_outcome(entry, step)

    def run_admm_iteration(self, initial_targets: Optional[Dict[str, float]] = None, n_iter: Optional[int] = None) -> Dict[str, float]:
        if not self.admm_on:
            self.logger.warning("ADMM ist deaktiviert, überspringe ADMM Update.")
            return initial_targets or {}

        max_iterations = n_iter if n_iter is not None else self.admm_config.max_iterations
        tolerance = self.admm_config.tolerance
        rho = self.admm_config.rho 

        active_producers = [p for p in self.model.producers if not getattr(p, "bankrupt", False)] # type: ignore
        if not active_producers:
            self.logger.warning("Keine aktiven Producer für ADMM gefunden.")
            return {}

        if not self.z_consensus or initial_targets:
             targets_to_use = initial_targets if initial_targets else self._estimate_global_demand()
             self.z_consensus = targets_to_use.copy()
             self.logger.info(f"Initialisiere/Resette ADMM z_consensus mit Zielen: {list(self.z_consensus.keys())}")

        self.logger.info(f"Starte ADMM-Iteration (max {max_iterations} Iterationen, rho={rho:.3f}, tol={tolerance:.4f})...")
        start_s4_step = self.model.current_step
        final_primal_res, final_dual_res = float('inf'), float('inf')
        
        for admm_i in range(max_iterations):
            agents_prev_solutions = {pid: sol.copy() for pid, sol in self._prev_x_locals.items()}
            # Use lambdas defined in self.lambdas; fall back to empty dict if not provided
            agents_local_solutions = self._admm_update_x(
                active_producers, self.lambdas, rho
            )  # type: ignore
            self._admm_update_z(agents_local_solutions)
            self._admm_update_u(agents_local_solutions)
            primal_res, dual_res = self._compute_admm_residuals_iteration(agents_local_solutions, agents_prev_solutions, rho)
            self.logger.debug(f"[ADMM Iter {admm_i+1}] Primal Res: {primal_res:.4f}, Dual Res: {dual_res:.4f}")
            self._prev_x_locals = {pid: sol.copy() for pid, sol in agents_local_solutions.items()}
            final_primal_res, final_dual_res = primal_res, dual_res
            if primal_res < tolerance and dual_res < tolerance:
                self.logger.info(f"ADMM konvergiert nach {admm_i + 1} Iterationen.")
                break
        else: 
            self.logger.warning(f"ADMM erreichte maximale Iterationszahl ({max_iterations}) ohne Konvergenz.")

        self.admm_convergence_history.append({
            "s4_step": start_s4_step,
            "iterations": admm_i + 1,
            "final_primal_res": final_primal_res,
            "final_dual_res": final_dual_res,
            "final_rho": rho,
        })
        return self.z_consensus.copy()

    def run_admm_optimization(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Führt die robuste ADMM-Optimierung durch, um den globalen Plan (u*)
        und die Schattenpreise (λ*) zu finden.

        Returns:
            Ein Tupel bestehend aus:
            - Dem optimalen Produktionsplan (z_consensus).
            - Den Schattenpreisen für Knappheit (u_duals).
        """
        self.logger.info("Starte DRO-ADMM Optimierung...")

        active_producers = [p for p in self.model.producers if not getattr(p, "bankrupt", False)]  # type: ignore
        if not active_producers:
            self.logger.warning("Keine aktiven Producer für ADMM-Optimierung gefunden. Breche ab.")
            return {}, {}

        if not self.z_consensus:
            demand_estimate = self.intelligence_system.get_demand_forecast()
            self.z_consensus = self._basic_capacity_allocation(demand_estimate)

        for k in range(self.max_iterations):
            x_locals = self._admm_x_update(active_producers)

            z_old = self.z_consensus.copy()
            self._admm_z_update(x_locals)

            self._admm_u_update(x_locals)

            primal_res, dual_res = self._compute_admm_residuals(z_old, x_locals)
            self.admm_convergence_history.append({
                "iteration": k, "primal_residual": primal_res, "dual_residual": dual_res
            })
            self.logger.debug(f"[ADMM Iter {k+1}] Primal Res: {primal_res:.4f}, Dual Res: {dual_res:.4f}")

            if primal_res < self.tolerance and dual_res < self.tolerance:
                self.logger.info(f"ADMM konvergiert nach {k+1} Iterationen.")
                break
        else:
            self.logger.warning(
                f"ADMM hat die maximale Anzahl von {self.max_iterations} Iterationen erreicht, ohne zu konvergieren."
            )

        optimal_plan = self.z_consensus
        shadow_prices = self.u_duals

        self.logger.info("DRO-ADMM Optimierung abgeschlossen.")
        return optimal_plan, shadow_prices

    def fed_update(self, global_vec: np.ndarray) -> None:
        """Simple federated learning update averaging with current parameters."""
        incoming = np.asarray(global_vec, dtype=np.float32)
        if self.global_params.shape != incoming.shape:
            self.global_params = np.zeros_like(incoming)
        self.global_params = 0.5 * self.global_params + 0.5 * incoming
        for p in self.model.producers:
            p.local_model_params = self.global_params.copy()

    def _admm_x_update(self, producers: List['ProducerAgent']) -> Dict[str, Dict[str, float]]:
        """
        Koordiniert den x-update Schritt, indem jeder Producer sein lokales Subproblem löst.
        Gibt die lokalen Lösungen aller Producer zurück.
        """
        local_solutions: Dict[str, Dict[str, float]] = {}
        for producer in producers:
            u_vals_for_producer = self.u_duals.get(producer.unique_id, {})
            solution = producer.local_subproblem_admm(
                goods=list(self.z_consensus.keys()),
                lambdas=self.lambdas,
                z_vals=self.z_consensus,
                u_vals=u_vals_for_producer,
                rho=self.rho,
            )
            local_solutions[producer.unique_id] = solution
        return local_solutions

    def _admm_z_update(self, local_solutions: Dict[str, Dict[str, float]]):
        """Aktualisiert die globale Konsens-Variable 'z'."""
        avg_term: DefaultDict[str, float] = defaultdict(float)
        counts: DefaultDict[str, int] = defaultdict(int)

        for good in self.z_consensus.keys():
            for producer_id, solution in local_solutions.items():
                if good in solution:
                    x_i = solution[good]
                    u_i = self.u_duals[producer_id].get(good, 0.0)
                    avg_term[good] += x_i + u_i
                    counts[good] += 1

        for good, total_val in avg_term.items():
            if counts[good] > 0:
                self.z_consensus[good] = max(0.0, total_val / counts[good])

    def _admm_u_update(self, local_solutions: Dict[str, Dict[str, float]]):
        """Aktualisiert die Dualvariablen 'u' für jeden Producer."""
        for producer_id, solution in local_solutions.items():
            for good, x_i in solution.items():
                z = self.z_consensus.get(good, 0.0)
                self.u_duals[producer_id][good] += (x_i - z)

    def _compute_admm_residuals(self, z_old: Dict[str, float], local_solutions: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
        """Berechnet die primalen und dualen Residuen zur Konvergenzprüfung."""
        primal_res_sq = 0.0
        for local_sol in local_solutions.values():
            for good, x_val in local_sol.items():
                primal_res_sq += (x_val - self.z_consensus.get(good, 0.0)) ** 2
        primal_residual = np.sqrt(primal_res_sq)

        dual_res_sq = 0.0
        for good, z_val in self.z_consensus.items():
            dual_res_sq += (z_val - z_old.get(good, 0.0)) ** 2
        dual_residual = self.rho * np.sqrt(dual_res_sq)

        return primal_residual, dual_residual

    def _admm_update_x(self, producers: List['ProducerAgent'], lambdas: Dict[str, float], rho: float) -> Dict[str, Dict[str, float]]:
        results = {}
        def _solve_producer_subproblem(producer: 'ProducerAgent'):
             if hasattr(producer, 'local_subproblem_admm'):
                  goods_to_optimize = [g for g in self.plan_relevant_goods if g in getattr(producer, 'can_produce', set())]
                  z_vals_for_agent = {g: self.z_consensus.get(g, 0.0) for g in goods_to_optimize}
                  u_vals_for_agent = {g: self.u_duals[producer.unique_id].get(g, 0.0) for g in goods_to_optimize}
                  local_solution = producer.local_subproblem_admm(
                       goods=goods_to_optimize, lambdas=lambdas, alpha_params={},
                       z_vals=z_vals_for_agent, u_vals=u_vals_for_agent, rho=rho
                  )
                  return producer.unique_id, local_solution
             else:
                  self.logger.warning(f"Producer {producer.unique_id} hat keine local_subproblem_admm Methode.")
                  return producer.unique_id, {}
        # Parallel execution can be added here if joblib is available and configured
        for p in producers:
            pid, sol = _solve_producer_subproblem(p)
            results[pid] = sol
        return results

    def _admm_update_z(self, agents_local_solutions: Dict[str, Dict[str, float]]) -> None:
        new_z = defaultdict(float)
        counts = defaultdict(int)
        for agent_id, local_sol in agents_local_solutions.items():
            for good, x_val in local_sol.items():
                 if good in self.plan_relevant_goods:
                      u_val = self.u_duals[agent_id].get(good, 0.0)
                      new_z[good] += (x_val + u_val)
                      counts[good] += 1
        for good in new_z: # Iterate over goods that at least one agent optimized
            if counts[good] > 0:
                self.z_consensus[good] = max(0.0, new_z[good] / counts[good]) 
            # If a good was in z_consensus but no agent optimized for it, it remains (or could be zeroed)
            elif good not in self.z_consensus: # Ensure all relevant goods have an entry
                 self.z_consensus[good] = 0.0

        self.logger.debug(f"ADMM z_consensus aktualisiert für {len(self.z_consensus)} Güter.")


    def _admm_update_u(self, agents_local_solutions: Dict[str, Dict[str, float]]) -> None:
        updated_count = 0
        for agent_id, local_sol in agents_local_solutions.items():
             for good, x_val in local_sol.items():
                  if good in self.z_consensus: 
                       z_val = self.z_consensus[good]
                       self.u_duals[agent_id][good] += (x_val - z_val)
                       updated_count += 1
        self.logger.debug(f"ADMM u_duals aktualisiert ({updated_count} Einträge).")

    def _compute_admm_residuals_iteration(self, current_x: Dict[str, Dict[str,float]], prev_x: Dict[str, Dict[str,float]], rho: float) -> Tuple[float, float]:
        primal_res_sq, dual_res_sq = 0.0, 0.0
        num_primal_terms, num_dual_terms = 0, 0

        for agent_id, local_sol in current_x.items():
            for good, x_val in local_sol.items():
                 if good in self.z_consensus:
                      z_val = self.z_consensus[good]
                      primal_res_sq += (x_val - z_val)**2
                      num_primal_terms +=1
                      
                      # Dual residual based on change in local solutions (x_i)
                      x_prev_val = prev_x.get(agent_id, {}).get(good, x_val) # Use current x_val if no previous
                      dual_res_sq += (rho**2) * ((x_val - x_prev_val)**2)
                      num_dual_terms +=1
                      
        primal_residual = math.sqrt(primal_res_sq / num_primal_terms) if num_primal_terms > 0 else 0.0
        dual_residual = math.sqrt(dual_res_sq / num_dual_terms) if num_dual_terms > 0 else 0.0
        return primal_residual, dual_residual

    def _flag_for_plan_review(self, regions: Optional[List[Union[int, str]]], conflict_type: Optional[str]) -> None:
        self.logger.info(f"Markiere Plan zur Überprüfung aufgrund von Feedback: Regionen={regions}, Typ={conflict_type}")
        # This could set more specific flags for comprehensive_planning to use.
        self.planner._needs_parameter_review = True # Generic flag for now


# --- PlannerIntelligence Subsystem ---

class PlannerIntelligence:
    """
    Subsystem für Analyse, Prognose und Lernfähigkeit des System 4 Planners.
    Analysiert historische Daten, identifiziert Trends, bewertet Planungs-
    und ADMM-Performance und schlägt Anpassungen vor. Enthält Kausalinferenz
    und Szenario-basierte Planbewertung.
    """
    planner: 'System4Planner'
    model: 'EconomicModel'
    logger: logging.Logger
    analysis_history: Deque[Dict] 
    resource_forecast: Dict[str, Dict]

    # For Causal Inference
    planning_decision_history: Deque[Dict[str, Any]]
    system_outcome_history: Deque[Dict[str, Any]]
    inferred_causal_links: Dict[str, Dict[str, Any]] 
    causal_inference_history_length: int
    causal_inference_update_interval: int
    _last_causal_analysis_step: int

    # Scenario Planning Config (mirrored from S4 for direct access)
    _enable_scenario_planning_intel: bool
    _num_scenarios_to_simulate_intel: int
    _default_shock_severity_intel: float
    

    def __init__(self, planner: 'System4Planner'):
        self.planner = planner
        self.model = planner.model
        if hasattr(planner.logger, 'getChild'):
            self.logger = planner.logger.getChild('PlannerIntelligence')
        else: 
            self.logger = logging.getLogger(f"{getattr(planner.logger, 'name', 'System4')}.PlannerIntelligence")

        # General intelligence history (not causal-specific)
        self.analysis_history = deque(maxlen=planner.config.get("intelligence_general_history_length", 100))
        self.resource_forecast = {}

        # Causal Inference Attributes from S4's config (passed from model.config.system4_params)
        s4_config = self.planner.config # S4's config, which should have system4_params from main model config
        
        self.causal_inference_history_length = s4_config.get("causal_inference_history_length", 50)
        self.causal_inference_update_interval = s4_config.get("causal_inference_update_interval", 10) 
        
        self.planning_decision_history = deque(maxlen=self.causal_inference_history_length)
        self.system_outcome_history = deque(maxlen=self.causal_inference_history_length)
        self.inferred_causal_links = {} 
        self._last_causal_analysis_step = -1 

        # Scenario Planning Config from S4's config
        self._enable_scenario_planning_intel = s4_config.get("enable_scenario_planning", False)
        self._num_scenarios_to_simulate_intel = s4_config.get("num_scenarios_to_simulate_per_plan", 3)
        self._default_shock_severity_intel = s4_config.get("default_shock_severity_for_simulation", 0.1) # e.g. 10% shock

        self.logger.info(f"PlannerIntelligence initialized. Causal History: {self.causal_inference_history_length}, Causal Update Interval: {self.causal_inference_update_interval} S4 steps. Scenario Planning Intel Enabled: {self._enable_scenario_planning_intel}, Num Scenarios: {self._num_scenarios_to_simulate_intel}, Default Severity: {self._default_shock_severity_intel}")

    def record_planning_decision(self, decision_type: str, details: Dict, step: int) -> None:
        """Records a planning decision. `step` is the S4 operational step."""
        if not isinstance(details, dict):
            self.logger.error(f"Decision details must be a dict, got {type(details)}. Decision not recorded.")
            return
        record = {"s4_step": step, "type": decision_type, **details}
        self.planning_decision_history.append(record)
        self.logger.debug(f"Recorded S4 planning decision at S4 step {step}: {decision_type} - {details}")

    def record_system_outcome(self, outcome_data: Dict, step: int) -> None:
        """Records observed system outcomes. `step` is the S4 operational step when outcome is observed/collated."""
        if not isinstance(outcome_data, dict):
            self.logger.error(f"Outcome data must be a dict, got {type(outcome_data)}. Outcome not recorded.")
            return
        record = {"s4_step": step, **outcome_data}
        self.system_outcome_history.append(record)
        self.logger.debug(f"Recorded system outcome for causal history at S4 step {step}: {outcome_data}")

    def perform_causal_analysis(self) -> None:
        """
        Performs a basic causal analysis (correlation-based) on historical data.
        Called by `update` method based on `causal_inference_update_interval`.
        """
        s4_current_step = self.model.current_step 
        self.logger.info(f"Performing causal analysis (S4 step: {s4_current_step}). History sizes: Decisions={len(self.planning_decision_history)}, Outcomes={len(self.system_outcome_history)}")
        
        min_data_points_for_analysis = max(5, self.causal_inference_history_length // 5) # e.g. 10 for history of 50
        if len(self.planning_decision_history) < min_data_points_for_analysis or \
           len(self.system_outcome_history) < min_data_points_for_analysis:
            self.logger.info(f"Not enough historical data for causal analysis. Need at least {min_data_points_for_analysis} data points in both deques.")
            self._last_causal_analysis_step = s4_current_step 
            return

        # --- Example Correlation: R&D Investment vs. Tech Level Change ---
        correlation_data_cache = defaultdict(lambda: {'xs': [], 'ys': []}) # Cache for correlation calculation

        for decision_event in list(self.planning_decision_history):
            if decision_event.get('type') == 'investment_allocation':
                decision_s4_step = decision_event['s4_step']
                allocations = decision_event.get('allocations', {}) # Expecting nested dict here
                if isinstance(allocations, dict):
                    for sector_key, sector_allocs in allocations.items(): # e.g. sector_key = "sector_electronics"
                        if isinstance(sector_allocs, dict) and 'technology_research' in sector_allocs:
                            sector_name = sector_key.replace('sector_', '') # "electronics"
                            rnd_investment = sector_allocs['technology_research']
                            if rnd_investment <= 1e-6: continue # Skip negligible investments

                            for lag in range(1, 6): # Lags from 1 to 5 S4 steps
                                outcome_s4_step_to_match = decision_s4_step + lag
                                for outcome_event in list(self.system_outcome_history):
                                    if outcome_event['s4_step'] == outcome_s4_step_to_match:
                                        tech_change_metric_key = f"sector_{sector_name}_tech_level_change" # Hypothetical outcome key
                                        if tech_change_metric_key in outcome_event:
                                            tech_change = outcome_event[tech_change_metric_key]
                                            link_id = f"invest_rnd_{sector_name}_vs_tech_change_lag{lag}"
                                            correlation_data_cache[link_id]['xs'].append(rnd_investment)
                                            correlation_data_cache[link_id]['ys'].append(tech_change)
                                            break # Found outcome for this lag
        
        newly_identified_links = {}
        for link_id, data in correlation_data_cache.items():
            num_samples = len(data['xs'])
            if num_samples >= min_data_points_for_analysis: # Use the same threshold
                try:
                    correlation_matrix = np.corrcoef(data['xs'], data['ys'])
                    correlation_score = correlation_matrix[0, 1]
                    if not np.isnan(correlation_score) and abs(correlation_score) >= 0.35: # Correlation strength threshold
                        link_type = "positive" if correlation_score > 0 else "negative"
                        newly_identified_links[link_id] = {
                            "correlation": round(correlation_score, 3), 
                            "samples": num_samples,
                            "type": link_type,
                            "last_calculated_s4_step": s4_current_step
                        }
                        self.logger.info(f"Causal link identified/updated: {link_id} -> Corr: {correlation_score:.3f}, Samples: {num_samples}")
                except Exception as e:
                    self.logger.warning(f"Could not compute correlation for {link_id} (samples: {num_samples}): {e}")
        
        if newly_identified_links:
            self.inferred_causal_links.update(newly_identified_links)
            self.logger.info(f"Causal analysis complete. Total inferred links: {len(self.inferred_causal_links)}. New/Updated this run: {len(newly_identified_links)}")
        else:
            self.logger.info("Causal analysis did not yield new/strong correlations in this run.")
            
        self._last_causal_analysis_step = s4_current_step

    def get_causal_insights(self) -> Dict[str, Dict[str, Any]]:
        return self.inferred_causal_links.copy()

    def update(self) -> None:
        s4_current_step = self.model.current_step 
        self.logger.debug(f"PlannerIntelligence: Update cycle for S4 step {s4_current_step}.")

        std_intel_interval = self.planner.config.get("intelligence_update_interval", 5)
        if s4_current_step % std_intel_interval == 0:
            self.logger.debug(f"Running standard intelligence analysis at S4 step {s4_current_step}.")
            economic_data = self._collect_economic_data() 
            plan_effectiveness = self._analyze_plan_effectiveness(economic_data)
            self.resource_forecast = self._forecast_resource_usage(economic_data)
            admm_performance = self._analyze_admm_performance()
            recommendations = self._generate_recommendations(plan_effectiveness, self.resource_forecast, admm_performance)
            self.analysis_history.append({
                "s4_step": s4_current_step, "economic_data": economic_data,
                "plan_effectiveness": plan_effectiveness, "resource_forecast": self.resource_forecast,
                "admm_performance": admm_performance, "recommendations": recommendations
            })
            self._apply_recommendations(recommendations) 
            self.logger.info(f"PlannerIntelligence: Standard analysis update complete for S4 step {s4_current_step}.")

        if s4_current_step >= self._last_causal_analysis_step + self.causal_inference_update_interval:
            self.perform_causal_analysis()

    # --- Scenario-Based Planning Methods ---

    def _get_simplified_economic_state(self) -> Dict[str, Any]:
        """Creates a simplified snapshot of the economy for 'what-if' simulation."""
        state = {
            "producer_capacities_per_good": defaultdict(float), 
            "resource_availability": defaultdict(float),
            "resource_input_needs_per_good_unit": defaultdict(lambda: defaultdict(float)) 
        }
        s4_current_step = self.model.current_step

        for producer_agent in self.model.producers: # type: ignore
            if getattr(producer_agent, 'bankrupt', False):
                continue
            # Attempt to get per-good capacity first
            current_caps = getattr(producer_agent, "current_max_capacity_per_good", {}) # Ideal: Producer calculates this based on its state
            if not current_caps and hasattr(producer_agent, 'productive_capacity') and hasattr(producer_agent, 'can_produce'):
                # Fallback: distribute general capacity among producible goods if specific not available
                gen_cap = getattr(producer_agent, 'productive_capacity', 0)
                can_produce_list = getattr(producer_agent, 'can_produce', [])
                if can_produce_list: # Avoid division by zero
                    cap_per_good = gen_cap / len(can_produce_list)
                    current_caps = {good: cap_per_good for good in can_produce_list}
            
            for good_name, capacity_amount in current_caps.items():
                if good_name in self.planner.plan_relevant_goods:
                    state["producer_capacities_per_good"][good_name] += capacity_amount
        
        self.logger.debug(f"[ScenarioSim S4Step {s4_current_step}] Simplified state: Initial capacities for relevant goods: {dict(state['producer_capacities_per_good'])}")

        default_resources = {"energy": 1000000.0, "water": 5000000.0, "universal_input": 200000.0} # Large fallback values
        # Try to get actual resource levels from the model (e.g. from a global stockpile or S1/S2 aggregated data)
        # This part is highly dependent on how the overall model tracks global resource levels.
        # For now, we assume it *might* exist on `self.model.global_resource_stockpiles`.
        model_stockpiles = getattr(self.model, 'global_resource_stockpiles', None)
        if model_stockpiles is not None and isinstance(model_stockpiles, dict):
            state["resource_availability"] = defaultdict(float, model_stockpiles.copy())
            self.logger.debug(f"[ScenarioSim S4Step {s4_current_step}] Using resource availability from model.global_resource_stockpiles: {dict(state['resource_availability'])}")
        else:
            state["resource_availability"] = defaultdict(float, default_resources)
            self.logger.warning(f"[ScenarioSim S4Step {s4_current_step}] Using DUMMY resource availability as model.global_resource_stockpiles not found or not a dict: {dict(state['resource_availability'])}")

        # Simplified IO coefficients (Resource input per unit of good output)
        # These should ideally be derived from current average technology levels in the economy.
        for good_name in self.planner.plan_relevant_goods:
            # Example: these would be based on analysis of producer recipes/technologies
            if "food" in good_name: 
                state["resource_input_needs_per_good_unit"][good_name]["water"] = 0.5 
                state["resource_input_needs_per_good_unit"][good_name]["energy"] = 0.1
            elif "metal" in good_name: 
                state["resource_input_needs_per_good_unit"][good_name]["energy"] = 2.0
                state["resource_input_needs_per_good_unit"][good_name]["universal_input"] = 0.5 # e.g. ore
            elif "electronics" in good_name:
                state["resource_input_needs_per_good_unit"][good_name]["energy"] = 0.5
                state["resource_input_needs_per_good_unit"][good_name]["rare_metals"] = 0.05 # Example specific input
            else: # Generic fallback for other goods
                state["resource_input_needs_per_good_unit"][good_name]["energy"] = 0.2
                state["resource_input_needs_per_good_unit"][good_name]["universal_input"] = 0.1
        self.logger.debug(f"[ScenarioSim S4Step {s4_current_step}] Using conceptual IO coefficients for resource needs.")
        return state

    def _apply_shock_to_state(self, original_state: Dict[str, Any], shock_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Applies a given shock to a copy of the simplified economic state."""
        shocked_state = {
            "producer_capacities_per_good": original_state["producer_capacities_per_good"].copy(),
            "resource_availability": original_state["resource_availability"].copy(),
            "resource_input_needs_per_good_unit": original_state["resource_input_needs_per_good_unit"] # Assuming IOs don't change with these shocks
        }
        shock_type = shock_scenario.get("type")
        s4_current_step = self.model.current_step # For logging
        self.logger.info(f"  [ScenarioSim S4Step {s4_current_step}] Applying shock: {shock_scenario}")

        if shock_type == "resource_availability_shock":
            resource = shock_scenario.get("resource")
            reduction_percentage = shock_scenario.get("reduction_percentage", 0.0)
            if resource in shocked_state["resource_availability"]:
                original_level = shocked_state["resource_availability"][resource]
                shocked_state["resource_availability"][resource] *= (1.0 - reduction_percentage)
                self.logger.debug(f"    Resource '{resource}' availability reduced by {reduction_percentage*100:.1f}% from {original_level:.2f} to {shocked_state['resource_availability'][resource]:.2f}.")
            else:
                self.logger.warning(f"    Attempted resource shock for '{resource}', but it's not in the known resource availability list: {list(shocked_state['resource_availability'].keys())}")
        
        elif shock_type == "capacity_reduction_shock": # Note: CrisisManager generates "capacity_reduction_shock"
            good_affected = shock_scenario.get("good_category") # Assuming CrisisManager uses "good_category"
            if good_affected is None: # Fallback for direct use or different key
                good_affected = shock_scenario.get("good_affected")

            reduction_percentage = shock_scenario.get("reduction_percentage", 0.0)
            
            if good_affected: # If a good category or specific good is identified
                 # This simplified simulation applies category shock to all relevant plan goods if not specific enough.
                 # A more complex model would map good_category to specific goods.
                 # For now, if good_affected is a category, we might apply it broadly or expect System4 to refine.
                 # Here, we assume good_affected from scenario maps directly to a plan_relevant_good for simplicity in simulation.
                if good_affected in shocked_state["producer_capacities_per_good"]:
                    original_capacity = shocked_state["producer_capacities_per_good"][good_affected]
                    shocked_state["producer_capacities_per_good"][good_affected] *= (1.0 - reduction_percentage)
                    self.logger.debug(f"    Capacity for good '{good_affected}' reduced by {reduction_percentage*100:.1f}% from {original_capacity:.2f} to {shocked_state['producer_capacities_per_good'][good_affected]:.2f}.")
                else: # If the good_category does not directly map to a single aggregated good capacity
                    self.logger.warning(f"    Capacity shock for category '{good_affected}'. This simulation applies it if it's a specific good in plan_relevant_goods. For broad category shocks, PlannerIntelligence would need to map category to specific goods.")
                    # Example: Apply to all goods if 'good_category' was 'all_consumer_goods' (conceptual)
                    # This part needs more refined logic if good_category is not directly a plan_relevant_good.
                    # For now, we only act if good_affected is a specific good in our capacities list.

            else:
                self.logger.warning(f"    Capacity shock scenario did not specify 'good_category' or 'good_affected'. Shock not applied effectively.")


        elif shock_type == "demand_shock": # This type of shock would typically alter the `candidate_plan_targets`
            self.logger.warning(f"    Demand shock type received ({shock_scenario}), but this simulation method primarily evaluates state vs. plan. Demand shocks should ideally adjust plan targets *before* this simulation.")
        
        else:
            self.logger.warning(f"    Unhandled shock type in _apply_shock_to_state: {shock_type}")
            
        return shocked_state

    def _evaluate_plan_on_shocked_state(self, candidate_plan_targets: Dict[str, float], shocked_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates how well a plan (production targets) can be met given a shocked (simplified) state."""
        projected_fulfillment_rates = {} 
        projected_production_levels = {} 
        resource_deficits_percentage = defaultdict(float) 
        s4_current_step = self.model.current_step # For logging
        
        # 1. Calculate total resource demand for the candidate plan based on its production targets and IO coefficients
        total_resource_demand_for_plan = defaultdict(float)
        for good, planned_amount in candidate_plan_targets.items():
            if planned_amount <= 1e-6: continue # Ignore negligible targets
            # Get the IO needs for this good (e.g. {energy:0.2, universal_input:0.1})
            resource_needs_per_unit = shocked_state["resource_input_needs_per_good_unit"].get(good, {})
            for resource, unit_need_amount in resource_needs_per_unit.items():
                total_resource_demand_for_plan[resource] += planned_amount * unit_need_amount

        # 2. Determine resource availability factors (how much of the demand for each resource can be met)
        resource_availability_factors = defaultdict(lambda: 1.0) # Default: 100% available
        for resource, total_demanded in total_resource_demand_for_plan.items():
            available = shocked_state["resource_availability"].get(resource, 0.0)
            if total_demanded > available:
                # Factor is ratio of available to demanded
                resource_availability_factors[resource] = available / total_demanded if total_demanded > 0 else 0.0
                # Deficit is calculated as (demanded - available) / demanded
                deficit_pct = (total_demanded - available) / total_demanded if total_demanded > 0 else 1.0 # 100% deficit if demanded > 0 and available = 0
                resource_deficits_percentage[resource] = deficit_pct
                self.logger.warning(f"    [ScenarioSimEval S4Step {s4_current_step}] Resource '{resource}': Demanded={total_demanded:.2f}, Available={available:.2f}, AvailabilityFactor={resource_availability_factors[resource]:.2f}, Deficit%={deficit_pct:.2%}")
        
        # 3. Calculate achievable production for each good based on its capacity and the availability of its specific input resources
        for good, planned_amount in candidate_plan_targets.items():
            if planned_amount <= 1e-6: # If plan target is zero or tiny
                projected_production_levels[good] = 0.0
                projected_fulfillment_rates[good] = 1.0 # No target, so 100% fulfilled
                continue

            capacity_limit_for_good = shocked_state["producer_capacities_per_good"].get(good, 0.0)
            
            # Determine the most limiting resource factor for *this specific good*
            # (i.e., if producing good X needs energy and water, which of their availability_factors is lower?)
            good_specific_resource_limitation_factor = 1.0
            resource_needs_for_this_good = shocked_state["resource_input_needs_per_good_unit"].get(good, {})
            if not resource_needs_for_this_good: # If this good has no defined resource inputs in our simple IO model
                 pass # Assumes it needs no limited resources from the list
            else:
                for resource_input in resource_needs_for_this_good:
                    good_specific_resource_limitation_factor = min(good_specific_resource_limitation_factor, 
                                                                   resource_availability_factors[resource_input]) # Use the pre-calculated general factor for that resource

            # Achievable production is limited by its own capacity and by the availability of its inputs
            achievable_production = min(capacity_limit_for_good, planned_amount * good_specific_resource_limitation_factor)
            projected_production_levels[good] = achievable_production
            projected_fulfillment_rates[good] = achievable_production / planned_amount if planned_amount > 0 else 1.0
            
            if projected_fulfillment_rates[good] < 0.99: # Log if not (almost) fully met
                self.logger.debug(f"    [ScenarioSimEval S4Step {s4_current_step}] Good '{good}': Target={planned_amount:.2f}, CapacityLimit={capacity_limit_for_good:.2f}, ResourceLimitationFactor={good_specific_resource_limitation_factor:.2f}, Achievable={achievable_production:.2f}, Fulfillment={projected_fulfillment_rates[good]:.2f}")
        
        return {
            "projected_fulfillment_rates": projected_fulfillment_rates, 
            "projected_production_levels": projected_production_levels,
            "projected_resource_deficits_percentage": dict(resource_deficits_percentage) # Convert defaultdict to dict for output
        }


    def simulate_plan_under_shocks(self, candidate_plan_targets: Dict[str, float], potential_shocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulates a plan against a list of shocks and returns robustness metrics."""
        s4_current_step = self.model.current_step # For logging with consistent step number
        if not self._enable_scenario_planning_intel: 
            self.logger.info(f"[S4 Step {s4_current_step}] Scenario-based planning is disabled in PlannerIntelligence config.")
            return {"s4_step": s4_current_step, "status": "disabled", "reason": "disabled_in_planner_intelligence_config"}

        self.logger.info(f"[S4 Step {s4_current_step}] Simulating plan robustness for {len(candidate_plan_targets)} goods against {len(potential_shocks)} shock scenarios.")
        
        original_simplified_state = self._get_simplified_economic_state() # Get current baseline state once
        all_scenario_outcomes_details = [] # To store detailed results for each shock

        for i, shock_scenario in enumerate(potential_shocks):
            self.logger.debug(f"  [S4 Step {s4_current_step}] Simulating scenario {i+1}/{len(potential_shocks)} with shock: {shock_scenario.get('type')}")
            shocked_state = self._apply_shock_to_state(original_simplified_state, shock_scenario)
            single_scenario_evaluation = self._evaluate_plan_on_shocked_state(candidate_plan_targets, shocked_state)
            all_scenario_outcomes_details.append({
                "shock_applied": shock_scenario, 
                "evaluation": single_scenario_evaluation # Contains fulfillment, production, deficits for this scenario
            })
            
        # Aggregate results for a summary report
        overall_fulfillment_rates_across_goods_and_scenarios = []
        resource_deficits_by_scenario = [] # Collect all deficit dicts from scenarios where deficits occurred
        
        for scenario_result in all_scenario_outcomes_details:
            # Collect all fulfillment rates from this scenario
            for good_fulfillment_rate in scenario_result["evaluation"]["projected_fulfillment_rates"].values():
                overall_fulfillment_rates_across_goods_and_scenarios.append(good_fulfillment_rate)
            # Collect deficit information if any occurred in this scenario
            if scenario_result["evaluation"]["projected_resource_deficits_percentage"]:
                resource_deficits_by_scenario.append(scenario_result["evaluation"]["projected_resource_deficits_percentage"])

        avg_fulfillment = np.mean(overall_fulfillment_rates_across_goods_and_scenarios) if overall_fulfillment_rates_across_goods_and_scenarios else 1.0 # Default to 1.0 if no rates (e.g. empty plan)
        
        # Summarize worst-case deficits across all scenarios for each resource type
        worst_case_overall_resource_deficits = defaultdict(float) # Store max deficit for each resource
        for deficit_dict_for_scenario in resource_deficits_by_scenario:
            for resource, deficit_rate in deficit_dict_for_scenario.items():
                if deficit_rate > worst_case_overall_resource_deficits[resource]: 
                    worst_case_overall_resource_deficits[resource] = deficit_rate
        
        summary_report = {
            "s4_step": s4_current_step,
            "num_shocks_simulated": len(potential_shocks),
            "average_projected_fulfillment_all_scenarios": round(avg_fulfillment, 3),
            "worst_case_resource_deficit_summary_percentage": dict(worst_case_overall_resource_deficits), # Convert to dict for output
            "detailed_scenario_outcomes": all_scenario_outcomes_details # For deeper analysis by S4 if needed
        }
        self.logger.info(f"[S4 Step {s4_current_step}] Robustness simulation summary: Avg Fulfillment={avg_fulfillment:.3f}, Worst-Case Resource Deficits (%)={dict(worst_case_overall_resource_deficits)}")
        return summary_report

    def evaluate_candidate_plan_robustness(self, candidate_plan_targets: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluates a candidate plan's robustness by simulating it under potential crisis scenarios.
        This is intended to be called by System4Planner.
        """
        s4_current_step = self.model.current_step # For logging
        if not self._enable_scenario_planning_intel: 
            return {"s4_step": s4_current_step, "status": "disabled", "reason": "Scenario planning not enabled in PlannerIntelligence."}

        self.logger.info(f"[S4 Step {s4_current_step}] Evaluating robustness for candidate plan (targets for {len(candidate_plan_targets)} goods).")
        
        potential_shocks: List[Dict[str, Any]] = [] # Ensure type, will hold list of shock dicts
        
        # Attempt to use CrisisManager if available on the model
        crisis_manager = getattr(self.model, 'crisis_manager', None)
        if crisis_manager and hasattr(crisis_manager, 'get_potential_shock_scenarios'):
            try:
                # Assume CrisisManager.get_potential_shock_scenarios exists and returns List[Dict]
                shocks_from_cm = crisis_manager.get_potential_shock_scenarios( # type: ignore 
                    num_scenarios=self._num_scenarios_to_simulate_intel,
                    severity_level=self._default_shock_severity_intel
                )
                if shocks_from_cm is not None and isinstance(shocks_from_cm, list): # Basic validation
                    potential_shocks = shocks_from_cm
                self.logger.info(f"[S4 Step {s4_current_step}] Retrieved {len(potential_shocks)} shock scenarios from CrisisManager.")
            except Exception as e:
                self.logger.error(f"[S4 Step {s4_current_step}] Error retrieving shocks from CrisisManager: {e}. Will attempt to use fallback scenarios.")
                potential_shocks = [] # Ensure fallback is used if CM fails or returns invalid data
        else:
            self.logger.warning(f"[S4 Step {s4_current_step}] CrisisManager or 'get_potential_shock_scenarios' method not found on model. Using fallback scenarios.")

        if not potential_shocks: # If CrisisManager didn't provide scenarios or isn't available
            self.logger.warning(f"[S4 Step {s4_current_step}] Using generic fallback shock scenarios.")
            relevant_goods = self.planner.plan_relevant_goods # Goods S4 is concerned with
            fallback_shocks_list: List[Dict[str, Any]] = [] # Explicitly typed list
            
            # Fallback Shock 1: Capacity shock on a random relevant good
            if relevant_goods: # Ensure there's at least one good to pick
                chosen_good1 = random.choice(relevant_goods)
                # CrisisManager now uses "good_category" or "resource", so we adapt.
                # For capacity shock, we'll use "good_affected" as the key for our simulation logic.
                fallback_shocks_list.append(
                    {"type": "capacity_reduction_shock", "good_affected": chosen_good1, # Or "good_category" if CM sends that
                     "reduction_percentage": self._default_shock_severity_intel * 1.2} # Slightly more severe
                )
                # Fallback Shock 2: Capacity shock on a *different* random relevant good (if possible)
                if len(relevant_goods) > 1:
                    possible_goods2 = [g for g in relevant_goods if g != chosen_good1]
                    if possible_goods2: # If there is another distinct good
                         chosen_good2 = random.choice(possible_goods2)
                         fallback_shocks_list.append(
                            {"type": "capacity_reduction_shock", "good_affected": chosen_good2,
                             "reduction_percentage": self._default_shock_severity_intel * 0.8} # Slightly less severe
                        )
            
            # Fallback Shock 3: Resource availability shock for 'energy'
            fallback_shocks_list.append(
                {"type": "resource_availability_shock", "resource": "energy", 
                 "reduction_percentage": self._default_shock_severity_intel}
            )
            # Add more fallback shocks if needed, up to _num_scenarios_to_simulate_intel
            if len(fallback_shocks_list) < self._num_scenarios_to_simulate_intel:
                 fallback_shocks_list.append(
                    {"type": "resource_availability_shock", "resource": "water", 
                    "reduction_percentage": self._default_shock_severity_intel * 0.5} # Less severe water shock
                )

            potential_shocks = fallback_shocks_list[:self._num_scenarios_to_simulate_intel] # Ensure correct number of scenarios
            self.logger.info(f"[S4 Step {s4_current_step}] Generated {len(potential_shocks)} fallback scenarios: {potential_shocks}")


        if not potential_shocks: # Still no shocks even after fallback logic (e.g., if no relevant_goods for capacity shock)
            self.logger.error(f"[S4 Step {s4_current_step}] No shock scenarios to simulate (neither from CrisisManager nor fallback). Plan robustness cannot be evaluated.")
            return {"s4_step": s4_current_step, "status": "no_shocks_available", "reason": "No shock scenarios available for simulation (CrisisManager and fallback)."}
            
        return self.simulate_plan_under_shocks(candidate_plan_targets, potential_shocks)

    # --- End Scenario-Based Planning Methods ---

    def _collect_economic_data(self) -> Dict:
        data = {"s4_step": self.model.current_step}
        gov_metrics = getattr(self.model, 'detailed_metrics', {})
        gov_welfare_metrics = gov_metrics.get('welfare', {})
        if 'satisfaction_gini' in gov_welfare_metrics:
            data['satisfaction_gini'] = gov_welfare_metrics['satisfaction_gini']
        data['resource_utilization_dummy'] = {"energy": np.random.rand() * 0.4 + 0.5} 
        self.logger.debug(f"Collected general economic data for intelligence: {list(data.keys())}")
        return data

    def _analyze_plan_effectiveness(self, economic_data: Dict) -> Dict:
        accuracy = np.random.rand() * 0.2 + 0.75 
        trend = random.choice(["stable", "improving", "worsening"])
        effectiveness_summary = {"avg_accuracy_dummy": accuracy, "trend_dummy": trend}
        if 'satisfaction_gini' in economic_data: 
            effectiveness_summary['current_gini_from_gov'] = economic_data['satisfaction_gini']
        self.logger.debug(f"Plan effectiveness analysis (dummy): {effectiveness_summary}")
        return effectiveness_summary

    def _forecast_resource_usage(self, economic_data: Dict) -> Dict:
        forecast = {}
        dummy_resource_util = economic_data.get('resource_utilization_dummy', {})
        if 'energy' in dummy_resource_util:
            current_energy_usage_norm = dummy_resource_util['energy']
            assumed_total_capacity_energy = 2000 
            current_energy_usage_abs = current_energy_usage_norm * assumed_total_capacity_energy
            forecast_val_abs = current_energy_usage_abs * (1 + (np.random.rand() - 0.45) * 0.15)
            forecast["energy_dummy"] = {
                "current_usage_norm": current_energy_usage_norm, "current_usage_abs_est": current_energy_usage_abs,
                "forecast_3_steps_abs_est": forecast_val_abs, "limit_abs_est": assumed_total_capacity_energy 
            }
        self.logger.debug(f"Resource usage forecast (dummy): {forecast}")
        return forecast

    def _analyze_admm_performance(self) -> Optional[Dict]:
        if not self.planner.admm_on or not self.planner.admm_convergence_history:
             self.logger.debug("ADMM not active or no history, skipping ADMM performance analysis.")
             return None
        recent_runs = self.planner.admm_convergence_history[-min(5, len(self.planner.admm_convergence_history)):]
        if not recent_runs: return None
        avg_iterations = np.mean([run.get('iterations', self.planner.admm_config.max_iterations) for run in recent_runs])
        avg_primal_res = np.mean([run.get('final_primal_res', np.nan) for run in recent_runs])
        admm_summary = {"avg_iterations": round(avg_iterations,1), "avg_primal_residual": round(avg_primal_res,4) if not np.isnan(avg_primal_res) else None}
        self.logger.debug(f"ADMM performance analysis: {admm_summary}")
        return admm_summary

    def _generate_recommendations(self, plan_eff: Dict, res_forecast: Dict, admm_perf: Optional[Dict]) -> Dict:
        recs = {}
        energy_forecast_data = res_forecast.get("energy_dummy", {}) 
        if energy_forecast_data.get("forecast_3_steps_abs_est", 0) > energy_forecast_data.get("limit_abs_est", float('inf')) * 0.9:
             recs["increase_priority_for_good"] = "energy"
             self.logger.info("Recommendation: Increase planning priority for 'energy' (dummy) due to high forecast.")
        if plan_eff.get("avg_accuracy_dummy", 1.0) < 0.8: 
             recs["review_planning_parameters"] = True
             self.logger.info(f"Recommendation: Review planning parameters due to low dummy accuracy ({plan_eff.get('avg_accuracy_dummy'):.2f}).")
        if admm_perf and admm_perf.get("avg_iterations", 0) > self.planner.admm_config.max_iterations * 0.75: 
            recs["adjust_admm_rho"] = "increase" 
            self.logger.info("Recommendation: Adjust ADMM rho (e.g. increase) due to many iterations.")
        self.logger.debug(f"Generated recommendations: {recs}")
        return recs

    def _apply_recommendations(self, recommendations: Dict) -> None:
        if not recommendations:
            self.logger.debug("No recommendations to apply/signal.")
            return
        self.logger.info(f"Signaling recommendations to System4Planner: {recommendations}")
        # Reset flags before applying new ones
        self.planner._recommended_priority_increase_good = None
        self.planner._needs_parameter_review = False
        self.planner._needs_admm_rho_adjustment = None

        if "increase_priority_for_good" in recommendations:
            self.planner._recommended_priority_increase_good = recommendations["increase_priority_for_good"]
        if "review_planning_parameters" in recommendations:
            self.planner._needs_parameter_review = True
        if "adjust_admm_rho" in recommendations:
            self.planner._needs_admm_rho_adjustment = recommendations["adjust_admm_rho"]

    def get_intelligence_report(self) -> Dict[str, Any]:
         latest_analysis = self.analysis_history[-1] if self.analysis_history else {}
         s4_current_step = self.model.current_step
         report = {
             "s4_step": s4_current_step, "resource_forecast": self.resource_forecast, 
             "plan_effectiveness_summary": latest_analysis.get("plan_effectiveness"),
             "admm_performance_summary": latest_analysis.get("admm_performance"),
             "recommendations_from_std_analysis": latest_analysis.get("recommendations"),
             "causal_insights": self.get_causal_insights() 
         }
         self.logger.debug(f"Generated intelligence report for S4 step {s4_current_step}: {list(report.keys())}")
         return report
