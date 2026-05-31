
Created on Wed Mar 11 21:23:46 2026

@author: melike
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np
import pandas as pd
import simpy


# ==========================================================
# ADVANCED WSN-SUPPORTED HOSPITAL CONTACT TRACING SIMULATOR
# ==========================================================
# Added advanced features:
# 1) 2D hospital floor map
# 2) Agent movement animation
# 3) BLE RSSI + wall attenuation model
# 4) Numpy-only KNN risk prediction
# 5) Dashboard-style analytics figure
#
# Designed for academic project use in Spyder / standard Python.
# ==========================================================


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Zone:
    name: str
    x: float
    y: float
    width: float
    height: float
    risk_multiplier: float
    anchor_count: int = 1

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class PersonType:
    name: str
    speed_min: float
    speed_max: float
    contact_radius: float
    dwell_min: int
    dwell_max: int
    transmit_interval: int
    battery_capacity_mAh: float
    role_risk_multiplier: float


@dataclass
class Person:
    pid: int
    ptype: PersonType
    current_zone: str
    x: float
    y: float
    infected: bool = False
    infectious: bool = False
    recovered: bool = False
    masked: bool = True
    isolated: bool = False
    symptom_onset_time: Optional[int] = None
    infection_time: Optional[int] = None
    battery_remaining_mAh: float = 0.0
    last_packet_time: int = 0
    movement_target: Optional[Tuple[float, float]] = None
    dwell_until: int = 0
    infection_source: Optional[int] = None
    history: List[Tuple[int, str, float, float, bool]] = field(default_factory=list)

    def __post_init__(self):
        self.battery_remaining_mAh = self.ptype.battery_capacity_mAh


@dataclass
class ScenarioConfig:
    simulation_minutes: int = 8 * 60
    seed: int = 42
    initial_infected: int = 2
    infection_probability_base: float = 0.05
    contact_time_threshold: int = 3
    distance_threshold_m: float = 2.0
    incubation_minutes: int = 60
    infectious_delay_minutes: int = 10
    recovery_minutes: int = 999999
    packet_tx_energy_mAh: float = 0.0008
    scan_energy_mAh: float = 0.00025
    idle_energy_mAh_per_min: float = 0.0001
    movement_step_minutes: int = 1
    enable_isolation: bool = True
    isolation_trigger_after_symptoms: int = 20
    visitor_restriction: bool = False
    mask_effectiveness_tx: float = 0.45
    mask_effectiveness_rx: float = 0.35
    ble_tx_power_dbm: float = -59.0
    path_loss_exponent: float = 2.1
    rssi_noise_std: float = 2.2
    wall_loss_db: float = 7.0
    ml_k_neighbors: int = 5


# -----------------------------
# Core simulator
# -----------------------------

class HospitalWSNSimulator:
    def __init__(self, config: ScenarioConfig):
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.env = simpy.Environment()

        self.zones: Dict[str, Zone] = self._build_zones()
        self.zone_connections = self._build_zone_connections()
        self.person_types: Dict[str, PersonType] = self._build_person_types()
        self.persons: Dict[int, Person] = {}

        self.contact_log: List[Dict] = []
        self.packet_log: List[Dict] = []
        self.infection_log: List[Dict] = []
        self.energy_log: List[Dict] = []
        self.presence_log: List[Dict] = []
        self.ml_dataset_cache: Optional[pd.DataFrame] = None

        self.active_contacts: Dict[Tuple[int, int], Dict] = {}
        self.infection_graph = nx.DiGraph()

        self._create_population()
        self._seed_infections()

    # -------------------------
    # Setup helpers
    # -------------------------

    def _build_zones(self) -> Dict[str, Zone]:
        return {
            "ER": Zone("ER", 0, 10, 12, 10, risk_multiplier=1.4, anchor_count=2),
            "ICU": Zone("ICU", 14, 10, 10, 8, risk_multiplier=1.7, anchor_count=2),
            "Ward_A": Zone("Ward_A", 0, 0, 12, 8, risk_multiplier=1.2, anchor_count=2),
            "Ward_B": Zone("Ward_B", 14, 0, 12, 8, risk_multiplier=1.2, anchor_count=2),
            "Corridor": Zone("Corridor", 0, 8.5, 26, 1.2, risk_multiplier=1.0, anchor_count=3),
            "Lab": Zone("Lab", 28, 10, 8, 6, risk_multiplier=1.1, anchor_count=1),
            "Waiting": Zone("Waiting", 28, 0, 10, 8, risk_multiplier=1.3, anchor_count=2),
            "StaffRoom": Zone("StaffRoom", 38.5, 10, 8, 6, risk_multiplier=0.8, anchor_count=1),
        }

    def _build_zone_connections(self) -> Dict[str, List[str]]:
        return {
            "ER": ["Corridor"],
            "ICU": ["Corridor"],
            "Ward_A": ["Corridor"],
            "Ward_B": ["Corridor"],
            "Lab": ["Corridor", "StaffRoom"],
            "Waiting": ["Corridor"],
            "StaffRoom": ["Lab", "Corridor"],
            "Corridor": ["ER", "ICU", "Ward_A", "Ward_B", "Lab", "Waiting", "StaffRoom"],
        }

    def _build_person_types(self) -> Dict[str, PersonType]:
        return {
            "patient": PersonType("patient", 0.18, 0.45, 2.0, 8, 30, 2, 220, 1.20),
            "doctor": PersonType("doctor", 0.75, 1.35, 2.0, 2, 10, 1, 260, 1.10),
            "nurse": PersonType("nurse", 0.70, 1.25, 2.0, 2, 12, 1, 260, 1.15),
            "visitor": PersonType("visitor", 0.45, 0.90, 2.0, 4, 20, 2, 200, 1.00),
            "staff": PersonType("staff", 0.50, 1.10, 2.0, 3, 15, 2, 230, 1.00),
        }

    def _random_point_in_zone(self, zone_name: str) -> Tuple[float, float]:
        z = self.zones[zone_name]
        margin = 0.4
        return (
            random.uniform(z.x + margin, z.x + z.width - margin),
            random.uniform(z.y + margin, z.y + z.height - margin),
        )

    def _create_population(self):
        pid = 0
        population_plan = {
            "patient": 16,
            "doctor": 5,
            "nurse": 8,
            "visitor": 6 if not self.config.visitor_restriction else 2,
            "staff": 4,
        }
        zone_bias = {
            "patient": ["Ward_A", "Ward_B", "ER", "ICU", "Waiting"],
            "doctor": ["ER", "ICU", "Ward_A", "Ward_B", "Lab", "StaffRoom", "Corridor"],
            "nurse": ["ER", "ICU", "Ward_A", "Ward_B", "Corridor", "StaffRoom"],
            "visitor": ["Waiting", "Ward_A", "Ward_B", "Corridor"],
            "staff": ["Lab", "Corridor", "StaffRoom", "Waiting"],
        }

        for role, count in population_plan.items():
            for _ in range(count):
                zone = random.choice(zone_bias[role])
                x, y = self._random_point_in_zone(zone)
                masked = random.random() < 0.82
                person = Person(
                    pid=pid,
                    ptype=self.person_types[role],
                    current_zone=zone,
                    x=x,
                    y=y,
                    masked=masked,
                )
                self.persons[pid] = person
                self.infection_graph.add_node(pid, role=role)
                pid += 1

    def _seed_infections(self):
        infected_ids = random.sample(list(self.persons.keys()), self.config.initial_infected)
        for pid in infected_ids:
            p = self.persons[pid]
            p.infected = True
            p.infectious = True
            p.infection_time = 0
            p.symptom_onset_time = self.config.incubation_minutes
            self.infection_log.append({
                "time": 0,
                "source": None,
                "target": pid,
                "zone": p.current_zone,
                "event": "seed_infection",
                "probability": np.nan,
            })

    # -------------------------
    # Movement model
    # -------------------------

    def _allowed_zones(self, person: Person) -> List[str]:
        role = person.ptype.name
        if person.isolated:
            return [person.current_zone]
        mapping = {
            "patient": ["Ward_A", "Ward_B", "ER", "ICU", "Lab", "Corridor", "Waiting"],
            "doctor": list(self.zones.keys()),
            "nurse": ["ER", "ICU", "Ward_A", "Ward_B", "Corridor", "StaffRoom"],
            "visitor": ["Waiting", "Ward_A", "Ward_B", "Corridor"],
            "staff": ["Lab", "Corridor", "StaffRoom", "Waiting", "ER"],
        }
        return mapping[role]

    def _select_next_zone(self, person: Person) -> str:
        allowed = self._allowed_zones(person)
        current = person.current_zone
        connected = [z for z in self.zone_connections.get(current, []) if z in allowed]
        if random.random() < 0.45:
            return current
        if connected:
            return random.choice(connected)
        return random.choice(allowed)

    def _move_person(self, person: Person):
        if person.isolated:
            person.history.append((self.env.now, person.current_zone, person.x, person.y, person.infected))
            return

        if self.env.now >= person.dwell_until or person.movement_target is None:
            next_zone = self._select_next_zone(person)
            if next_zone != person.current_zone:
                person.current_zone = next_zone
                person.x, person.y = self._random_point_in_zone(next_zone)
                person.movement_target = self._random_point_in_zone(next_zone)
            else:
                person.movement_target = self._random_point_in_zone(person.current_zone)
            person.dwell_until = self.env.now + random.randint(person.ptype.dwell_min, person.ptype.dwell_max)

        tx, ty = person.movement_target
        dx = tx - person.x
        dy = ty - person.y
        dist = math.hypot(dx, dy)
        speed = random.uniform(person.ptype.speed_min, person.ptype.speed_max)
        step = speed * self.config.movement_step_minutes

        if dist <= step or dist == 0:
            person.x, person.y = tx, ty
            person.movement_target = self._random_point_in_zone(person.current_zone)
        else:
            person.x += dx / dist * step
            person.y += dy / dist * step

        person.history.append((self.env.now, person.current_zone, person.x, person.y, person.infected))
        self.presence_log.append({
            "time": self.env.now,
            "pid": person.pid,
            "role": person.ptype.name,
            "zone": person.current_zone,
            "x": person.x,
            "y": person.y,
            "infected": person.infected,
            "isolated": person.isolated,
        })

    # -------------------------
    # BLE / RSSI model
    # -------------------------

    def _distance(self, a: Person, b: Person) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _wall_count_between(self, zone_a: str, zone_b: str) -> int:
        if zone_a == zone_b:
            return 0
        if zone_b in self.zone_connections.get(zone_a, []):
            return 1
        return 2

    def _estimate_rssi(self, a: Person, b: Person) -> float:
        d = max(0.2, self._distance(a, b))
        tx_power = self.config.ble_tx_power_dbm
        n = self.config.path_loss_exponent
        walls = self._wall_count_between(a.current_zone, b.current_zone)
        wall_loss = walls * self.config.wall_loss_db
        rssi = tx_power - 10 * n * math.log10(d) - wall_loss + np.random.normal(0, self.config.rssi_noise_std)
        return rssi

    def _estimate_distance_from_rssi(self, rssi: float) -> float:
        tx_power = self.config.ble_tx_power_dbm
        n = self.config.path_loss_exponent
        return 10 ** ((tx_power - rssi) / (10 * n))

    def _consume_energy(self, person: Person, tx_packets: int = 0, scans: int = 0, idle_minutes: int = 1):
        person.battery_remaining_mAh -= tx_packets * self.config.packet_tx_energy_mAh
        person.battery_remaining_mAh -= scans * self.config.scan_energy_mAh
        person.battery_remaining_mAh -= idle_minutes * self.config.idle_energy_mAh_per_min
        person.battery_remaining_mAh = max(person.battery_remaining_mAh, 0)
        self.energy_log.append({
            "time": self.env.now,
            "pid": person.pid,
            "role": person.ptype.name,
            "battery_remaining_mAh": person.battery_remaining_mAh,
        })

    def _perform_scan_and_contacts(self, person: Person):
        scans = 1
        tx_packets = 0

        for other in self.persons.values():
            if other.pid <= person.pid:
                continue

            rssi = self._estimate_rssi(person, other)
            est_d = self._estimate_distance_from_rssi(rssi)
            same_zone = person.current_zone == other.current_zone
            in_contact = same_zone and est_d <= self.config.distance_threshold_m
            pair = (person.pid, other.pid)

            if in_contact:
                if pair not in self.active_contacts:
                    self.active_contacts[pair] = {
                        "start": self.env.now,
                        "zone": person.current_zone,
                        "distance_samples": [est_d],
                        "rssi_samples": [rssi],
                        "mask_pair": (person.masked, other.masked),
                        "roles": (person.ptype.name, other.ptype.name),
                    }
                else:
                    self.active_contacts[pair]["distance_samples"].append(est_d)
                    self.active_contacts[pair]["rssi_samples"].append(rssi)
            else:
                self._close_contact_if_open(person.pid, other.pid)

            tx_packets += 1
            self.packet_log.append({
                "time": self.env.now,
                "src": person.pid,
                "dst": other.pid,
                "src_zone": person.current_zone,
                "dst_zone": other.current_zone,
                "same_zone": same_zone,
                "rssi": rssi,
                "estimated_distance": est_d,
            })

        self._consume_energy(person, tx_packets=tx_packets, scans=scans)

    def _close_contact_if_open(self, pid1: int, pid2: int):
        pair = tuple(sorted((pid1, pid2)))
        if pair not in self.active_contacts:
            return

        rec = self.active_contacts.pop(pair)
        duration = self.env.now - rec["start"]
        if duration <= 0:
            return

        avg_distance = float(np.mean(rec["distance_samples"]))
        avg_rssi = float(np.mean(rec["rssi_samples"]))
        self.contact_log.append({
            "p1": pair[0],
            "p2": pair[1],
            "start": rec["start"],
            "end": self.env.now,
            "duration": duration,
            "zone": rec["zone"],
            "avg_distance": avg_distance,
            "avg_rssi": avg_rssi,
            "p1_masked": rec["mask_pair"][0],
            "p2_masked": rec["mask_pair"][1],
            "p1_role": rec["roles"][0],
            "p2_role": rec["roles"][1],
        })

    # -------------------------
    # Infection model
    # -------------------------

    def _contact_infection_probability(self, src: Person, dst: Person, duration: int, avg_distance: float, zone_name: str) -> float:
        if avg_distance > self.config.distance_threshold_m:
            return 0.0
        zone_risk = self.zones[zone_name].risk_multiplier
        duration_factor = min(2.0, duration / max(1, self.config.contact_time_threshold))
        distance_factor = max(0.2, 1.0 - (avg_distance / (self.config.distance_threshold_m + 0.01)))
        role_factor = src.ptype.role_risk_multiplier * dst.ptype.role_risk_multiplier

        mask_factor = 1.0
        if src.masked:
            mask_factor *= (1 - self.config.mask_effectiveness_tx)
        if dst.masked:
            mask_factor *= (1 - self.config.mask_effectiveness_rx)

        prob = (
            self.config.infection_probability_base
            * zone_risk
            * duration_factor
            * distance_factor
            * role_factor
            * mask_factor
        )
        return min(0.95, max(0.0, prob))

    def _process_new_contact_infections(self):
        if not self.contact_log:
            return

        recent = [c for c in self.contact_log if c["end"] == self.env.now]
        for contact in recent:
            if contact["duration"] < self.config.contact_time_threshold:
                continue

            p1 = self.persons[contact["p1"]]
            p2 = self.persons[contact["p2"]]

            candidate_pairs = []
            if p1.infectious and not p2.infected and not p2.recovered:
                candidate_pairs.append((p1, p2))
            if p2.infectious and not p1.infected and not p1.recovered:
                candidate_pairs.append((p2, p1))

            for src, dst in candidate_pairs:
                prob = self._contact_infection_probability(
                    src, dst, contact["duration"], contact["avg_distance"], contact["zone"]
                )
                if random.random() < prob:
                    dst.infected = True
                    dst.infectious = False
                    dst.infection_time = self.env.now
                    dst.symptom_onset_time = self.env.now + self.config.incubation_minutes
                    dst.infection_source = src.pid
                    self.infection_graph.add_edge(src.pid, dst.pid, time=self.env.now, zone=contact["zone"])
                    self.infection_log.append({
                        "time": self.env.now,
                        "source": src.pid,
                        "target": dst.pid,
                        "zone": contact["zone"],
                        "event": "transmission",
                        "probability": prob,
                    })

    def _update_disease_states(self, person: Person):
        if person.infected and not person.infectious and person.infection_time is not None:
            if self.env.now - person.infection_time >= self.config.infectious_delay_minutes:
                person.infectious = True

        if person.infected and person.infection_time is not None:
            if self.env.now - person.infection_time >= self.config.recovery_minutes:
                person.infected = False
                person.infectious = False
                person.recovered = True

        if self.config.enable_isolation and person.infected and person.symptom_onset_time is not None:
            if self.env.now >= person.symptom_onset_time + self.config.isolation_trigger_after_symptoms:
                person.isolated = True

    # -------------------------
    # SimPy processes
    # -------------------------

    def person_process(self, person: Person):
        while self.env.now < self.config.simulation_minutes:
            self._update_disease_states(person)
            self._move_person(person)

            if (self.env.now - person.last_packet_time) >= person.ptype.transmit_interval:
                self._perform_scan_and_contacts(person)
                person.last_packet_time = self.env.now
            else:
                self._consume_energy(person, tx_packets=0, scans=0, idle_minutes=1)

            yield self.env.timeout(self.config.movement_step_minutes)

    def global_monitor_process(self):
        while self.env.now < self.config.simulation_minutes:
            # Close invalid contacts
            for pair in list(self.active_contacts.keys()):
                a = self.persons[pair[0]]
                b = self.persons[pair[1]]
                if a.current_zone != b.current_zone:
                    self._close_contact_if_open(*pair)
                    continue
                d = self._distance(a, b)
                if d > self.config.distance_threshold_m * 1.35:
                    self._close_contact_if_open(*pair)

            self._process_new_contact_infections()
            yield self.env.timeout(1)

        for pair in list(self.active_contacts.keys()):
            self._close_contact_if_open(*pair)

    # -------------------------
    # Run and results
    # -------------------------

    def run(self):
        for person in self.persons.values():
            self.env.process(self.person_process(person))
        self.env.process(self.global_monitor_process())
        self.env.run(until=self.config.simulation_minutes)

    def results(self) -> Dict[str, pd.DataFrame]:
        contacts_df = pd.DataFrame(self.contact_log)
        packets_df = pd.DataFrame(self.packet_log)
        infections_df = pd.DataFrame(self.infection_log)
        energy_df = pd.DataFrame(self.energy_log)
        presence_df = pd.DataFrame(self.presence_log)

        if contacts_df.empty:
            contacts_df = pd.DataFrame(columns=["p1", "p2", "start", "end", "duration", "zone", "avg_distance", "avg_rssi"])
        if packets_df.empty:
            packets_df = pd.DataFrame(columns=["time", "src", "dst", "src_zone", "dst_zone", "same_zone", "rssi", "estimated_distance"])
        if infections_df.empty:
            infections_df = pd.DataFrame(columns=["time", "source", "target", "zone", "event", "probability"])
        if energy_df.empty:
            energy_df = pd.DataFrame(columns=["time", "pid", "role", "battery_remaining_mAh"])
        if presence_df.empty:
            presence_df = pd.DataFrame(columns=["time", "pid", "role", "zone", "x", "y", "infected", "isolated"])

        return {
            "contacts": contacts_df,
            "packets": packets_df,
            "infections": infections_df,
            "energy": energy_df,
            "presence": presence_df,
        }

    def summary(self) -> Dict[str, float]:
        dfs = self.results()
        contacts_df = dfs["contacts"]
        infections_df = dfs["infections"]
        energy_df = dfs["energy"]

        ever_infected = sum(1 for p in self.persons.values() if p.infection_time is not None)
        total_population = len(self.persons)
        total_contacts = len(contacts_df)
        mean_contact_duration = float(contacts_df["duration"].mean()) if not contacts_df.empty else 0.0
        transmissions = int((infections_df["event"] == "transmission").sum()) if not infections_df.empty else 0

        latest_energy = energy_df.sort_values("time").groupby("pid").tail(1) if not energy_df.empty else pd.DataFrame()
        avg_remaining_battery = float(latest_energy["battery_remaining_mAh"].mean()) if not latest_energy.empty else 0.0
        return {
            "population": total_population,
            "ever_infected": ever_infected,
            "infection_ratio": ever_infected / total_population,
            "total_contacts": total_contacts,
            "mean_contact_duration": mean_contact_duration,
            "transmissions": transmissions,
            "avg_remaining_battery_mAh": avg_remaining_battery,
        }

    # -------------------------
    # Analytics helpers
    # -------------------------

    def contact_graph(self) -> nx.Graph:
        g = nx.Graph()
        for pid, person in self.persons.items():
            g.add_node(pid, role=person.ptype.name, infected=person.infection_time is not None)
        for rec in self.contact_log:
            weight = rec["duration"]
            if g.has_edge(rec["p1"], rec["p2"]):
                g[rec["p1"]][rec["p2"]]["weight"] += weight
                g[rec["p1"]][rec["p2"]]["count"] += 1
            else:
                g.add_edge(rec["p1"], rec["p2"], weight=weight, count=1, zone=rec["zone"])
        return g

    def high_risk_individuals(self, top_n: int = 10) -> pd.DataFrame:
        g = self.contact_graph()
        degree = dict(g.degree())
        weighted_degree = dict(g.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(g) if g.number_of_edges() > 0 else {n: 0 for n in g.nodes}

        rows = []
        for pid, person in self.persons.items():
            rows.append({
                "pid": pid,
                "role": person.ptype.name,
                "infected": person.infection_time is not None,
                "isolated": person.isolated,
                "degree": degree.get(pid, 0),
                "weighted_degree": weighted_degree.get(pid, 0.0),
                "betweenness": betweenness.get(pid, 0.0),
            })
        df = pd.DataFrame(rows)
        return df.sort_values(["weighted_degree", "betweenness"], ascending=False).head(top_n)

    def risky_zones(self) -> pd.DataFrame:
        contacts_df = pd.DataFrame(self.contact_log)
        infections_df = pd.DataFrame(self.infection_log)

        if contacts_df.empty:
            return pd.DataFrame(columns=["zone", "contact_count", "total_duration", "transmissions"])

        zone_stats = contacts_df.groupby("zone").agg(
            contact_count=("zone", "count"),
            total_duration=("duration", "sum"),
            mean_distance=("avg_distance", "mean"),
            mean_rssi=("avg_rssi", "mean"),
        ).reset_index()

        if not infections_df.empty:
            trans = infections_df[infections_df["event"] == "transmission"].groupby("zone").size().reset_index(name="transmissions")
            zone_stats = zone_stats.merge(trans, on="zone", how="left")
        else:
            zone_stats["transmissions"] = 0
        zone_stats["transmissions"] = zone_stats["transmissions"].fillna(0).astype(int)
        return zone_stats.sort_values(["transmissions", "total_duration", "contact_count"], ascending=False)

    # -------------------------
    # ML-style risk prediction (KNN, numpy only)
    # -------------------------

    def build_ml_dataset(self) -> pd.DataFrame:
        contacts_df = pd.DataFrame(self.contact_log)
        if contacts_df.empty:
            self.ml_dataset_cache = pd.DataFrame()
            return self.ml_dataset_cache

        records = []
        for _, row in contacts_df.iterrows():
            p1 = self.persons[int(row["p1"])]
            p2 = self.persons[int(row["p2"])]
            transmission = 0
            for rec in self.infection_log:
                if rec["event"] != "transmission":
                    continue
                src, dst = rec["source"], rec["target"]
                if {src, dst} == {int(row["p1"]), int(row["p2"])} and rec["time"] >= row["start"]:
                    transmission = 1
                    break

            records.append({
                "duration": row["duration"],
                "avg_distance": row["avg_distance"],
                "avg_rssi": row["avg_rssi"],
                "zone_risk": self.zones[row["zone"]].risk_multiplier,
                "mask_count": int(bool(row.get("p1_masked", False))) + int(bool(row.get("p2_masked", False))),
                "source_risk_role": max(p1.ptype.role_risk_multiplier, p2.ptype.role_risk_multiplier),
                "label": transmission,
            })

        self.ml_dataset_cache = pd.DataFrame(records)
        return self.ml_dataset_cache

    def _knn_predict_probability(self, X_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray, k: int) -> float:
        if len(X_train) == 0:
            return 0.0
        dists = np.linalg.norm(X_train - x_query, axis=1)
        idx = np.argsort(dists)[: min(k, len(dists))]
        nearest_labels = y_train[idx]
        return float(np.mean(nearest_labels))

    def predict_contact_risk_with_knn(self, top_n: int = 10) -> pd.DataFrame:
        df = self.build_ml_dataset().copy()
        if df.empty or len(df) < 6:
            return pd.DataFrame(columns=["duration", "avg_distance", "avg_rssi", "zone_risk", "mask_count", "predicted_risk"])

        features = ["duration", "avg_distance", "avg_rssi", "zone_risk", "mask_count", "source_risk_role"]
        X = df[features].to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=float)

        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-9
        Xs = (X - mu) / sigma

        preds = []
        for i in range(len(df)):
            train_idx = np.array([j for j in range(len(df)) if j != i])
            p = self._knn_predict_probability(Xs[train_idx], y[train_idx], Xs[i], self.config.ml_k_neighbors)
            preds.append(p)
        df["predicted_risk"] = preds
        return df.sort_values("predicted_risk", ascending=False).head(top_n)

    # -------------------------
    # Visualization: map, animation, dashboard
    # -------------------------

    def draw_hospital_map(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        color_map = {
            "ER": "#ffd7d7",
            "ICU": "#ffe8b3",
            "Ward_A": "#d9f0ff",
            "Ward_B": "#d9f0ff",
            "Corridor": "#efefef",
            "Lab": "#e6ddff",
            "Waiting": "#e1ffd8",
            "StaffRoom": "#f9ddff",
        }
        for zone in self.zones.values():
            rect = patches.Rectangle((zone.x, zone.y), zone.width, zone.height,
                                     linewidth=1.5, edgecolor="black",
                                     facecolor=color_map.get(zone.name, "white"), alpha=0.8)
            ax.add_patch(rect)
            ax.text(zone.x + zone.width / 2, zone.y + zone.height / 2, zone.name,
                    ha="center", va="center", fontsize=10, weight="bold")
        ax.set_xlim(-1, 48)
        ax.set_ylim(-1, 22)
        ax.set_aspect("equal")
        ax.set_title("2D Hospital Layout")
        return ax

    def plot_current_map(self, time_step: Optional[int] = None):
        presence_df = pd.DataFrame(self.presence_log)
        if presence_df.empty:
            print("No presence data available.")
            return
        if time_step is None:
            time_step = int(presence_df["time"].max())
        frame = presence_df[presence_df["time"] == time_step]

        fig, ax = plt.subplots(figsize=(12, 6))
        self.draw_hospital_map(ax)
        for role, marker in [("patient", "o"), ("doctor", "s"), ("nurse", "^"), ("visitor", "D"), ("staff", "P")]:
            part = frame[frame["role"] == role]
            if part.empty:
                continue
            colors = ["red" if inf else "blue" for inf in part["infected"]]
            ax.scatter(part["x"], part["y"], s=70, marker=marker, c=colors, label=role)
        ax.legend(loc="upper right")
        ax.set_title(f"Hospital Map at t={time_step} min")
        plt.tight_layout()
        plt.show()

    def animate_agents(self, save_path: Optional[str] = None, interval_ms: int = 200):
        presence_df = pd.DataFrame(self.presence_log)
        if presence_df.empty:
            print("No presence data to animate.")
            return
        times = sorted(presence_df["time"].unique())

        fig, ax = plt.subplots(figsize=(12, 6))
        self.draw_hospital_map(ax)
        scat = ax.scatter([], [], s=80)
        txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

        def update(frame_time):
            frame = presence_df[presence_df["time"] == frame_time]
            offsets = frame[["x", "y"]].to_numpy() if not frame.empty else np.empty((0, 2))
            colors = ["red" if inf else "royalblue" for inf in frame["infected"]]
            sizes = [120 if role in ["doctor", "nurse"] else 80 for role in frame["role"]]
            scat.set_offsets(offsets)
            scat.set_color(colors)
            scat.set_sizes(sizes)
            txt.set_text(f"Time: {frame_time} min | Agents: {len(frame)}")
            return scat, txt

        anim = FuncAnimation(fig, update, frames=times, interval=interval_ms, blit=False, repeat=False)
        plt.tight_layout()
        if save_path:
            try:
                anim.save(save_path, writer="pillow", fps=6)
                print(f"Animation saved to: {save_path}")
            except Exception as e:
                print("Animation save failed:", e)
        plt.show()
        return anim

    def plot_infection_timeline(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        minutes = np.arange(0, self.config.simulation_minutes + 1)
        cumulative = []
        for t in minutes:
            infected_count = sum(1 for p in self.persons.values() if p.infection_time is not None and p.infection_time <= t)
            cumulative.append(infected_count)
        ax.plot(minutes, cumulative)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Cumulative infected")
        ax.set_title("Infection Spread Timeline")
        return ax

    def plot_zone_heatmap(self, ax=None):
        zone_df = self.risky_zones()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        if zone_df.empty:
            ax.text(0.5, 0.5, "No zone activity", ha="center", va="center")
            ax.set_axis_off()
            return ax
        ax.bar(zone_df["zone"], zone_df["contact_count"])
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel("Contact count")
        ax.set_title("Risky Zones by Contact Volume")
        return ax

    def plot_battery_levels(self, ax=None):
        energy_df = pd.DataFrame(self.energy_log)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        if energy_df.empty:
            ax.text(0.5, 0.5, "No energy data", ha="center", va="center")
            ax.set_axis_off()
            return ax
        latest = energy_df.sort_values("time").groupby("pid").tail(1)
        ax.bar(latest["pid"].astype(str), latest["battery_remaining_mAh"])
        ax.set_xlabel("Person ID")
        ax.set_ylabel("Remaining battery (mAh)")
        ax.set_title("Wearable Sensor Battery Levels")
        return ax

    def plot_contact_graph(self, ax=None):
        g = self.contact_graph()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if g.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No graph data", ha="center", va="center")
            ax.set_axis_off()
            return ax
        pos = nx.spring_layout(g, seed=11)
        colors = []
        for node in g.nodes():
            person = self.persons[node]
            if person.infection_time is not None:
                colors.append("red")
            elif person.ptype.name in {"doctor", "nurse"}:
                colors.append("orange")
            else:
                colors.append("skyblue")
        weights = [max(1, g[u][v]["weight"] / 5) for u, v in g.edges()]
        nx.draw_networkx(g, pos=pos, node_color=colors, with_labels=True, width=weights, font_size=7, ax=ax)
        ax.set_title("Hospital Contact Graph")
        ax.axis("off")
        return ax

    def dashboard(self):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3)

        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_infection_timeline(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_zone_heatmap(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_battery_levels(ax3)

        ax4 = fig.add_subplot(gs[1, 0:2])
        self.plot_contact_graph(ax4)

        ax5 = fig.add_subplot(gs[1, 2])
        pred_df = self.predict_contact_risk_with_knn(8)
        ax5.axis("off")
        if pred_df.empty:
            ax5.text(0.5, 0.5, "ML risk table not available yet", ha="center", va="center")
        else:
            show = pred_df[["duration", "avg_distance", "avg_rssi", "predicted_risk"]].round(3)
            table = ax5.table(cellText=show.values,
                              colLabels=show.columns,
                              loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.1, 1.4)
            ax5.set_title("KNN Contact Risk Prediction")

        fig.suptitle("Hospital WSN Analytics Dashboard", fontsize=16, weight="bold")
        plt.tight_layout()
        plt.show()


# -----------------------------
# Scenario utilities
# -----------------------------

def run_scenario(name: str, **kwargs) -> Tuple[HospitalWSNSimulator, Dict[str, float]]:
    cfg = ScenarioConfig(**kwargs)
    sim = HospitalWSNSimulator(cfg)
    sim.run()
    summary = sim.summary()
    print(f"\n=== Scenario: {name} ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return sim, summary


def compare_scenarios() -> pd.DataFrame:
    base_sim, base_summary = run_scenario("Base Scenario")
    iso_sim, iso_summary = run_scenario(
        "Aggressive Isolation",
        enable_isolation=True,
        isolation_trigger_after_symptoms=5,
        seed=43,
    )
    restrict_sim, restrict_summary = run_scenario(
        "Visitor Restriction",
        visitor_restriction=True,
        seed=44,
    )
    low_mask_sim, low_mask_summary = run_scenario(
        "Low Mask Compliance",
        seed=45,
    )
    for p in low_mask_sim.persons.values():
        p.masked = False

    comparison = pd.DataFrame([
        {"scenario": "Base", **base_summary},
        {"scenario": "Aggressive Isolation", **iso_summary},
        {"scenario": "Visitor Restriction", **restrict_summary},
        {"scenario": "Low Mask Compliance", **low_mask_summary},
    ])
    print("\nScenario comparison:")
    print(comparison)
    return comparison


# -----------------------------
# Main execution
# -----------------------------

if __name__ == "__main__":
    sim, summary = run_scenario(
        "Advanced Hospital WSN Contact Tracing",
        simulation_minutes=8 * 60,
        initial_infected=2,
        infection_probability_base=0.06,
        contact_time_threshold=3,
        enable_isolation=True,
        isolation_trigger_after_symptoms=15,
        visitor_restriction=False,
        seed=42,
    )

    print("\nTop high-risk individuals:")
    print(sim.high_risk_individuals(10))

    print("\nRisky zones:")
    print(sim.risky_zones())

    print("\nTop ML-predicted risky contacts:")
    print(sim.predict_contact_risk_with_knn(10))

    # 2D map snapshot
    sim.plot_current_map()

    # Main dashboard
    sim.dashboard()

    # Separate optional plots
    # sim.plot_infection_timeline()
    # sim.plot_zone_heatmap()
    # sim.plot_battery_levels()
    # sim.plot_contact_graph()

    # Agent animation (optionally save GIF if pillow is available)
    sim.animate_agents(save_path="hospital_agents.gif")

    # Scenario comparison
    comparison_df = compare_scenarios()
    print(comparison_df)

    # Raw tables
    dfs = sim.results()
    # dfs["contacts"].to_csv("contacts.csv", index=False)
    # dfs["infections"].to_csv("infections.csv", index=False)
    # dfs["presence"].to_csv("presence.csv", index=False)

