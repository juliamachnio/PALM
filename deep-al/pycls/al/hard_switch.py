"""Proxy-derived three-stage hard-switch baseline (TypiClust -> CoreSet -> uncertainty)."""

import json
from pathlib import Path

import numpy as np

from .Sampling import CoreSetMIPSampling, Sampling
from .typiclust import TypiClust


class HardSwitch:
    def __init__(self, cfg, dataObj, lSet, uSet, budgetSize):
        self.cfg, self.dataObj = cfg, dataObj
        self.lSet, self.uSet = np.asarray(lSet, dtype=np.int64), np.asarray(uSet, dtype=np.int64)
        self.budgetSize = int(budgetSize)
        path = Path(str(cfg.ACTIVE_LEARNING.HARD_SWITCH_SCHEDULE))
        if not path.is_file():
            raise FileNotFoundError("hard_switch requires --hard-switch-schedule produced by proxy phase analysis")
        payload = json.loads(path.read_text())
        if payload.get("schema") != "mechanistic-hard-switch-v1" or payload.get("source") != "proxy_phase_analysis":
            raise ValueError("hard_switch schedule is not a proxy-phase-analysis artifact")
        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, list) or len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1]:
            raise ValueError("hard_switch schedule must contain two increasing positive thresholds")
        self.thresholds = tuple(float(value) for value in thresholds)

    def stage(self):
        n = len(self.lSet)
        return "typiclust" if n < self.thresholds[0] else "coreset" if n < self.thresholds[1] else "uncertainty"

    def _save_diagnostics(self, stage):
        episode_dir = getattr(self.cfg, "EPISODE_DIR", "")
        if episode_dir:
            path = Path(episode_dir) / "hard_switch_diagnostics.json"
            path.write_text(json.dumps({"method": "hard_switch", "source": "proxy_phase_analysis", "current_labeled": len(self.lSet), "thresholds": self.thresholds, "stage": stage}, indent=2))

    def select_samples(self, clf_model, train_dataset):
        stage = self.stage()
        if stage == "typiclust":
            active, remaining = TypiClust(self.cfg, self.lSet, self.uSet, budgetSize=self.budgetSize).select_samples()
        elif stage == "coreset":
            penultimate, training = clf_model.penultimate_active, clf_model.training
            clf_model.penultimate_active = True
            clf_model.eval()
            active, remaining = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj).query(self.lSet, self.uSet, clf_model, train_dataset)
            clf_model.penultimate_active, _ = penultimate, clf_model.train(training)
        else:
            training = clf_model.training
            clf_model.eval()
            active, remaining = Sampling(self.dataObj, self.cfg).uncertainty(self.budgetSize, self.lSet, self.uSet, clf_model, train_dataset)
            clf_model.train(training)
        self._save_diagnostics(stage)
        return active, remaining
