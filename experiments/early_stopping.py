import time
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class MultiCondEarlyStop(BaseCallback):
    def __init__(
        self, 
        loss_threshold=None,
        loss_window=3, 
        max_time_seconds=None, 
        verbose=1
    ):
        super().__init__(verbose)
        self.loss_threshold = loss_threshold
        self.max_time_seconds = max_time_seconds
        self.start_time = None
        self.loss_window = loss_window
        self.latest_losses = deque(maxlen=self.loss_window)

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_rollout_end(self) -> bool:
        # PPO/RecurrentPPO does its training updates at rollout end,
        # so this is where train/loss gets updated.
        loss = self.logger.name_to_value.get("train/loss", None)
        if loss is not None:
            self.latest_losses.append(loss)
            if self.verbose:
                print(f"[MultiCondEarlyStop] train/loss = {loss:.6f}")
        return True  # continue, actual stopping decision is in _on_step

    def _on_step(self) -> bool:
        # ---- 1) Time-based stopping ----
        if self.max_time_seconds is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_time_seconds:
                if self.verbose:
                    print(f"\n⏱ Stopping: time limit reached ({elapsed:.1f} s)")
                return False

        # ---- 2) Loss-based stopping ----
        if self.loss_threshold is not None and\
            len(self.latest_losses) == self.loss_window:

            if all(loss < self.loss_threshold for loss in self.latest_losses):
                if self.verbose:
                    print(
                        f"\n📉 Stopping: train/loss in past {self.loss_window} steps"
                        f"< {self.loss_threshold}"
                        f"\nAverage: {np.mean(self.latest_losses).round(4)}"
                    )
                return False

        return True  # keep training