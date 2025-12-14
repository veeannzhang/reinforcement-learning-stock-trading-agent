import time
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class MultiCondEarlyStop(BaseCallback):
    """Custom callback that handles multiple early-stopping criteria.

    Parameters:
    -----------
        loss_threshold : float
            The threshold to evaluate training loss.
        loss_window : int
            If training loss < `loss_threshold` in each step of
            the loss window, stop early.
        max_time_seconds : int
            If the time elapsed from the start of training exceeds
            'max_time_seconds', stop early.
        verbose : int
    """
    def __init__(
        self, 
        loss_threshold: float=None,
        loss_window: int=3, 
        max_time_seconds: int=None, 
        verbose: int=1
    ):
        super().__init__(verbose)
        self.loss_threshold = loss_threshold
        self.max_time_seconds = max_time_seconds
        self.start_time = None
        self.loss_window = loss_window
        # track only the past <loss_window> losses
        self.latest_losses = deque(maxlen=self.loss_window)

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_rollout_end(self) -> bool:
        """Prints and stores training loss at each rollout end."""
        loss = self.logger.name_to_value.get("train/loss", None)
        if loss is not None:
            self.latest_losses.append(loss)
            if self.verbose:
                print(f"[MultiCondEarlyStop] train/loss = {loss:.6f}")
        # continue
        return True

    def _on_step(self) -> bool:
        """Checks early-stopping criteria after each step."""
        # ---- 1) Time-based stopping ----
        if self.max_time_seconds is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_time_seconds:
                if self.verbose:
                    print(f"\nStopping: time limit reached ({elapsed:.1f} s)")
                # terminate
                return False

        # ---- 2) Loss-based stopping ----
        if self.loss_threshold is not None and\
            len(self.latest_losses) == self.loss_window:

            if all(loss < self.loss_threshold for loss in self.latest_losses):
                if self.verbose:
                    print(
                        f"\nStopping: train/loss in past {self.loss_window} steps"
                        f"< {self.loss_threshold}"
                        f"\nAverage: {np.mean(self.latest_losses).round(4)}"
                    )
                # terminate
                return False
        # continue
        return True