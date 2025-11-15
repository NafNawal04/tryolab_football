from typing import Optional


class TackleAttempt:
    """
    Tracks a single tackle attempt lifecycle and determines outcome.
    All thresholds are in pixel units and time is in frames.
    """

    STATE_CANDIDATE = "candidate"
    STATE_CONTACT = "contact"
    STATE_RESOLVE = "resolve"
    STATE_DONE = "done"

    OUTCOME_SUCCESS = "success"
    OUTCOME_FAIL = "failure"
    OUTCOME_INCONCLUSIVE = "inconclusive"

    def __init__(
        self,
        start_frame: int,
        attacker_id: int,
        defender_id: int,
        defender_team_name: str,
        attacker_team_name: str,
        horizon_frames: int,
        confirm_frames: int,
    ):
        self.start_frame = start_frame
        self.attacker_id = attacker_id
        self.defender_id = defender_id
        self.defender_team_name = defender_team_name
        self.attacker_team_name = attacker_team_name
        self.horizon_frames = horizon_frames
        self.confirm_frames = confirm_frames

        self.state = TackleAttempt.STATE_CANDIDATE
        self.contact_frame = None
        self.resolved_frame = None
        self.outcome = None

        # Internal persistence counters
        self._attacker_control_frames = 0
        self._defender_team_control_frames = 0
        self._frames_since_contact = 0

    def mark_contact(self, frame_number: int):
        if self.state == TackleAttempt.STATE_CANDIDATE:
            self.state = TackleAttempt.STATE_CONTACT
            self.contact_frame = frame_number
            self._frames_since_contact = 0

    def update_resolution(
        self,
        frame_number: int,
        current_possessor_id: Optional[int],
        current_possessor_team_name: str,
        attacker_distance_to_ball_px: float,
        defender_distance_to_ball_px: float,
    ):
        if self.state not in (TackleAttempt.STATE_CONTACT, TackleAttempt.STATE_RESOLVE):
            return

        # Enter resolve state right after contact
        if self.state == TackleAttempt.STATE_CONTACT:
            self.state = TackleAttempt.STATE_RESOLVE
            self._frames_since_contact = 0

        self._frames_since_contact += 1

        # Persistence counts
        if current_possessor_id is not None and current_possessor_id == self.attacker_id:
            self._attacker_control_frames += 1
        if current_possessor_team_name == self.defender_team_name and self.defender_team_name != "":
            self._defender_team_control_frames += 1

        # Resolution rules (pixels-only, persistence based)
        if self._defender_team_control_frames >= self.confirm_frames:
            self._resolve(frame_number, TackleAttempt.OUTCOME_SUCCESS)
            return

        if self._attacker_control_frames >= self.confirm_frames:
            # Attacker retained control long enough after contact → failed tackle
            self._resolve(frame_number, TackleAttempt.OUTCOME_FAIL)
            return

        # Horizon timeout
        if self._frames_since_contact >= self.horizon_frames:
            self._resolve(frame_number, TackleAttempt.OUTCOME_INCONCLUSIVE)

    def _resolve(self, frame_number: int, outcome: str):
        self.state = TackleAttempt.STATE_DONE
        self.outcome = outcome
        self.resolved_frame = frame_number

    @property
    def is_done(self) -> bool:
        return self.state == TackleAttempt.STATE_DONE

    def as_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "contact_frame": self.contact_frame,
            "resolved_frame": self.resolved_frame,
            "attacker_id": self.attacker_id,
            "defender_id": self.defender_id,
            "attacker_team": self.attacker_team_name,
            "defender_team": self.defender_team_name,
            "outcome": self.outcome or "",
            "state": self.state,  # Include state for visualization
        }

