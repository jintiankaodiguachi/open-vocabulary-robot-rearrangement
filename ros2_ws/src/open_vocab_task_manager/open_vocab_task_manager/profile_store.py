from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import yaml

from .task_models import DestinationProfile


PROFILE_PARAMETER_KEYS = (
    "destination_rim_height_m",
    "destination_xy_offset_x_m",
    "destination_xy_offset_y_m",
    "release_grasp_center_above_rim_m",
    "preplace_clearance_m",
)


class DestinationProfileStore:
    """Resolve place *geometry* without restricting destination semantics.

    V10.5 deliberately separates two concerns:

    1. Semantic grounding is open-vocabulary.  Any non-empty destination string is
       allowed and is forwarded to PlaceExecutor / Grounded-SAM2 unchanged unless
       an optional alias override explicitly supplies a canonical query.
    2. Physical place geometry (rim height, XY calibration bias, release height,
       pre-place clearance) comes from a generic default policy and may be refined
       by optional per-receptacle overrides.

    The default policy is intentionally allowed to inherit the PlaceExecutor
    parameters captured at TaskManager startup.  In the user's current system,
    those are the already-validated glass-bowl place parameters, so an unseen
    destination such as "cardboard box" can reuse the same calibrated geometry
    while still being localized dynamically by Grounded-SAM2 + Homography.
    """

    def __init__(self, path: str):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Destination geometry file not found: {self.path}")

        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Destination geometry YAML must be a mapping: {self.path}")

        # V10.5 schema: `default` + optional `overrides`.
        # Backward compatibility: old `profiles` are accepted as overrides.
        raw_default = data.get("default") or {}
        if not isinstance(raw_default, dict):
            raise ValueError("'default' must be a mapping")

        self._default = DestinationProfile(
            name="<generic-default>",
            query="",
            aliases=(),
            inherit_place_baseline=bool(raw_default.get("inherit_place_baseline", True)),
            require_explicit_geometry=bool(raw_default.get("require_explicit_geometry", False)),
            values={k: raw_default.get(k) for k in PROFILE_PARAMETER_KEYS},
            matched_override=False,
        )
        self._validate(self._default, allow_empty_query=True)

        raw_overrides = data.get("overrides")
        if raw_overrides is None:
            raw_overrides = data.get("profiles") or {}
        if not isinstance(raw_overrides, dict):
            raise ValueError("'overrides' must be a mapping")

        self._profiles: Dict[str, DestinationProfile] = {}
        self._aliases: Dict[str, str] = {}
        for name, raw in raw_overrides.items():
            if not isinstance(raw, dict):
                raise ValueError(f"Destination override '{name}' must be a mapping")

            canonical = str(name).strip()
            query = str(raw.get("query", canonical)).strip()
            if not canonical or not query:
                raise ValueError("Destination override name/query may not be empty")

            aliases = tuple(
                str(v).strip()
                for v in (raw.get("aliases") or [])
                if str(v).strip()
            )
            profile = DestinationProfile(
                name=canonical,
                query=query,
                aliases=aliases,
                inherit_place_baseline=bool(
                    raw.get(
                        "inherit_place_baseline",
                        self._default.inherit_place_baseline,
                    )
                ),
                require_explicit_geometry=bool(raw.get("require_explicit_geometry", False)),
                values={k: raw.get(k) for k in PROFILE_PARAMETER_KEYS},
                matched_override=True,
            )
            self._validate(profile)

            canonical_key = canonical.casefold()
            self._profiles[canonical_key] = profile
            for alias in (canonical, query, *aliases):
                key = alias.casefold()
                previous = self._aliases.get(key)
                if previous is not None and previous != canonical_key:
                    raise ValueError(f"Destination alias '{alias}' is ambiguous")
                self._aliases[key] = canonical_key

    def resolve_or_default(self, requested: str) -> DestinationProfile:
        """Return an optional override or a generic profile for any query.

        Unknown destination text is NOT an error.  The raw user/VLM query is
        preserved verbatim for Grounded-SAM2 and only the physical geometry falls
        back to the generic policy.
        """
        requested_clean = requested.strip()
        if not requested_clean:
            raise ValueError("destination_query is empty")

        canonical_key = self._aliases.get(requested_clean.casefold())
        if canonical_key is not None:
            profile = self._profiles[canonical_key]
            self._validate(profile)
            return profile

        # Open-vocabulary fallback: semantic query remains EXACTLY what the user
        # supplied; only geometry is inherited/defaulted.
        profile = DestinationProfile(
            name=requested_clean,
            query=requested_clean,
            aliases=(),
            inherit_place_baseline=self._default.inherit_place_baseline,
            require_explicit_geometry=self._default.require_explicit_geometry,
            values=dict(self._default.values),
            matched_override=False,
        )
        self._validate(profile)
        return profile

    # Kept for source compatibility with any external helper that still calls
    # resolve().  In V10.5 it intentionally has open-vocabulary semantics.
    def resolve(self, requested: str) -> DestinationProfile:
        return self.resolve_or_default(requested)

    @staticmethod
    def _validate(profile: DestinationProfile, *, allow_empty_query: bool = False) -> None:
        if not allow_empty_query and not profile.query.strip():
            raise ValueError("Destination query may not be empty")

        if profile.require_explicit_geometry:
            missing = [k for k, v in profile.values.items() if v is None]
            if missing:
                raise ValueError(
                    f"Destination geometry '{profile.name}' requires explicit values, missing: {missing}"
                )

        rim = profile.values.get("destination_rim_height_m")
        if profile.require_explicit_geometry and (rim is None or float(rim) <= 0.0):
            raise ValueError(
                f"Destination geometry '{profile.name}' has invalid "
                f"destination_rim_height_m={rim}. Measure tabletop-to-rim height "
                "in metres before autonomous execution."
            )

    def names(self) -> Iterable[str]:
        """Names of optional geometry overrides, not allowed semantic classes."""
        return (p.name for p in self._profiles.values())

    @property
    def uses_startup_baseline_by_default(self) -> bool:
        return bool(self._default.inherit_place_baseline)
