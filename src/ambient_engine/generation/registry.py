from __future__ import annotations

from dataclasses import dataclass

from ambient_engine.core.runtime import RuntimeContext
from ambient_engine.generation.contracts import BaseProvider
from ambient_engine.generation.providers.acestep import AceStepProvider
from ambient_engine.generation.providers.procedural import ProceduralAmbientProvider
from ambient_engine.generation.providers.stable_audio import StableAudioOpenProvider
from ambient_engine.generation.providers.yue import YuEProvider


@dataclass
class ProviderSelection:
    task: str
    provider_name: str
    fallback_chain: list[str]
    reason: str


class ProviderRegistry:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.providers: dict[str, BaseProvider] = {
            "procedural": ProceduralAmbientProvider(
                runtime.provider_capabilities["procedural"].available,
                runtime.provider_capabilities["procedural"].reason,
            ),
            "ace_step": AceStepProvider(
                runtime.provider_capabilities["ace_step"].available,
                runtime.provider_capabilities["ace_step"].reason,
            ),
            "yue": YuEProvider(
                runtime.provider_capabilities["yue"].available,
                runtime.provider_capabilities["yue"].reason,
            ),
            "stable_audio_open": StableAudioOpenProvider(
                runtime.provider_capabilities["stable_audio_open"].available,
                runtime.provider_capabilities["stable_audio_open"].reason,
            ),
        }

    def select(self, task: str, preferred_chain: list[str]) -> ProviderSelection:
        fallback_chain = []
        for candidate in preferred_chain:
            normalized = self._normalize(candidate)
            if normalized not in self.providers:
                continue
            provider = self.providers[normalized]
            fallback_chain.append(provider.name)
            if provider.available:
                return ProviderSelection(
                    task=task,
                    provider_name=provider.name,
                    fallback_chain=fallback_chain,
                    reason=provider.reason,
                )
        provider = self.providers["procedural"]
        fallback_chain.append(provider.name)
        return ProviderSelection(
            task=task,
            provider_name=provider.name,
            fallback_chain=fallback_chain,
            reason="Defaulted to bundled procedural DSP provider.",
        )

    def get(self, provider_name: str) -> BaseProvider:
        mapping = {
            "procedural_dsp": self.providers["procedural"],
            "ace_step_1_5": self.providers["ace_step"],
            "yue": self.providers["yue"],
            "stable_audio_open": self.providers["stable_audio_open"],
        }
        return mapping[provider_name]

    @staticmethod
    def _normalize(candidate: str) -> str:
        lookup = {
            "procedural_dsp": "procedural",
            "ace_step_1_5": "ace_step",
            "stable_audio_open": "stable_audio_open",
            "yue": "yue",
        }
        return lookup.get(candidate, candidate)

