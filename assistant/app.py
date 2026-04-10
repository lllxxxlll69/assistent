from __future__ import annotations

from dataclasses import dataclass

from assistant.config.settings import SettingsManager
from assistant.core.agent import Agent
from assistant.core.orchestrator import Orchestrator
from assistant.llm.client import LLMClient
from assistant.memory.memory_manager import MemoryManager
from assistant.tools.file_tools import FileTools
from assistant.tools.search_tools import SearchTools
from assistant.tools.vision_tools import VisionTools


@dataclass(slots=True)
class AssistantBackend:
    settings_manager: SettingsManager
    memory_manager: MemoryManager
    orchestrator: Orchestrator


def build_backend(settings_manager: SettingsManager | None = None) -> AssistantBackend:
    manager = settings_manager or SettingsManager()
    settings = manager.get_settings()

    llm_client = LLMClient(settings)
    memory_manager = MemoryManager(manager)
    orchestrator = Orchestrator(
        agent=Agent(),
        settings_manager=manager,
        memory_manager=memory_manager,
        llm_client=llm_client,
        file_tools=FileTools(),
        vision_tools=VisionTools(LLMClient(settings)),
        search_tools=SearchTools(settings),
    )
    return AssistantBackend(
        settings_manager=manager,
        memory_manager=memory_manager,
        orchestrator=orchestrator,
    )
