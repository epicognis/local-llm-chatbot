"""llm-hw-monitor — realtime LLM + local-hardware monitor.

Standalone utility: samples GPU (power/temp/VRAM/util via nvidia-smi),
CPU/memory/process (via psutil), and the Ollama daemon (/api/ps) so you can
gauge process, memory, power, and temperature per resident model while the
local hardware is being exercised. The LLM name is shown as informational.
"""

__version__ = "0.1.0"
