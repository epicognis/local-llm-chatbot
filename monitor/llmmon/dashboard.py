"""Rendering: a live `rich` dashboard, plus a plain-text renderer for non-TTY /
--no-tui / --once use. Both consume a Sample and present the same information.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .sampler import Sample


def _age(expires_at: str) -> str:
    """Render how long until an Ollama model is unloaded, from an RFC3339 stamp."""
    if not expires_at:
        return "?"
    try:
        # Ollama emits e.g. 2024-01-02T03:04:05.123456789-08:00; trim fractional ns.
        cleaned = expires_at
        if "." in cleaned:
            head, tail = cleaned.split(".", 1)
            tz = ""
            for i, ch in enumerate(tail):
                if ch in "+-Zz":
                    tz = tail[i:]
                    break
            cleaned = head + (tz if tz not in ("Z", "z") else "+00:00")
        cleaned = cleaned.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        secs = (dt - datetime.now(timezone.utc)).total_seconds()
        if secs <= 0:
            return "now"
        if secs < 60:
            return f"{secs:.0f}s"
        if secs < 3600:
            return f"{secs / 60:.0f}m"
        return f"{secs / 3600:.1f}h"
    except (ValueError, TypeError):
        return "?"


def _fmt(v, unit: str = "", nd: int = 0) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}{unit}"


# ---------------------------------------------------------------------------
# Plain text (no rich required)
# ---------------------------------------------------------------------------

def render_plain(s: Sample) -> str:
    out: list[str] = []
    stamp = datetime.fromtimestamp(s.ts).strftime("%H:%M:%S")
    out.append(f"[{stamp}]  ollama={s.ollama.base_url}")

    out.append(
        f"  SYSTEM  cpu={_fmt(s.sys.cpu_pct, '%', 1)}  "
        f"mem={_fmt(s.sys.mem_used_mb / 1024, 'GB', 1)}/"
        f"{_fmt(s.sys.mem_total_mb / 1024, 'GB', 1)} ({_fmt(s.sys.mem_pct, '%', 0)})  "
        f"cpu_pwr={_fmt(s.sys.cpu_power_w, 'W', 1)}  "
        f"cpu_temp={_fmt(s.sys.cpu_temp_c, 'C', 0)}"
    )

    if s.gpu.available:
        for d in s.gpu.devices:
            out.append(
                f"  GPU{d.index} {d.name}  util={_fmt(d.util_pct, '%', 0)}  "
                f"vram={_fmt(d.mem_used_mb, 'MB', 0)}/{_fmt(d.mem_total_mb, 'MB', 0)}  "
                f"power={_fmt(d.power_w, 'W', 1)}/{_fmt(d.power_limit_w, 'W', 0)}  "
                f"temp={_fmt(d.temp_c, 'C', 0)}"
            )
    else:
        out.append(f"  GPU     unavailable - {s.gpu.reason}")

    if not s.ollama.reachable:
        out.append(f"  MODELS  ollama unreachable - {s.ollama.reason}")
    elif not s.ollama.running:
        out.append("  MODELS  none resident (daemon idle)")
    else:
        for m in s.ollama.running:
            out.append(
                f"  MODEL   {m.name}  [{m.param_size} {m.quant}]  "
                f"size={_fmt(m.size_mb, 'MB', 0)}  vram={_fmt(m.vram_mb, 'MB', 0)} "
                f"({_fmt(m.pct_gpu, '%', 0)} on GPU)  expires={_age(m.expires_at)}"
            )

    if s.ollama_procs:
        for p in s.ollama_procs:
            out.append(
                f"  PROC    pid={p.pid} {p.name}  cpu={_fmt(p.cpu_pct, '%', 1)}  "
                f"rss={_fmt(p.rss_mb, 'MB', 0)}  gpu_vram={_fmt(p.gpu_vram_mb, 'MB', 0)}"
            )

    if not s.turns_available:
        out.append(f"  TOK/S   unavailable - {s.turns_reason}")
    elif not s.recent_turns:
        out.append("  TOK/S   no completed turns yet")
    else:
        for t in s.recent_turns:
            out.append(
                f"  TURN    {t.model}  tokens={t.completion_tokens}  "
                f"elapsed={_fmt(t.elapsed_s, 's', 1)}  tok/s={_fmt(t.tokens_per_sec, '', 1)}"
            )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Rich live dashboard
# ---------------------------------------------------------------------------

def build_renderable(s: Sample):
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Group
    from rich.text import Text

    def heat(v, lo, hi):
        if v is None:
            return "dim"
        if v >= hi:
            return "bold red"
        if v >= (lo + hi) / 2:
            return "yellow"
        return "green"

    # --- System panel ---
    sys_t = Table.grid(padding=(0, 2))
    sys_t.add_column(style="bold cyan")
    sys_t.add_column()
    sys_t.add_row("CPU", Text(_fmt(s.sys.cpu_pct, "%", 1), style=heat(s.sys.cpu_pct, 40, 85)))
    sys_t.add_row(
        "Memory",
        f"{_fmt(s.sys.mem_used_mb / 1024, 'GB', 1)} / "
        f"{_fmt(s.sys.mem_total_mb / 1024, 'GB', 1)}  ({_fmt(s.sys.mem_pct, '%', 0)})",
    )
    sys_t.add_row("CPU power", _fmt(s.sys.cpu_power_w, " W", 1))
    sys_t.add_row("CPU temp", Text(_fmt(s.sys.cpu_temp_c, " C", 0), style=heat(s.sys.cpu_temp_c, 60, 85)))

    # --- GPU panel ---
    if s.gpu.available and s.gpu.devices:
        gpu_t = Table(expand=True, header_style="bold")
        for col in ("GPU", "Util", "VRAM", "Power", "Temp"):
            gpu_t.add_column(col)
        for d in s.gpu.devices:
            gpu_t.add_row(
                f"{d.index}:{d.name}",
                Text(_fmt(d.util_pct, "%", 0), style=heat(d.util_pct, 30, 90)),
                f"{_fmt(d.mem_used_mb, '', 0)}/{_fmt(d.mem_total_mb, ' MB', 0)}",
                Text(
                    f"{_fmt(d.power_w, '', 1)}/{_fmt(d.power_limit_w, ' W', 0)}",
                    style=heat(
                        (d.power_w / d.power_limit_w * 100)
                        if d.power_w and d.power_limit_w else None,
                        50, 90,
                    ),
                ),
                Text(_fmt(d.temp_c, " C", 0), style=heat(d.temp_c, 60, 84)),
            )
        gpu_panel = Panel(gpu_t, title="GPU (device-level power/temp)", border_style="magenta")
    else:
        gpu_panel = Panel(
            Text(s.gpu.reason, style="dim"),
            title="GPU", border_style="dim",
        )

    # --- Models panel (LLM name informational) ---
    if not s.ollama.reachable:
        models_panel = Panel(
            Text(f"ollama unreachable: {s.ollama.reason}", style="red"),
            title="Resident models", border_style="red",
        )
    elif not s.ollama.running:
        models_panel = Panel(
            Text("none resident — daemon idle", style="dim"),
            title="Resident models", border_style="blue",
        )
    else:
        mt = Table(expand=True, header_style="bold")
        for col in ("Model", "Params", "Quant", "Size", "VRAM", "%GPU", "Expires"):
            mt.add_column(col)
        for m in s.ollama.running:
            mt.add_row(
                Text(m.name, style="bold green"),
                m.param_size,
                m.quant,
                _fmt(m.size_mb, " MB", 0),
                _fmt(m.vram_mb, " MB", 0),
                _fmt(m.pct_gpu, "%", 0),
                _age(m.expires_at),
            )
        models_panel = Panel(mt, title="Resident models (LLM name)", border_style="green")

    # --- Ollama processes panel ---
    if s.ollama_procs:
        pt = Table(expand=True, header_style="bold")
        for col in ("PID", "Process", "CPU", "RSS", "GPU VRAM"):
            pt.add_column(col)
        for p in s.ollama_procs:
            pt.add_row(
                str(p.pid), p.name,
                _fmt(p.cpu_pct, "%", 1),
                _fmt(p.rss_mb, " MB", 0),
                _fmt(p.gpu_vram_mb, " MB", 0),
            )
        proc_panel = Panel(pt, title="Ollama processes", border_style="cyan")
    else:
        proc_panel = Panel(Text("no ollama processes found", style="dim"),
                           title="Ollama processes", border_style="dim")

    # --- Recent turns / tok-per-sec panel ---
    if not s.turns_available:
        turns_panel = Panel(Text(s.turns_reason, style="dim"),
                             title="Recent turns (tok/s)", border_style="dim")
    elif not s.recent_turns:
        turns_panel = Panel(Text("no completed turns yet", style="dim"),
                             title="Recent turns (tok/s)", border_style="dim")
    else:
        tt = Table(expand=True, header_style="bold")
        for col in ("Model", "Tokens", "Elapsed", "tok/s"):
            tt.add_column(col)
        def tps_style(v):
            # Unlike heat(), higher tok/s is better, so the scale is inverted.
            if not v:
                return "dim"
            if v < 5:
                return "bold red"
            if v < 15:
                return "yellow"
            return "green"

        for t in s.recent_turns:
            tt.add_row(
                t.model,
                str(t.completion_tokens),
                _fmt(t.elapsed_s, "s", 1),
                Text(_fmt(t.tokens_per_sec, "", 1), style=tps_style(t.tokens_per_sec)),
            )
        turns_panel = Panel(tt, title="Recent turns (tok/s)", border_style="yellow")

    stamp = datetime.fromtimestamp(s.ts).strftime("%Y-%m-%d %H:%M:%S")
    header = Text(f"  llm-hw-monitor   {stamp}   ollama={s.ollama.base_url}",
                  style="bold white on blue")

    return Group(
        header,
        Panel(sys_t, title="System (CPU power: Linux RAPL only)", border_style="cyan"),
        gpu_panel,
        models_panel,
        proc_panel,
        turns_panel,
    )
