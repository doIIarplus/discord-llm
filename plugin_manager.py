"""Plugin manager for hot-swappable plugin loading/unloading/reloading.

Uses fresh imports (not importlib.reload) to avoid stale reference issues.
Each plugin callback is wrapped in error isolation with auto-disable.
"""

import asyncio
import importlib
import inspect
import logging
import os
import py_compile
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from discord import app_commands

from plugin_base import BasePlugin, HookType, PluginBotContext

logger = logging.getLogger("PluginManager")

PLUGINS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
MAX_FAILURES = 3
CALLBACK_TIMEOUT = 30.0


class PluginManager:
    """Discovers, loads, unloads, and reloads plugins. Dispatches hooks."""

    def __init__(self, bot):
        self.bot = bot
        self._plugins: Dict[str, BasePlugin] = {}       # name -> instance
        self._modules: Dict[str, str] = {}               # name -> module path
        self._failure_counts: Dict[str, int] = {}
        self._disabled: set = set()
        self._hooks: Dict[HookType, List[tuple]] = {h: [] for h in HookType}
        self._pending_sync = False

    # ── Discovery ──────────────────────────────────────────────────────

    def discover_plugins(self) -> List[str]:
        """Scan plugins/ directory for loadable plugins."""
        found = []
        plugins_path = Path(PLUGINS_DIR)
        if not plugins_path.exists():
            return found
        for item in sorted(plugins_path.iterdir()):
            if item.name.startswith("_"):
                continue
            if item.suffix == ".py":
                found.append(item.stem)
            elif item.is_dir() and (item / "__init__.py").exists():
                found.append(item.name)
        return found

    # ── Loading ────────────────────────────────────────────────────────

    async def load_plugin(self, name: str) -> bool:
        """Load a single plugin by name. Returns True on success."""
        success, _ = await self.load_plugin_verbose(name)
        return success

    async def load_plugin_verbose(self, name: str) -> tuple:
        """Load a plugin, returning (success, error_message).

        error_message is None on success, or a string describing the failure.
        """
        if name in self._plugins:
            logger.warning(f"Plugin {name} already loaded, unloading first")
            await self.unload_plugin(name)

        module_name = f"plugins.{name}"

        # Stage 1: Syntax check
        plugin_path = os.path.join(PLUGINS_DIR, f"{name}.py")
        if os.path.isfile(plugin_path):
            try:
                py_compile.compile(plugin_path, doraise=True)
            except py_compile.PyCompileError as e:
                msg = f"Syntax error in plugin {name}: {e}"
                logger.error(msg)
                return False, msg

        # Remove from sys.modules for a completely fresh import
        to_remove = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
        for k in to_remove:
            del sys.modules[k]

        # Stage 2: Import
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            msg = f"Failed to import plugin {name}: {e}"
            logger.error(msg)
            return False, msg

        # Find the BasePlugin subclass
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (inspect.isclass(attr)
                    and issubclass(attr, BasePlugin)
                    and attr is not BasePlugin):
                plugin_class = attr
                break

        if plugin_class is None:
            msg = f"No BasePlugin subclass found in {name}"
            logger.error(msg)
            return False, msg

        # Stage 3: Instantiation + on_load
        ctx = PluginBotContext(self.bot)
        try:
            instance = plugin_class(ctx)
            await instance.on_load()
        except Exception as e:
            msg = f"Plugin {name} failed on_load: {e}"
            logger.error(msg)
            return False, msg

        # Stage 4: self_test
        try:
            passed = await asyncio.wait_for(instance.self_test(), timeout=5.0)
            if not passed:
                msg = f"Plugin {name} self_test returned False"
                logger.error(msg)
                return False, msg
        except asyncio.TimeoutError:
            msg = f"Plugin {name} self_test timed out"
            logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"Plugin {name} self_test raised: {e}"
            logger.error(msg)
            return False, msg

        # Register everything
        self._plugins[name] = instance
        self._modules[name] = module_name
        self._failure_counts[name] = 0
        self._disabled.discard(name)
        self._register_commands(instance)
        self._register_hooks(instance)
        logger.info(f"Loaded plugin: {name} v{instance.version}")
        return True, None

    async def unload_plugin(self, name: str) -> bool:
        """Unload a plugin, removing all its registrations."""
        if name not in self._plugins:
            return False

        instance = self._plugins[name]

        try:
            await instance.on_unload()
        except Exception as e:
            logger.error(f"Plugin {name} on_unload error: {e}")

        self._unregister_commands(instance)
        self._unregister_hooks(name)

        del self._plugins[name]
        self._failure_counts.pop(name, None)
        self._disabled.discard(name)

        # Remove from sys.modules
        module_name = self._modules.pop(name, None)
        if module_name:
            to_remove = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
            for k in to_remove:
                del sys.modules[k]

        logger.info(f"Unloaded plugin: {name}")
        return True

    async def reload_plugin(self, name: str) -> bool:
        """Unload then load a plugin (hot-swap). Returns True on success."""
        success, _ = await self.reload_plugin_verbose(name)
        return success

    async def reload_plugin_verbose(self, name: str) -> tuple:
        """Unload then load a plugin. Returns (success, error_message)."""
        await self.unload_plugin(name)
        return await self.load_plugin_verbose(name)

    async def load_all(self):
        """Discover and load all plugins, respecting dependencies."""
        names = self.discover_plugins()
        if not names:
            logger.info("No plugins found")
            return

        # Simple dependency ordering: load plugins with no deps first,
        # then retry the rest until all loaded or stuck
        loaded = set()
        remaining = list(names)
        max_rounds = len(remaining) + 1

        for _ in range(max_rounds):
            if not remaining:
                break
            still_remaining = []
            for name in remaining:
                # Peek at the module to check deps without full load
                success = await self.load_plugin(name)
                if success:
                    loaded.add(name)
                else:
                    still_remaining.append(name)
            remaining = still_remaining

        if remaining:
            logger.warning(f"Could not load plugins: {remaining}")

        logger.info(f"Loaded {len(loaded)} plugins: {sorted(loaded)}")

    # ── Command registration ───────────────────────────────────────────

    def _register_commands(self, instance: BasePlugin):
        """Register plugin's slash commands with the bot's CommandTree."""
        for cmd_info in instance._slash_commands:
            callback = cmd_info["callback"]
            command = app_commands.Command(
                name=cmd_info["name"],
                description=cmd_info["description"],
                callback=callback,
            )
            # Tag so we can find it later for removal
            command._plugin_name = instance.name
            try:
                self.bot.tree.add_command(command)
            except Exception as e:
                logger.error(f"Failed to register command {cmd_info['name']}: {e}")
        if instance._slash_commands:
            self._pending_sync = True

    def _unregister_commands(self, instance: BasePlugin):
        """Remove plugin's slash commands from the tree."""
        for cmd_info in instance._slash_commands:
            try:
                self.bot.tree.remove_command(cmd_info["name"])
            except Exception:
                pass
        if instance._slash_commands:
            self._pending_sync = True

    # ── Hook registration ──────────────────────────────────────────────

    def _register_hooks(self, instance: BasePlugin):
        """Register plugin's hooks into the dispatch table."""
        for hook_info in instance._hooks:
            hook_type = hook_info["hook_type"]
            entry = (instance.name, hook_info["callback"], instance.priority,
                     hook_info.get("timeout"))
            self._hooks[hook_type].append(entry)
            # Keep sorted by priority (lower = first)
            self._hooks[hook_type].sort(key=lambda x: x[2])

    def _unregister_hooks(self, name: str):
        """Remove all hooks for a given plugin."""
        for hook_type in HookType:
            self._hooks[hook_type] = [
                entry for entry in self._hooks[hook_type]
                if entry[0] != name
            ]

    # ── Dispatch ───────────────────────────────────────────────────────

    async def dispatch_message_handlers(self, message) -> bool:
        """Check message against plugin message handlers.

        Returns True if a plugin consumed the message.
        Handlers are checked in priority order (lower = first).
        """
        # Build a sorted list of all handlers across plugins
        all_handlers = []
        for name, instance in self._plugins.items():
            if name in self._disabled:
                continue
            for handler in instance._message_handlers:
                all_handlers.append((name, handler))
        all_handlers.sort(key=lambda x: x[1].get("priority", 100))

        for plugin_name, handler in all_handlers:
            if re.search(handler["pattern"], message.content, re.IGNORECASE):
                result = await self._safe_call(
                    plugin_name, handler["callback"](message),
                    timeout=handler.get("timeout"),
                )
                if result is True:
                    return True
        return False

    async def dispatch_hook(self, hook_type: HookType, **kwargs) -> List[Any]:
        """Call all registered hooks of a given type, in priority order."""
        results = []
        for entry in self._hooks.get(hook_type, []):
            plugin_name, callback, _priority = entry[0], entry[1], entry[2]
            hook_timeout = entry[3] if len(entry) > 3 else None
            if plugin_name in self._disabled:
                continue
            result = await self._safe_call(
                plugin_name, callback(**kwargs), timeout=hook_timeout,
            )
            if result is not None:
                results.append(result)
        return results

    def should_suppress_text(self, message) -> bool:
        """Check if any plugin wants to suppress text for this message.

        Plugins can define a `suppress_text(message)` method that returns True
        to signal the bot should not send text (e.g. voice mode sends audio instead).
        """
        for name, instance in self._plugins.items():
            if name in self._disabled:
                continue
            if hasattr(instance, "suppress_text"):
                result = instance.suppress_text(message)
                logger.info(f"[TTS-DEBUG] {name}.suppress_text() = {result} "
                            f"for user {message.author.id}")
                if result:
                    return True
        return False

    def get_system_prompt_override(self, user_id: int) -> str | None:
        """Check if any plugin wants to override the system prompt for this user.

        Plugins can define a `get_system_prompt_override(user_id)` method that
        returns a replacement system prompt, or None to use the default.
        """
        for name, instance in self._plugins.items():
            if name in self._disabled:
                continue
            if hasattr(instance, "get_system_prompt_override"):
                override = instance.get_system_prompt_override(user_id)
                if override is not None:
                    return override
        return None

    # ── Error isolation ────────────────────────────────────────────────

    async def _safe_call(self, plugin_name: str, coro, timeout: float = None) -> Any:
        """Execute a plugin coroutine with error containment."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout or CALLBACK_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Plugin {plugin_name} timed out")
            self._record_failure(plugin_name)
        except Exception as e:
            logger.error(f"Plugin {plugin_name} raised: {e}", exc_info=True)
            self._record_failure(plugin_name)
        return None

    def _record_failure(self, name: str):
        """Track consecutive failures; auto-disable after MAX_FAILURES."""
        self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
        if self._failure_counts[name] >= MAX_FAILURES:
            self._disabled.add(name)
            logger.warning(
                f"Plugin {name} auto-disabled after {MAX_FAILURES} consecutive failures"
            )

    def reset_failures(self, name: str):
        """Reset failure count (called on successful callback)."""
        self._failure_counts[name] = 0

    # ── Info ───────────────────────────────────────────────────────────

    def list_plugins(self) -> List[dict]:
        """Return info about all loaded plugins."""
        result = []
        for name, instance in self._plugins.items():
            result.append({
                "name": name,
                "version": instance.version,
                "description": instance.description,
                "disabled": name in self._disabled,
                "failures": self._failure_counts.get(name, 0),
                "commands": [c["name"] for c in instance._slash_commands],
                "message_handlers": len(instance._message_handlers),
                "hooks": len(instance._hooks),
            })
        return result

    @property
    def plugin_names(self) -> List[str]:
        """Names of all loaded plugins."""
        return list(self._plugins.keys())
