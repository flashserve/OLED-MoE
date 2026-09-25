"""Local OpenAI-compatible chat server."""

__all__ = ["ServerSettings", "create_app"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from .app import ServerSettings, create_app

        exports = {
            "ServerSettings": ServerSettings,
            "create_app": create_app,
        }
        return exports[name]
    raise AttributeError(name)
