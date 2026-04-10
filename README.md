# Modular Python AI Assistant

Production-style AI assistant with:

- desktop window UI on PySide6
- file system tools
- vision/image analysis via vision-capable LLM
- memory with context summarization
- JSON-based settings
- local retrieval over project files
- auto mode for simple scaffold tasks

## Project Structure

```text
assistant/
  __init__.py
  __main__.py
  models.py
  config/
    __init__.py
    settings.py
  core/
    __init__.py
    agent.py
    orchestrator.py
  llm/
    __init__.py
    client.py
    prompts.py
  memory/
    __init__.py
    memory_manager.py
  tools/
    __init__.py
    file_tools.py
    search_tools.py
    vision_tools.py
  ui/
    __init__.py
    window.py
main.py
```

## Run

```bash
python main.py
```

The application opens as a normal desktop window with:

- chat panel
- action log panel
- settings dialog for model, context size, memory length, token budget, search root, and timeout
- image analysis button
- persistent local history in `.assistant_data/`

To use vision, choose an image through the GUI and enter a prompt like:

- `Что на этом изображении?`
- `Исправь ошибку со скриншота`
- `Объясни диаграмму`
