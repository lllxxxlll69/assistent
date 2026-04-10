from __future__ import annotations

from assistant.models import Message, RetrievalChunk


def build_system_prompt(language: str = "ru") -> str:
    if language.lower().startswith("ru"):
        return (
            "Ты AI assistant для разработки. "
            "Отвечай точно, кратко и по делу. "
            "Учитывай память и найденный локальный контекст. "
            "После действий с файлами или изображением дай полезный итог."
        )

    return (
        "You are a development assistant. "
        "Be concise, accurate, and action-oriented. "
        "Use memory and retrieved local context when relevant."
    )


def build_chat_messages(
    system_prompt: str,
    context_messages: list[Message],
    user_input: str,
    retrieval_chunks: list[RetrievalChunk] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if retrieval_chunks:
        joined_chunks = "\n\n".join(chunk.to_prompt_block() for chunk in retrieval_chunks)
        messages.append({"role": "system", "content": f"Relevant local context:\n{joined_chunks}"})

    messages.extend({"role": item.role, "content": item.content} for item in context_messages)
    messages.append({"role": "user", "content": user_input})
    return messages


def build_auto_mode_prompt(task: str) -> str:
    return (
        "Составь короткий исполнимый план для задачи ниже. "
        "Верни нумерованный список шагов без лишнего текста.\n\n"
        f"Задача: {task}"
    )


def build_vision_prompt(user_prompt: str) -> str:
    return (
        "Проанализируй изображение и верни полезный структурированный ответ. "
        "Сначала дай короткое summary, потом ключевые наблюдения.\n\n"
        f"Инструкция пользователя: {user_prompt}"
    )
