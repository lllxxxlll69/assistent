from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QDateTime, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from assistant.app import AssistantBackend, build_backend
from assistant.config.settings import Settings
from assistant.llm.client import LLMClientError
from assistant.models import AssistantResponse, utc_now_iso


COLORS = {
    "bg_main": "#16181d",
    "bg_panel": "#20242b",
    "bg_block": "#2b313b",
    "bg_accent": "#334155",
    "bg_accent_soft": "#293343",
    "border": "#475569",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
}


class AutoResizeTextEdit(QTextEdit):
    send_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.document().contentsChanged.connect(self.update_height)
        self._min_height = 56
        self._max_height = 180
        self.update_height()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
                event.accept()
            return
        super().keyPressEvent(event)

    def update_height(self) -> None:
        doc_height = self.document().size().height()
        margins = self.contentsMargins().top() + self.contentsMargins().bottom()
        frame = self.frameWidth() * 2
        new_height = int(doc_height + margins + frame + 14)
        new_height = max(self._min_height, min(self._max_height, new_height))
        self.setFixedHeight(new_height)

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self.update_height()


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(4)

        role_label = QLabel("Вы" if role == "user" else "Ассистент")
        role_label.setObjectName("messageRole")

        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_label.setObjectName("messageText")

        outer.addWidget(role_label)
        outer.addWidget(self.text_label)
        self.setObjectName("messageBubble")
        self.setProperty("messageType", role)

    def set_text(self, text: str) -> None:
        self._text = text
        self.text_label.setText(text)

    def append_text(self, chunk: str) -> None:
        self.set_text(self._text + chunk)

    def plain_text(self) -> str:
        return self._text


class ChatRow(QWidget):
    def __init__(self, role: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)

        self.bubble = MessageBubble(role=role, text=text)
        self.bubble.setMinimumWidth(240)
        self.bubble.setMaximumWidth(720)

        if role == "user":
            layout.addStretch(1)
            layout.addWidget(self.bubble, 0, Qt.AlignRight | Qt.AlignTop)
        else:
            layout.addWidget(self.bubble, 0, Qt.AlignLeft | Qt.AlignTop)
            layout.addStretch(1)

    def append_text(self, chunk: str) -> None:
        self.bubble.append_text(chunk)

    def plain_text(self) -> str:
        return self.bubble.plain_text()


class ChatView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.viewport().setObjectName("chatViewport")

        self.container = QWidget()
        self.container.setObjectName("chatContainer")
        self.messages_layout = QVBoxLayout(self.container)
        self.messages_layout.setContentsMargins(0, 10, 0, 10)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch(1)
        self._streaming_row: ChatRow | None = None

        self.scroll_area.setWidget(self.container)
        layout.addWidget(self.scroll_area)

    def clear_messages(self) -> None:
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._streaming_row = None

    def add_message(self, role: str, text: str) -> None:
        row = ChatRow(role=role, text=text)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, row)
        QTimer.singleShot(0, self.scroll_to_bottom)

    def start_streaming_assistant_message(self) -> None:
        if self._streaming_row is not None:
            return
        self._streaming_row = ChatRow(role="assistant", text="")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, self._streaming_row)
        QTimer.singleShot(0, self.scroll_to_bottom)

    def append_streaming_chunk(self, chunk: str) -> None:
        if self._streaming_row is None:
            self.start_streaming_assistant_message()
        if self._streaming_row is not None:
            self._streaming_row.append_text(chunk)
        QTimer.singleShot(0, self.scroll_to_bottom)

    def finish_streaming_message(self) -> str:
        if self._streaming_row is None:
            return ""
        text = self._streaming_row.plain_text()
        self._streaming_row = None
        return text

    def abort_streaming_message(self) -> None:
        if self._streaming_row is not None:
            self.messages_layout.removeWidget(self._streaming_row)
            self._streaming_row.deleteLater()
        self._streaming_row = None

    def scroll_to_bottom(self) -> None:
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())


class AssistantWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    chunk = Signal(str)

    def __init__(self, backend: AssistantBackend, prompt: str, assistant_mode: str) -> None:
        super().__init__()
        self.backend = backend
        self.prompt = prompt
        self.assistant_mode = assistant_mode

    def run(self) -> None:  # type: ignore[override]
        try:
            response = asyncio.run(
                self.backend.orchestrator.handle_with_callbacks(
                    self.prompt,
                    on_text_chunk=self.chunk.emit,
                    assistant_profile=self.assistant_mode,
                )
            )
            self.completed.emit(response)
        except LLMClientError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class WarmupWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, backend: AssistantBackend) -> None:
        super().__init__()
        self.backend = backend

    def run(self) -> None:  # type: ignore[override]
        try:
            timings = self.backend.orchestrator.warm_up_models()
            self.completed.emit(timings)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: Settings,
        memory_stats: dict[str, int],
        memory_summary: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.defaults = Settings()
        self.setWindowTitle("Настройки")
        self.setModal(True)
        self.resize(740, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Настройки ассистента")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)

        connection_box = QGroupBox("Модель и подключение")
        connection_form = QFormLayout(connection_box)
        self.model_edit = QLineEdit(settings.model)
        self.vision_model_edit = QLineEdit(settings.vision_model)
        self.api_url_edit = QLineEdit(settings.api_url)
        self.search_root_edit = QLineEdit(settings.search_root)
        connection_form.addRow("Chat model", self.model_edit)
        connection_form.addRow("Vision model", self.vision_model_edit)
        connection_form.addRow("API URL", self.api_url_edit)
        connection_form.addRow("Search root", self.search_root_edit)
        layout.addWidget(connection_box)

        generation_box = QGroupBox("Генерация и контекст")
        generation_form = QFormLayout(generation_box)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 16000)
        self.context_size_spin = QSpinBox()
        self.context_size_spin.setRange(512, 32768)
        self.context_size_spin.setSingleStep(512)
        self.memory_length_spin = QSpinBox()
        self.memory_length_spin.setRange(1, 200)
        self.memory_max_tokens_spin = QSpinBox()
        self.memory_max_tokens_spin.setRange(256, 32000)
        self.memory_max_tokens_spin.setSingleStep(256)
        self.max_search_results_spin = QSpinBox()
        self.max_search_results_spin.setRange(1, 25)
        self.search_chunk_size_spin = QSpinBox()
        self.search_chunk_size_spin.setRange(200, 4000)
        self.search_chunk_size_spin.setSingleStep(100)
        self.request_timeout_spin = QSpinBox()
        self.request_timeout_spin.setRange(10, 600)
        self.stream_check = QCheckBox("Показывать ответ по мере генерации")
        self.show_logs_check = QCheckBox("Показывать окно логов")

        generation_form.addRow("Temperature", self.temperature_spin)
        generation_form.addRow("Max tokens", self.max_tokens_spin)
        generation_form.addRow("Context size", self.context_size_spin)
        generation_form.addRow("Memory length", self.memory_length_spin)
        generation_form.addRow("Memory max tokens", self.memory_max_tokens_spin)
        generation_form.addRow("Search results", self.max_search_results_spin)
        generation_form.addRow("Chunk size", self.search_chunk_size_spin)
        generation_form.addRow("Timeout (sec)", self.request_timeout_spin)
        generation_form.addRow("", self.stream_check)
        generation_form.addRow("", self.show_logs_check)
        layout.addWidget(generation_box)

        stats_box = QGroupBox("Память и контекст")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.addWidget(
            QLabel(
                "Сообщений: {stored_messages} | "
                "Оценка токенов: {stored_tokens_estimate} | "
                "В контексте сейчас: {context_messages} | "
                "Чатов: {session_count}".format(**memory_stats)
            )
        )
        self.summary_preview = QPlainTextEdit(memory_summary or "Сводка пока не нужна.")
        self.summary_preview.setReadOnly(True)
        self.summary_preview.setMinimumHeight(120)
        stats_layout.addWidget(self.summary_preview)
        layout.addWidget(stats_box)

        buttons = QHBoxLayout()
        self.reset_button = QPushButton("Сброс")
        cancel_button = QPushButton("Отмена")
        save_button = QPushButton("Сохранить")
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)
        layout.addLayout(buttons)

        self.reset_button.clicked.connect(self._reset_form)
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self.accept)
        self._fill_form(settings)

    def _fill_form(self, settings: Settings) -> None:
        self.model_edit.setText(settings.model)
        self.vision_model_edit.setText(settings.vision_model)
        self.api_url_edit.setText(settings.api_url)
        self.search_root_edit.setText(settings.search_root)
        self.temperature_spin.setValue(settings.temperature)
        self.max_tokens_spin.setValue(settings.max_tokens)
        self.context_size_spin.setValue(settings.context_size)
        self.memory_length_spin.setValue(settings.memory_length)
        self.memory_max_tokens_spin.setValue(settings.memory_max_tokens)
        self.max_search_results_spin.setValue(settings.max_search_results)
        self.search_chunk_size_spin.setValue(settings.search_chunk_size)
        self.request_timeout_spin.setValue(settings.request_timeout)
        self.stream_check.setChecked(settings.stream)
        self.show_logs_check.setChecked(settings.show_logs)

    def _reset_form(self) -> None:
        answer = QMessageBox.question(
            self,
            "Сброс настроек",
            "Вернуть настройки к значениям по умолчанию?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._fill_form(self.defaults)
        self.accept()

    def get_values(self) -> dict[str, object]:
        return {
            "model": self.model_edit.text().strip(),
            "vision_model": self.vision_model_edit.text().strip(),
            "api_url": self.api_url_edit.text().strip(),
            "search_root": self.search_root_edit.text().strip() or ".",
            "temperature": self.temperature_spin.value(),
            "max_tokens": self.max_tokens_spin.value(),
            "context_size": self.context_size_spin.value(),
            "memory_length": self.memory_length_spin.value(),
            "memory_max_tokens": self.memory_max_tokens_spin.value(),
            "max_search_results": self.max_search_results_spin.value(),
            "search_chunk_size": self.search_chunk_size_spin.value(),
            "request_timeout": self.request_timeout_spin.value(),
            "stream": self.stream_check.isChecked(),
            "show_logs": self.show_logs_check.isChecked(),
        }


class AssistantWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.backend = build_backend()
        self.worker: AssistantWorker | None = None
        self.warmup_worker: WarmupWorker | None = None
        self.last_assistant_message = ""
        self.feedback_path = self.backend.settings_manager.settings_path.parent / "feedback.log"
        self.runtime_log_path = self.backend.settings_manager.settings_path.parent / "app.log"
        self.request_started_at: float | None = None
        self.stream_started = False
        self.response_update_timer = QTimer(self)
        self.response_update_timer.timeout.connect(self._update_response_time_label)

        self.setWindowTitle("AI Assistant")
        self.resize(1360, 900)
        self.setMinimumSize(1120, 720)

        self._build_ui()
        self._apply_styles()
        self._start_clock()
        self._refresh_sessions()
        self._load_current_session()
        self._refresh_runtime_labels()
        self._load_runtime_logs()
        self._apply_log_panel_visibility(self.backend.settings_manager.get_settings().show_logs)
        self._append_log("INFO", "Приложение запущено")

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("centralRoot")
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        app_title = QLabel("AI Assistant")
        app_title.setObjectName("appTitle")
        self.context_label = QLabel("")
        self.context_label.setObjectName("infoLabel")
        self.model_label = QLabel("")
        self.model_label.setObjectName("infoLabel")
        self.clock_label = QLabel("")
        self.clock_label.setObjectName("infoLabel")

        chats_title = QLabel("Чаты")
        chats_title.setObjectName("sectionTitle")
        self.sessions_list = QListWidget()
        self.sessions_list.setObjectName("sessionsList")
        self.sessions_list.setMouseTracking(True)
        self.sessions_list.setContextMenuPolicy(Qt.CustomContextMenu)

        self.new_chat_button = QPushButton("Новый чат")
        self.image_button = QPushButton("Анализ изображения")
        self.settings_button = QPushButton("Настройки")
        self.warmup_button = QPushButton("Прогреть модели")

        sidebar_layout.addWidget(app_title)
        sidebar_layout.addWidget(self.context_label)
        sidebar_layout.addWidget(self.model_label)
        sidebar_layout.addWidget(self.clock_label)
        sidebar_layout.addSpacing(8)
        sidebar_layout.addWidget(chats_title)
        sidebar_layout.addWidget(self.sessions_list, 1)
        sidebar_layout.addWidget(self.new_chat_button)
        sidebar_layout.addWidget(self.image_button)
        sidebar_layout.addWidget(self.settings_button)
        sidebar_layout.addWidget(self.warmup_button)

        self.content = QFrame()
        self.content.setObjectName("contentPanel")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Готово")
        self.status_label.setObjectName("topStatus")
        self.response_time_label = QLabel("Ответ: --")
        self.response_time_label.setObjectName("infoLabel")
        top_bar.addWidget(self.status_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self.response_time_label)

        mode_bar = QFrame()
        mode_bar.setObjectName("modeBar")
        mode_layout = QHBoxLayout(mode_bar)
        mode_layout.setContentsMargins(12, 10, 12, 10)
        mode_layout.setSpacing(10)

        mode_title = QLabel("Режим чата")
        mode_title.setObjectName("sectionTitle")
        self.mode_hint_label = QLabel("")
        self.mode_hint_label.setObjectName("infoLabel")
        self.lua_mode_button = QPushButton("Lua-код")
        self.lua_mode_button.setObjectName("modeToggleButton")
        self.lua_mode_button.setCheckable(True)
        self.chat_mode_button = QPushButton("Чат-бот")
        self.chat_mode_button.setObjectName("modeToggleButton")
        self.chat_mode_button.setCheckable(True)

        mode_layout.addWidget(mode_title)
        mode_layout.addSpacing(8)
        mode_layout.addWidget(self.lua_mode_button)
        mode_layout.addWidget(self.chat_mode_button)
        mode_layout.addStretch(1)
        mode_layout.addWidget(self.mode_hint_label)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        chat_panel = QFrame()
        chat_panel.setObjectName("chatPanel")
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(12)

        self.chat_view = ChatView()
        chat_layout.addWidget(self.chat_view, 1)

        composer = QFrame()
        composer.setObjectName("composerPanel")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(0, 0, 0, 0)
        composer_layout.setSpacing(12)

        self.input_box = AutoResizeTextEdit()
        self.input_box.setPlaceholderText(
            "Напишите задачу. Enter отправит сообщение, Shift+Enter добавит новую строку."
        )
        self.send_button = QPushButton("Отправить")
        self.send_button.setFixedWidth(170)
        self.send_button.setFixedHeight(44)

        composer_layout.addWidget(self.input_box, 1, Qt.AlignBottom)
        composer_layout.addWidget(self.send_button, 0, Qt.AlignBottom)
        chat_layout.addWidget(composer)

        self.side_panel = QFrame()
        self.side_panel.setObjectName("sidePanel")
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)

        logs_title = QLabel("Журнал действий")
        logs_title.setObjectName("sectionTitle")
        self.logs_list = QListWidget()
        self.logs_list.setObjectName("logsList")

        feedback_row = QHBoxLayout()
        self.like_button = QPushButton("Полезно")
        self.dislike_button = QPushButton("Не полезно")
        feedback_row.addWidget(self.like_button)
        feedback_row.addWidget(self.dislike_button)

        self.details_box = QPlainTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setPlaceholderText("Здесь видны метрики и служебная информация последнего ответа.")

        side_layout.addWidget(logs_title)
        side_layout.addWidget(self.logs_list, 1)
        side_layout.addLayout(feedback_row)
        side_layout.addWidget(self.details_box)

        self.main_splitter.addWidget(chat_panel)
        self.main_splitter.addWidget(self.side_panel)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([900, 340])

        content_layout.addLayout(top_bar)
        content_layout.addWidget(mode_bar)
        content_layout.addWidget(self.main_splitter, 1)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.content, 1)

        self.new_chat_button.clicked.connect(self._start_new_chat)
        self.image_button.clicked.connect(self._analyze_image)
        self.settings_button.clicked.connect(self._open_settings)
        self.warmup_button.clicked.connect(self._warm_up_models)
        self.send_button.clicked.connect(self._send_message)
        self.input_box.send_requested.connect(self._send_message)
        self.lua_mode_button.clicked.connect(lambda: self._switch_session_mode("localscript"))
        self.chat_mode_button.clicked.connect(lambda: self._switch_session_mode("chat"))
        self.like_button.clicked.connect(lambda: self._save_feedback(True))
        self.dislike_button.clicked.connect(lambda: self._save_feedback(False))
        self.sessions_list.itemClicked.connect(self._open_selected_session)
        self.sessions_list.customContextMenuRequested.connect(self._open_sessions_context_menu)
        self.logs_list.currentItemChanged.connect(self._show_log_details)

    def _apply_styles(self) -> None:
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            f"""
            QWidget {{
                color: {COLORS["text"]};
            }}
            QMainWindow, QDialog, QWidget#centralRoot {{
                background: {COLORS["bg_main"]};
            }}
            QLabel {{
                background: transparent;
            }}
            QFrame#sidebar, QFrame#contentPanel {{
                background: transparent;
                border: none;
            }}
            QFrame#chatPanel, QFrame#sidePanel {{
                background: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 14px;
            }}
            QFrame#modeBar {{
                background: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 14px;
            }}
            QFrame#composerPanel {{
                background: transparent;
                border: none;
            }}
            QWidget#chatRow, QWidget#chatContainer, QWidget#chatViewport {{
                background: transparent;
            }}
            QFrame#messageBubble {{
                border-radius: 12px;
                border: 1px solid {COLORS["border"]};
                background: {COLORS["bg_block"]};
            }}
            QFrame#messageBubble[messageType="user"] {{
                background: {COLORS["bg_accent"]};
            }}
            QLabel#appTitle {{
                font-size: 22px;
                font-weight: 700;
                padding: 8px 4px 14px 4px;
            }}
            QLabel#sectionTitle {{
                font-size: 16px;
                font-weight: 600;
                padding-top: 4px;
                padding-bottom: 4px;
            }}
            QLabel#messageRole {{
                color: {COLORS["muted"]};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#messageText {{
                font-size: 15px;
            }}
            QLabel#topStatus {{
                font-size: 15px;
                font-weight: 600;
            }}
            QLabel#infoLabel {{
                color: {COLORS["muted"]};
                font-size: 13px;
            }}
            QLabel#dialogTitle {{
                font-size: 20px;
                font-weight: 700;
            }}
            QPushButton {{
                background: {COLORS["bg_block"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 10px;
                padding: 11px 14px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {COLORS["bg_accent"]};
            }}
            QPushButton#modeToggleButton {{
                min-width: 118px;
                padding: 9px 14px;
            }}
            QPushButton#modeToggleButton:checked {{
                background: {COLORS["bg_accent"]};
                border: 1px solid {COLORS["text"]};
                font-weight: 700;
            }}
            QPushButton:disabled {{
                color: {COLORS["muted"]};
            }}
            QTextEdit, QLineEdit, QPlainTextEdit, QListWidget, QSpinBox, QDoubleSpinBox {{
                background: {COLORS["bg_block"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 10px;
                padding: 8px;
                font-size: 14px;
            }}
            QListWidget#sessionsList::item {{
                padding: 10px 12px;
                border-radius: 8px;
                margin: 2px 0;
                background: transparent;
            }}
            QListWidget#sessionsList::item:hover {{
                background: {COLORS["bg_accent_soft"]};
            }}
            QListWidget#sessionsList::item:selected {{
                background: {COLORS["bg_accent"]};
            }}
            QGroupBox {{
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                margin-top: 14px;
                padding-top: 14px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QMenu {{
                background: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 10px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 16px;
                border-radius: 8px;
            }}
            QMenu::item:selected {{
                background: {COLORS["bg_accent"]};
            }}
            """
        )

    def _start_clock(self) -> None:
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

    def _update_clock(self) -> None:
        self.clock_label.setText(QDateTime.currentDateTime().toString("dd.MM.yyyy HH:mm"))

    def _refresh_sessions(self) -> None:
        active_id = self.backend.memory_manager.get_active_session_id()
        self.sessions_list.clear()
        for session in self.backend.memory_manager.list_sessions():
            item = QListWidgetItem(session.title)
            item.setData(Qt.UserRole, session.id)
            item.setToolTip(f"{session.title}\nРежим: {self._assistant_mode_title(session.assistant_mode)}")
            self.sessions_list.addItem(item)
            if session.id == active_id:
                self.sessions_list.setCurrentItem(item)

    def _load_current_session(self) -> None:
        self.stream_started = False
        self.chat_view.clear_messages()
        for message in self.backend.memory_manager.get_all_messages():
            self.chat_view.add_message(message.role, message.content)
        self.chat_view.scroll_to_bottom()
        self._update_mode_controls()

    def _refresh_runtime_labels(self) -> None:
        settings = self.backend.settings_manager.get_settings()
        stats = self.backend.memory_manager.get_stats()
        session = self.backend.memory_manager.get_current_session()
        self.context_label.setText(
            f"Контекст: {stats['context_messages']} сообщений / {settings.memory_max_tokens} токенов"
        )
        self.model_label.setText(f"Chat: {settings.model} | Vision: {settings.vision_model}")
        self.status_label.setToolTip(
            f"Текущий чат: {session.title}\nРежим: {self._assistant_mode_title(session.assistant_mode)}"
        )

    def _assistant_mode_title(self, assistant_mode: str) -> str:
        return "Чат-бот" if assistant_mode == "chat" else "Lua-код"

    def _assistant_mode_hint(self, assistant_mode: str) -> str:
        if assistant_mode == "chat":
            return "Обычный AI-ассистент на русском языке"
        return "Генерация и доработка LocalScript / Lua"

    def _current_assistant_mode(self) -> str:
        return self.backend.memory_manager.get_active_session_mode()

    def _update_mode_controls(self) -> None:
        mode = self._current_assistant_mode()
        self.lua_mode_button.setChecked(mode == "localscript")
        self.chat_mode_button.setChecked(mode == "chat")
        self.mode_hint_label.setText(self._assistant_mode_hint(mode))
        if mode == "chat":
            self.input_box.setPlaceholderText(
                "Напишите сообщение ассистенту. Enter отправит сообщение, Shift+Enter добавит новую строку."
            )
        else:
            self.input_box.setPlaceholderText(
                "Опишите задачу для генерации или доработки LocalScript/Lua. Enter отправит сообщение, Shift+Enter добавит новую строку."
            )

    def _switch_session_mode(self, assistant_mode: str) -> None:
        if self._is_busy():
            QMessageBox.information(self, "Подождите", "Сменить режим можно после завершения текущего запроса.")
            self._update_mode_controls()
            return

        normalized = "chat" if assistant_mode == "chat" else "localscript"
        session_id = self.backend.memory_manager.get_active_session_id()
        current_mode = self._current_assistant_mode()
        if current_mode == normalized:
            self._update_mode_controls()
            return

        if not self.backend.memory_manager.set_session_mode(session_id, normalized):
            self._update_mode_controls()
            return

        session = self.backend.memory_manager.get_current_session()
        mode_title = self._assistant_mode_title(normalized)
        self._update_mode_controls()
        self._refresh_sessions()
        self._refresh_runtime_labels()
        self.status_label.setText(f"Режим чата: {mode_title}")
        self._append_log("INFO", f"Режим чата переключен: {session.title} -> {mode_title}")

    def _apply_log_panel_visibility(self, visible: bool) -> None:
        self.side_panel.setVisible(visible)
        if visible:
            self.main_splitter.setSizes([900, 340])
        else:
            self.main_splitter.setSizes([1240, 0])

    def _load_runtime_logs(self) -> None:
        if not self.runtime_log_path.exists():
            return
        try:
            lines = self.runtime_log_path.read_text(encoding="utf-8").splitlines()[-300:]
        except OSError:
            return
        for line in reversed(lines):
            if not line.startswith("["):
                continue
            item = QListWidgetItem(line)
            item.setData(Qt.UserRole, line)
            self.logs_list.addItem(item)
        if self.logs_list.count() > 0:
            self.logs_list.setCurrentRow(0)

    def _append_log(self, level: str, message: str, *, details: str | None = None) -> None:
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        entry = f"[{timestamp}] {level} | {message}"
        item = QListWidgetItem(entry)
        item.setData(Qt.UserRole, details or entry)
        self.logs_list.insertItem(0, item)
        while self.logs_list.count() > 500:
            self.logs_list.takeItem(self.logs_list.count() - 1)
        self.logs_list.setCurrentRow(0)
        if details is not None:
            self.details_box.setPlainText(details)

        self.runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.runtime_log_path.open("a", encoding="utf-8") as stream:
                stream.write(entry + "\n")
        except OSError:
            pass

    def _show_log_details(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        details = current.data(Qt.UserRole)
        if details:
            self.details_box.setPlainText(str(details))

    def _send_message(self) -> None:
        prompt = self.input_box.toPlainText().strip()
        if not prompt:
            return
        self._send_prompt(prompt)

    def _send_prompt(self, prompt: str) -> None:
        if self._is_busy():
            self._append_log("WARN", "Попытка отправить новый запрос во время активной операции")
            QMessageBox.information(self, "Подождите", "Предыдущий запрос ещё выполняется.")
            return

        assistant_mode = self._current_assistant_mode()
        self.chat_view.add_message("user", prompt)
        self.input_box.clear()
        self.status_label.setText("Ожидаю ответ модели...")
        self._append_log(
            "INFO",
            f"Запрос отправлен модели ({self._assistant_mode_title(assistant_mode)})",
            details=prompt,
        )
        self._set_controls_enabled(False)
        self._start_response_timer()
        self.stream_started = False

        self.worker = AssistantWorker(self.backend, prompt, assistant_mode)
        self.worker.chunk.connect(self._on_response_chunk)
        self.worker.completed.connect(self._on_response_ready)
        self.worker.failed.connect(self._on_response_failed)
        self.worker.start()

    def _on_response_ready(self, response: AssistantResponse) -> None:
        self.worker = None
        if self.stream_started:
            streamed_text = self.chat_view.finish_streaming_message()
            if streamed_text.strip() and streamed_text.strip() != response.text.strip():
                self.chat_view.add_message("assistant", response.text)
        else:
            self.chat_view.add_message("assistant", response.text)
        self.stream_started = False
        self.last_assistant_message = response.text

        for log in response.logs:
            prefix = "OK" if log.success else "ERR"
            self._append_log(prefix, log.message)

        duration = self._stop_response_timer()
        response.metrics["response_seconds"] = round(duration, 2)
        details = json.dumps(response.metrics, ensure_ascii=False, indent=2) + f"\n\nПоследний ответ:\n{response.text}"
        self.details_box.setPlainText(details)
        self._append_log("OK", f"Ответ получен за {duration:.2f} s", details=details)
        self.status_label.setText("Готово")
        self._set_controls_enabled(True)
        self._refresh_sessions()
        self._refresh_runtime_labels()

    def _on_response_chunk(self, chunk: str) -> None:
        if not self.stream_started:
            self.chat_view.start_streaming_assistant_message()
            self.stream_started = True
            self._append_log("INFO", "Начат потоковый вывод ответа")
        self.chat_view.append_streaming_chunk(chunk)

    def _on_response_failed(self, error_text: str) -> None:
        self.worker = None
        self.chat_view.abort_streaming_message()
        self.stream_started = False
        duration = self._stop_response_timer()
        self.response_time_label.setText(f"Ответ: ошибка через {duration:.1f} s")
        self.status_label.setText("Ошибка")
        self._append_log("ERR", f"Ошибка запроса через {duration:.2f} s", details=error_text)
        self._set_controls_enabled(True)
        self._refresh_sessions()
        self._refresh_runtime_labels()
        QMessageBox.critical(self, "Ошибка ассистента", error_text)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.send_button.setEnabled(enabled)
        self.lua_mode_button.setEnabled(enabled)
        self.chat_mode_button.setEnabled(enabled)
        self.new_chat_button.setEnabled(enabled)
        self.image_button.setEnabled(enabled)
        self.settings_button.setEnabled(enabled)
        self.warmup_button.setEnabled(enabled)
        self.sessions_list.setEnabled(enabled)

    def _is_busy(self) -> bool:
        return (
            (self.worker is not None and self.worker.isRunning())
            or (self.warmup_worker is not None and self.warmup_worker.isRunning())
        )

    def _start_new_chat(self) -> None:
        if self._is_busy():
            QMessageBox.information(self, "Подождите", "Сначала дождитесь завершения текущего запроса.")
            return

        session = self.backend.memory_manager.create_session()
        self.stream_started = False
        self.last_assistant_message = ""
        self._refresh_sessions()
        self._load_current_session()
        self.status_label.setText(f"Создан {session.title}")
        self._append_log(
            "INFO",
            f"Создан новый чат: {session.title} ({self._assistant_mode_title(session.assistant_mode)})",
        )
        self._refresh_runtime_labels()

    def _open_selected_session(self, item: QListWidgetItem) -> None:
        if self._is_busy():
            return
        session_id = item.data(Qt.UserRole)
        if session_id and self.backend.memory_manager.set_active_session(session_id):
            self.stream_started = False
            self.last_assistant_message = ""
            self._load_current_session()
            self._refresh_sessions()
            self._refresh_runtime_labels()
            self.status_label.setText("Чат загружен")
            self._append_log(
                "INFO",
                f"Открыт чат: {item.text()} ({self._assistant_mode_title(self._current_assistant_mode())})",
            )

    def _open_sessions_context_menu(self, position) -> None:
        item = self.sessions_list.itemAt(position)
        if item is None or self._is_busy():
            return

        self.sessions_list.setCurrentItem(item)
        session_id = item.data(Qt.UserRole)
        menu = QMenu(self)
        rename_action = menu.addAction("Переименовать")
        delete_action = menu.addAction("Удалить")
        chosen = menu.exec(self.sessions_list.mapToGlobal(position))

        if chosen == rename_action:
            self._rename_session(session_id)
        elif chosen == delete_action:
            self._delete_session(session_id)

    def _rename_session(self, session_id: str) -> None:
        session = next((item for item in self.backend.memory_manager.list_sessions() if item.id == session_id), None)
        if session is None:
            return

        old_title = session.title
        new_title, ok = QInputDialog.getText(self, "Переименовать чат", "Новое название:", text=session.title)
        if not ok:
            return
        if not self.backend.memory_manager.rename_session(session_id, new_title):
            QMessageBox.warning(self, "Переименование", "Название не должно быть пустым.")
            return

        self._refresh_sessions()
        self._refresh_runtime_labels()
        self._append_log("INFO", f"Чат переименован: {old_title} -> {new_title.strip()[:80]}")

    def _delete_session(self, session_id: str) -> None:
        answer = QMessageBox.question(
            self,
            "Удаление чата",
            "Удалить выбранный чат?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self.backend.memory_manager.delete_session(session_id)
        self.stream_started = False
        self.last_assistant_message = ""
        self._refresh_sessions()
        self._load_current_session()
        self._refresh_runtime_labels()
        self.status_label.setText("Чат удалён")
        self._append_log("INFO", "Чат удалён")

    def _analyze_image(self) -> None:
        if self._is_busy():
            QMessageBox.information(self, "Подождите", "Сначала дождитесь завершения текущего запроса.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if not file_path:
            return

        prompt, ok = QInputDialog.getText(
            self,
            "Промпт для изображения",
            "Что сделать с изображением?",
            text="Что изображено на картинке?",
        )
        if not ok or not prompt.strip():
            return

        self._append_log("INFO", f"Выбрано изображение для анализа: {file_path}", details=prompt.strip())
        self._send_prompt(f"{prompt.strip()} {file_path}")

    def _open_settings(self) -> None:
        if self._is_busy():
            QMessageBox.information(self, "Подождите", "Менять настройки лучше после завершения текущего запроса.")
            return

        settings = self.backend.settings_manager.get_settings()
        stats = self.backend.memory_manager.get_stats()
        summary = self.backend.memory_manager.summarize_context()
        dialog = SettingsDialog(settings=settings, memory_stats=stats, memory_summary=summary, parent=self)
        self._append_log("INFO", "Открыто окно настроек")
        if dialog.exec() != QDialog.Accepted:
            self._append_log("INFO", "Настройки закрыты без сохранения")
            return

        values = dialog.get_values()
        if not values["model"] or not values["vision_model"] or not values["api_url"]:
            QMessageBox.warning(self, "Настройки", "Заполните model, vision model и API URL.")
            return

        updated_settings = self.backend.settings_manager.update_settings(**values)
        active_session_id = self.backend.memory_manager.get_active_session_id()
        self.backend = build_backend(self.backend.settings_manager)
        self.backend.memory_manager.set_active_session(active_session_id)
        self._apply_log_panel_visibility(updated_settings.show_logs)
        self.status_label.setText("Настройки сохранены")
        self._append_log(
            "INFO",
            f"Настройки сохранены. Окно логов: {'включено' if updated_settings.show_logs else 'выключено'}",
            details=json.dumps(values, ensure_ascii=False, indent=2),
        )
        self._refresh_sessions()
        self._load_current_session()
        self._refresh_runtime_labels()

    def _save_feedback(self, positive: bool) -> None:
        if not self.last_assistant_message:
            QMessageBox.information(self, "Фидбек", "Сначала получите ответ ассистента.")
            return

        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": utc_now_iso(),
            "session_id": self.backend.memory_manager.get_active_session_id(),
            "positive": positive,
            "message": self.last_assistant_message,
        }
        with self.feedback_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self.status_label.setText("Фидбек сохранён")
        self._append_log("INFO", f"Сохранён фидбек: {'положительный' if positive else 'отрицательный'}")

    def _warm_up_models(self) -> None:
        if self._is_busy():
            QMessageBox.information(self, "Подождите", "Сначала дождитесь завершения текущей операции.")
            return

        self.status_label.setText("Прогреваю модели...")
        self._append_log("INFO", "Запущен прогрев моделей")
        self._set_controls_enabled(False)
        self.warmup_worker = WarmupWorker(self.backend)
        self.warmup_worker.completed.connect(self._on_warmup_ready)
        self.warmup_worker.failed.connect(self._on_warmup_failed)
        self.warmup_worker.start()

    def _on_warmup_ready(self, timings: dict[str, float]) -> None:
        self.warmup_worker = None
        self._set_controls_enabled(True)
        summary = "\n".join(f"{model}: {seconds:.2f} s" for model, seconds in timings.items())
        self.details_box.setPlainText(f"Warm-up timings:\n{summary}")
        self.status_label.setText("Модели прогреты")
        self._append_log("OK", "Модели прогреты", details=summary)

    def _on_warmup_failed(self, error_text: str) -> None:
        self.warmup_worker = None
        self._set_controls_enabled(True)
        self.status_label.setText("Ошибка прогрева")
        self._append_log("ERR", "Ошибка прогрева моделей", details=error_text)
        QMessageBox.critical(self, "Прогрев моделей", error_text)

    def _start_response_timer(self) -> None:
        self.request_started_at = perf_counter()
        self.response_time_label.setText("Ответ: 0.0 s")
        self.response_update_timer.start(100)

    def _update_response_time_label(self) -> None:
        if self.request_started_at is None:
            self.response_time_label.setText("Ответ: --")
            return
        elapsed = perf_counter() - self.request_started_at
        self.response_time_label.setText(f"Ответ: {elapsed:.1f} s")

    def _stop_response_timer(self) -> float:
        self.response_update_timer.stop()
        if self.request_started_at is None:
            self.response_time_label.setText("Ответ: --")
            return 0.0
        elapsed = perf_counter() - self.request_started_at
        self.request_started_at = None
        self.response_time_label.setText(f"Ответ: {elapsed:.1f} s")
        return elapsed

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._append_log("INFO", "Приложение закрывается")
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        if self.warmup_worker is not None and self.warmup_worker.isRunning():
            self.warmup_worker.wait()
        super().closeEvent(event)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("AI Assistant")
    window = AssistantWindow()
    window.show()
    app.exec()
