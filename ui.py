from __future__ import annotations

from PySide6.QtCore import QDateTime, QTimer, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import APP_NAME, DEFAULT_MODEL, DEFAULT_OLLAMA_URL
from storage import (
    create_session,
    ensure_db,
    get_setting,
    list_sessions,
    load_session_history,
    rename_session_if_needed,
    save_message,
    set_setting,
)
from voice_assistant import VoiceAssistant
from worker import AssistantWorker


COLORS = {
    "bg_main": "#1E1E1E",
    "bg_panel": "#2D2D2D",
    "bg_block": "#393939",
    "border": "#555555",
    "text": "#D9D9D9",
}


class AutoResizeTextEdit(QTextEdit):
    send_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.document().contentsChanged.connect(self.update_height)

        self._min_height = 54
        self._max_height = 180
        self.update_height()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
                event.accept()
            return
        super().keyPressEvent(event)

    def update_height(self):
        doc_height = self.document().size().height()
        margins = self.contentsMargins().top() + self.contentsMargins().bottom()
        frame = self.frameWidth() * 2
        new_height = int(doc_height + margins + frame + 14)

        if new_height < self._min_height:
            new_height = self._min_height
        if new_height > self._max_height:
            new_height = self._max_height

        self.setFixedHeight(new_height)

    def clear(self):
        super().clear()
        self.update_height()


class MessageBubble(QFrame):
    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self.role = role
        self._plain_text = text

        self.setObjectName("messageBubble")
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(4)

        self.role_label = QLabel("Вы:" if role == "user" else "Ассистент:")
        self.role_label.setObjectName("messageRole")
        self.role_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.role_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.text_label = QLabel()
        self.text_label.setObjectName("messageText")
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.PlainText)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.text_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        outer.addWidget(self.role_label)
        outer.addWidget(self.text_label)

        self.set_text(text)

    def set_text(self, text: str):
        self._plain_text = text
        self.text_label.setText(text)
        self.text_label.adjustSize()
        self.adjustSize()

    def append_text(self, chunk: str):
        self.set_text(self._plain_text + chunk)

    def plain_text(self) -> str:
        return self._plain_text


class ChatMessageRow(QWidget):
    def __init__(self, role: str, text: str = "", parent=None):
        super().__init__(parent)
        self.role = role

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(0)

        self.bubble = MessageBubble(role=role, text=text)
        self.bubble.setMinimumWidth(180)
        self.bubble.setMaximumWidth(560)

        if role == "user":
            layout.addStretch(1)
            layout.addWidget(self.bubble, 0, Qt.AlignRight | Qt.AlignTop)
        else:
            layout.addWidget(self.bubble, 0, Qt.AlignLeft | Qt.AlignTop)
            layout.addStretch(1)

    def set_text(self, text: str):
        self.bubble.set_text(text)

    def append_text(self, chunk: str):
        self.bubble.append_text(chunk)

    def plain_text(self) -> str:
        return self.bubble.plain_text()


class ChatView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.messages_layout = QVBoxLayout(self.container)
        self.messages_layout.setContentsMargins(0, 8, 0, 8)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch(1)

        self.scroll_area.setWidget(self.container)
        root.addWidget(self.scroll_area)

        self._streaming_row: ChatMessageRow | None = None

    def clear_messages(self):
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._streaming_row = None

    def add_message(self, role: str, text: str):
        row = ChatMessageRow(role=role, text=text)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, row)
        QTimer.singleShot(0, self.scroll_to_bottom)
        return row

    def start_streaming_assistant_message(self):
        self._streaming_row = self.add_message("assistant", "")
        return self._streaming_row

    def update_streaming_assistant_message(self, chunk: str):
        if self._streaming_row is None:
            self.start_streaming_assistant_message()
        self._streaming_row.append_text(chunk)
        QTimer.singleShot(0, self.scroll_to_bottom)

    def finish_streaming_assistant_message(self) -> str:
        if self._streaming_row is None:
            return ""
        text = self._streaming_row.plain_text()
        self._streaming_row = None
        return text

    def scroll_to_bottom(self):
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())


class SettingsDialog(QDialog):
    def __init__(self, parent=None, ollama_model=DEFAULT_MODEL, ollama_url=DEFAULT_OLLAMA_URL):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.setModal(True)
        self.resize(520, 190)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("Настройки")
        title.setObjectName("dialogTitle")

        model_label = QLabel("Модель Ollama")
        self.model_edit = QLineEdit(ollama_model)

        url_label = QLabel("URL Ollama")
        self.url_edit = QLineEdit(ollama_url)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        self.save_button = QPushButton("Сохранить")
        self.cancel_button = QPushButton("Отмена")

        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.save_button)

        layout.addWidget(title)
        layout.addWidget(model_label)
        layout.addWidget(self.model_edit)
        layout.addWidget(url_label)
        layout.addWidget(self.url_edit)
        layout.addStretch(1)
        layout.addLayout(buttons)

        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_values(self):
        return self.model_edit.text().strip(), self.url_edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        ensure_db()

        self.model_name = get_setting("model_name", DEFAULT_MODEL)
        self.ollama_url = get_setting("ollama_url", DEFAULT_OLLAMA_URL)

        self.worker = None
        self.voice_assistant = None
        self.voice_stop_in_progress = False

        self.streaming_in_progress = False
        self.current_assistant_plain = ""

        sessions = list_sessions()
        if sessions:
            self.current_session_id = sessions[0][0]
        else:
            self.current_session_id = create_session("Основной чат")

        self.setWindowTitle(APP_NAME)
        self.resize(1180, 760)
        self.setMinimumSize(980, 620)

        self.build_ui()
        self.apply_styles()
        self.refresh_sessions()
        self.load_current_chat()
        self.start_clock()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(195)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        self.new_chat_button = QPushButton("Новый чат")
        self.settings_button = QPushButton("Настройки")
        self.voice_button = QPushButton("Голос: вкл")

        self.sessions_list = QListWidget()
        self.sessions_list.setObjectName("sessionsList")

        sidebar_layout.addWidget(self.new_chat_button)
        sidebar_layout.addWidget(self.settings_button)
        sidebar_layout.addWidget(self.voice_button)
        sidebar_layout.addWidget(self.sessions_list, 1)

        self.content = QFrame()
        self.content.setObjectName("contentPanel")

        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(10)

        self.status_label = QLabel("Готово")
        self.status_label.setObjectName("topStatus")

        self.time_label = QLabel("")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_bar.addWidget(self.status_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self.time_label)

        self.chat_frame = QFrame()
        self.chat_frame.setObjectName("chatFrame")

        chat_frame_layout = QVBoxLayout(self.chat_frame)
        chat_frame_layout.setContentsMargins(0, 0, 0, 0)
        chat_frame_layout.setSpacing(0)

        self.chat_view = ChatView()
        chat_frame_layout.addWidget(self.chat_view)

        bottom_panel = QFrame()
        bottom_panel.setObjectName("inputPanel")

        bottom_layout = QHBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(12)

        self.input_box = AutoResizeTextEdit()
        self.input_box.setPlaceholderText("Введите запрос...")

        self.send_button = QPushButton("Отправить")
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedWidth(155)
        self.send_button.setFixedHeight(42)

        bottom_layout.addWidget(self.input_box, 1, Qt.AlignBottom)
        bottom_layout.addWidget(self.send_button, 0, Qt.AlignBottom)

        content_layout.addLayout(top_bar)
        content_layout.addWidget(self.chat_frame, 1)
        content_layout.addWidget(bottom_panel)

        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.content, 1)

        self.new_chat_button.clicked.connect(self.create_new_chat)
        self.settings_button.clicked.connect(self.open_settings)
        self.voice_button.clicked.connect(self.toggle_voice_mode)
        self.send_button.clicked.connect(self.send_message)
        self.input_box.send_requested.connect(self.send_message)
        self.sessions_list.itemClicked.connect(self.on_session_selected)

    def apply_styles(self):
        font = QFont("Segoe UI", 10)
        self.setFont(font)

        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {COLORS["bg_main"]};
                color: {COLORS["text"]};
            }}

            QMainWindow {{
                background-color: {COLORS["bg_main"]};
            }}

            QFrame#sidebar {{
                background-color: transparent;
                border: none;
            }}

            QFrame#contentPanel {{
                background-color: transparent;
                border: none;
            }}

            QPushButton {{
                background-color: {COLORS["bg_panel"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 15px;
            }}

            QPushButton:hover {{
                background-color: {COLORS["bg_block"]};
            }}

            QPushButton:pressed {{
                background-color: {COLORS["border"]};
            }}

            QPushButton#sendButton {{
                background-color: {COLORS["bg_block"]};
                border-radius: 8px;
                padding: 8px 14px;
                font-size: 15px;
            }}

            QLabel#topStatus {{
                font-size: 15px;
                color: {COLORS["text"]};
                padding-left: 2px;
            }}

            QLabel#timeLabel {{
                font-size: 15px;
                color: {COLORS["text"]};
                padding-right: 2px;
            }}

            QFrame#chatFrame {{
                background-color: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
            }}

            QFrame#inputPanel {{
                background-color: transparent;
                border: none;
            }}

            QTextEdit {{
                background-color: {COLORS["bg_panel"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                padding: 10px;
                font-size: 15px;
            }}

            QScrollArea {{
                background-color: transparent;
                border: none;
            }}

            QListWidget#sessionsList {{
                background-color: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                padding: 4px;
                outline: none;
                font-size: 14px;
            }}

            QListWidget#sessionsList::item {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: 8px 6px;
                margin: 2px;
            }}

            QListWidget#sessionsList::item:selected {{
                background-color: {COLORS["bg_block"]};
                color: {COLORS["text"]};
            }}

            QListWidget#sessionsList::item:hover {{
                background-color: {COLORS["bg_block"]};
            }}

            QScrollBar:vertical {{
                background: transparent;
                width: 10px;
                margin: 4px 2px 4px 2px;
            }}

            QScrollBar::handle:vertical {{
                background: {COLORS["border"]};
                border-radius: 4px;
                min-height: 24px;
            }}

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
                background: transparent;
                border: none;
            }}

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}

            QFrame#messageBubble {{
                background-color: {COLORS["bg_block"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
            }}

            QLabel#messageRole {{
                color: {COLORS["text"]};
                font-size: 13px;
                background: transparent;
            }}

            QLabel#messageText {{
                color: {COLORS["text"]};
                font-size: 15px;
                background: transparent;
            }}

            QDialog {{
                background-color: {COLORS["bg_main"]};
                color: {COLORS["text"]};
            }}

            QLabel#dialogTitle {{
                font-size: 18px;
                font-weight: 600;
                padding-bottom: 6px;
            }}

            QLineEdit {{
                background-color: {COLORS["bg_panel"]};
                color: {COLORS["text"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }}
            """
        )

    def start_clock(self):
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_time)
        self.clock_timer.start(1000)
        self.update_time()

    def update_time(self):
        current = QDateTime.currentDateTime().toString("HH:mm")
        self.time_label.setText(current)

    def refresh_sessions(self):
        self.sessions_list.clear()
        sessions = list_sessions()

        for session_id, title in sessions:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, session_id)
            self.sessions_list.addItem(item)

            if session_id == self.current_session_id:
                self.sessions_list.setCurrentItem(item)

    def load_current_chat(self):
        self.chat_view.clear_messages()
        history = load_session_history(self.current_session_id)

        for role, content, _created_at in history:
            self.append_message(role, content)

        self.chat_view.scroll_to_bottom()

    def append_message(self, role: str, content: str):
        self.chat_view.add_message(role, content)

    def start_streaming_assistant_message(self):
        self.streaming_in_progress = True
        self.current_assistant_plain = ""
        self.chat_view.start_streaming_assistant_message()

    def update_streaming_assistant_message(self, chunk: str):
        if not self.streaming_in_progress:
            self.start_streaming_assistant_message()

        self.current_assistant_plain += chunk
        self.chat_view.update_streaming_assistant_message(chunk)

    def finish_streaming_assistant_message(self):
        self.streaming_in_progress = False
        self.chat_view.finish_streaming_assistant_message()

    def create_new_chat(self):
        self.current_session_id = create_session()
        self.refresh_sessions()
        self.chat_view.clear_messages()
        self.status_label.setText("Новый чат создан")

    def on_session_selected(self, item):
        session_id = item.data(Qt.UserRole)
        if not session_id:
            return

        self.current_session_id = session_id
        self.load_current_chat()
        self.status_label.setText("Чат загружен")

    def open_settings(self):
        dialog = SettingsDialog(
            self,
            ollama_model=self.model_name,
            ollama_url=self.ollama_url,
        )

        if dialog.exec():
            model_name, ollama_url = dialog.get_values()

            if not model_name or not ollama_url:
                QMessageBox.warning(self, "Настройки", "Заполни модель и URL.")
                return

            self.model_name = model_name
            self.ollama_url = ollama_url

            set_setting("model_name", self.model_name)
            set_setting("ollama_url", self.ollama_url)

            self.status_label.setText("Настройки сохранены")

    def send_message(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        self.send_message_text(text)

    def send_message_text(self, text: str):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Подожди", "Предыдущий запрос ещё обрабатывается.")
            return

        rename_session_if_needed(self.current_session_id, text)
        self.refresh_sessions()

        save_message(self.current_session_id, "user", text)
        self.append_message("user", text)

        self.input_box.clear()
        self.status_label.setText("Отправляю запрос...")

        self.worker = AssistantWorker(
            session_id=self.current_session_id,
            user_text=text,
            model_name=self.model_name,
            ollama_url=self.ollama_url,
            speak_enabled=False,
        )

        self.worker.status.connect(self.on_worker_status)
        self.worker.stream_chunk.connect(self.on_worker_stream_chunk)
        self.worker.finished_ok.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.start()

    def on_worker_status(self, text: str):
        self.status_label.setText(text)

    def on_worker_stream_chunk(self, chunk: str):
        self.update_streaming_assistant_message(chunk)

    def on_worker_finished(self, answer: str):
        if self.streaming_in_progress:
            self.finish_streaming_assistant_message()
            final_text = self.current_assistant_plain
        else:
            final_text = answer
            self.append_message("assistant", final_text)

        save_message(self.current_session_id, "assistant", final_text)
        self.status_label.setText("Готово")
        self.worker = None

    def on_worker_failed(self, error_text: str):
        self.streaming_in_progress = False
        self.current_assistant_plain = ""
        self.status_label.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка", error_text)
        self.worker = None

    def toggle_voice_mode(self):
        if self.voice_assistant is not None and self.voice_assistant.isRunning():
            self.voice_stop_in_progress = True
            self.voice_button.setEnabled(False)
            self.voice_button.setText("Голос: выключается...")
            self.status_label.setText("Останавливаю голосовой режим...")
            self.voice_assistant.stop()
            return

        if self.voice_assistant is not None:
            self.voice_assistant.deleteLater()
            self.voice_assistant = None

        self.voice_stop_in_progress = False

        self.voice_assistant = VoiceAssistant(
            session_id=self.current_session_id,
            ollama_url=self.ollama_url,
            model_name=self.model_name,
        )
        self.voice_assistant.status.connect(self.on_voice_status)
        self.voice_assistant.error.connect(self.on_voice_error)
        self.voice_assistant.recognized_text.connect(self.on_voice_recognized)
        self.voice_assistant.assistant_text.connect(self.on_voice_answer)
        self.voice_assistant.finished.connect(self.on_voice_thread_finished)
        self.voice_assistant.start()

        self.voice_button.setEnabled(True)
        self.voice_button.setText("Голос: вкл")
        self.status_label.setText("Запускаю голосовой режим...")

    def on_voice_status(self, text: str):
        self.status_label.setText(text)

    def on_voice_error(self, text: str):
        QMessageBox.critical(self, "Голосовой режим", text)
        self.status_label.setText("Ошибка голосового режима")
        self.voice_stop_in_progress = True
        if self.voice_assistant:
            self.voice_assistant.stop()
        self.voice_button.setEnabled(True)
        self.voice_button.setText("Голос: выкл")

    def on_voice_thread_finished(self):
        if self.voice_assistant:
            self.voice_assistant.deleteLater()
            self.voice_assistant = None

        self.voice_button.setEnabled(True)
        self.voice_button.setText("Голос: выкл")

        if self.voice_stop_in_progress:
            self.status_label.setText("Голосовой режим выключен")
        elif self.status_label.text() == "Запускаю голосовой режим...":
            self.status_label.setText("Голосовой режим выключен")

        self.voice_stop_in_progress = False

    def on_voice_recognized(self, text: str):
        save_message(self.current_session_id, "user", text)
        self.append_message("user", text)
        rename_session_if_needed(self.current_session_id, text)
        self.refresh_sessions()

    def on_voice_answer(self, text: str):
        save_message(self.current_session_id, "assistant", text)
        self.append_message("assistant", text)

    def closeEvent(self, event):
        if self.voice_assistant and self.voice_assistant.isRunning():
            self.voice_assistant.stop()
            self.voice_assistant.wait()

        if self.worker and self.worker.isRunning():
            self.worker.wait()

        super().closeEvent(event)