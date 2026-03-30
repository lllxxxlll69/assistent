from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
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


class ChatInput(QTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
                event.accept()
        else:
            super().keyPressEvent(event)


class SettingsDialog(QDialog):
    def __init__(self, parent=None, ollama_model=DEFAULT_MODEL, ollama_url=DEFAULT_OLLAMA_URL):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.resize(520, 160)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Модель Ollama:"))
        self.model_edit = QLineEdit(ollama_model)
        layout.addWidget(self.model_edit)

        layout.addWidget(QLabel("URL Ollama:"))
        self.url_edit = QLineEdit(ollama_url)
        layout.addWidget(self.url_edit)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Сохранить")
        self.cancel_button = QPushButton("Отмена")
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.cancel_button)
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

        self.current_assistant_plain = ""
        self.streaming_in_progress = False

        sessions = list_sessions()
        if sessions:
            self.current_session_id = sessions[0][0]
        else:
            self.current_session_id = create_session("Основной чат")

        self.setWindowTitle(APP_NAME)
        self.resize(1100, 720)

        self.build_ui()
        self.refresh_sessions()
        self.load_current_chat()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)
        splitter = QSplitter()
        outer.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.new_chat_button = QPushButton("Новый чат")
        self.settings_button = QPushButton("Настройки")
        self.voice_button = QPushButton("🎤 Голос: выкл")
        self.sessions_list = QListWidget()

        left_layout.addWidget(self.new_chat_button)
        left_layout.addWidget(self.settings_button)
        left_layout.addWidget(self.voice_button)
        left_layout.addWidget(self.sessions_list)

        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.status_label = QLabel("Готово")
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)

        input_row = QHBoxLayout()
        self.input_box = ChatInput()
        self.input_box.setPlaceholderText("Напиши запрос. Enter — отправить, Shift+Enter — новая строка.")
        self.send_button = QPushButton("Отправить")

        input_row.addWidget(self.input_box, 1)
        input_row.addWidget(self.send_button)

        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.chat_view, 1)
        right_layout.addLayout(input_row)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        self.new_chat_button.clicked.connect(self.create_new_chat)
        self.settings_button.clicked.connect(self.open_settings)
        self.voice_button.clicked.connect(self.toggle_voice_mode)
        self.send_button.clicked.connect(self.send_message)
        self.input_box.send_requested.connect(self.send_message)
        self.sessions_list.itemClicked.connect(self.on_session_selected)

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
        self.chat_view.clear()
        history = load_session_history(self.current_session_id)

        for role, content, _created_at in history:
            self.append_message(role, content)

    def _escape_html(self, text):
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    def append_message(self, role, content):
        title = "Ты" if role == "user" else "Ассистент"
        safe = self._escape_html(content)
        self.chat_view.append(f"<p><b>{title}:</b><br>{safe}</p>")
        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def start_streaming_assistant_message(self):
        self.streaming_in_progress = True
        self.current_assistant_plain = ""
        self.chat_view.append("<p><b>Ассистент:</b><br></p>")

    def update_streaming_assistant_message(self, chunk):
        if not self.streaming_in_progress:
            self.start_streaming_assistant_message()

        self.current_assistant_plain += chunk
        safe = self._escape_html(self.current_assistant_plain)

        cursor = self.chat_view.textCursor()
        cursor.movePosition(cursor.End)
        cursor.select(cursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        cursor.insertHtml(f"<p><b>Ассистент:</b><br>{safe}</p>")

        self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    def finish_streaming_assistant_message(self):
        self.streaming_in_progress = False

    def create_new_chat(self):
        self.current_session_id = create_session()
        self.refresh_sessions()
        self.chat_view.clear()
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

    def send_message_text(self, text):
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

    def on_worker_status(self, text):
        self.status_label.setText(text)

    def on_worker_stream_chunk(self, chunk):
        self.update_streaming_assistant_message(chunk)

    def on_worker_finished(self, answer):
        if self.streaming_in_progress:
            self.finish_streaming_assistant_message()
            final_text = self.current_assistant_plain
        else:
            final_text = answer
            self.append_message("assistant", final_text)

        save_message(self.current_session_id, "assistant", final_text)
        self.status_label.setText("Готово")
        self.worker = None

    def on_worker_failed(self, error_text):
        self.streaming_in_progress = False
        self.status_label.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка", error_text)
        self.worker = None

    def toggle_voice_mode(self):
        if self.voice_assistant and self.voice_assistant.isRunning():
            self.voice_assistant.stop()
            self.voice_assistant.wait()
            self.voice_assistant = None
            self.voice_button.setText("🎤 Голос: выкл")
            self.status_label.setText("Голосовой режим выключен")
            return

        self.voice_assistant = VoiceAssistant(
            session_id=self.current_session_id,
            ollama_url=self.ollama_url,
            model_name=self.model_name,
        )
        self.voice_assistant.status.connect(self.on_voice_status)
        self.voice_assistant.error.connect(self.on_voice_error)
        self.voice_assistant.recognized_text.connect(self.on_voice_recognized)
        self.voice_assistant.assistant_text.connect(self.on_voice_answer)
        self.voice_assistant.start()

        self.voice_button.setText("🎤 Голос: вкл")
        self.status_label.setText("Запускаю голосовой режим...")

    def on_voice_status(self, text):
        self.status_label.setText(text)

    def on_voice_error(self, text):
        QMessageBox.critical(self, "Голосовой режим", text)
        self.status_label.setText("Ошибка голосового режима")
        if self.voice_assistant:
            self.voice_assistant.stop()
            self.voice_assistant = None
        self.voice_button.setText("🎤 Голос: выкл")

    def on_voice_recognized(self, text):
        save_message(self.current_session_id, "user", text)
        self.append_message("user", text)
        rename_session_if_needed(self.current_session_id, text)
        self.refresh_sessions()

    def on_voice_answer(self, text):
        save_message(self.current_session_id, "assistant", text)
        self.append_message("assistant", text)

    def closeEvent(self, event):
        if self.voice_assistant and self.voice_assistant.isRunning():
            self.voice_assistant.stop()
            self.voice_assistant.wait()
        if self.worker and self.worker.isRunning():
            self.worker.wait()
        super().closeEvent(event)
