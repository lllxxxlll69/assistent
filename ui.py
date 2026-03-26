import sqlite3
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import APP_NAME, AUTO_START_VOICE, DEFAULT_MODEL, DEFAULT_OLLAMA_URL, DB_PATH
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
from voice_listener import VoiceListener, get_input_devices, test_microphone_levels
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
    def __init__(
        self,
        parent=None,
        ollama_model=DEFAULT_MODEL,
        ollama_url=DEFAULT_OLLAMA_URL,
        speak=False,
        input_device=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.resize(560, 300)

        self.model_edit = QLineEdit(ollama_model)
        self.url_edit = QLineEdit(ollama_url)

        self.speak_checkbox = QCheckBox("Озвучивать ответы")
        self.speak_checkbox.setChecked(speak)

        self.input_device_combo = QComboBox()
        self.input_device_combo.addItem("Система по умолчанию", None)
        for idx, name in get_input_devices():
            self.input_device_combo.addItem(f"[{idx}] {name}", idx)

        self._set_combo_value(self.input_device_combo, input_device)

        form = QFormLayout()
        form.addRow("Ollama model:", self.model_edit)
        form.addRow("Ollama URL:", self.url_edit)
        form.addRow("Микрофон:", self.input_device_combo)
        form.addRow("", self.speak_checkbox)

        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(self.accept)

        test_button = QPushButton("Проверить микрофон")
        test_button.clicked.connect(self.test_microphone)

        buttons = QHBoxLayout()
        buttons.addWidget(test_button)
        buttons.addWidget(save_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def _set_combo_value(self, combo, value):
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def values(self):
        return (
            self.model_edit.text().strip(),
            self.url_edit.text().strip(),
            self.speak_checkbox.isChecked(),
            self.input_device_combo.currentData(),
        )

    def test_microphone(self):
        device = self.input_device_combo.currentData()
        try:
            mean_level, max_level = test_microphone_levels(device=device, seconds=3)

            if max_level < 0.01:
                QMessageBox.warning(
                    self,
                    "Проверка микрофона",
                    f"Сигнал почти не слышен.\n\n"
                    f"Средний уровень: {mean_level:.5f}\n"
                    f"Максимальный уровень: {max_level:.5f}\n\n"
                    f"Попробуй выбрать другой микрофон или сказать что-нибудь громче."
                )
            else:
                QMessageBox.information(
                    self,
                    "Проверка микрофона",
                    f"Микрофон работает.\n\n"
                    f"Средний уровень: {mean_level:.5f}\n"
                    f"Максимальный уровень: {max_level:.5f}"
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Проверка микрофона",
                f"Не удалось проверить микрофон:\n{exc}"
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_db()

        self.model_name = get_setting("model_name", DEFAULT_MODEL)
        self.ollama_url = get_setting("ollama_url", DEFAULT_OLLAMA_URL)
        self.speak_enabled = get_setting("speak_enabled", "0") == "1"

        raw_input_device = get_setting("input_device", "")
        self.input_device = int(raw_input_device) if raw_input_device not in ("", "None") else None

        self.worker = None
        self.voice_listener = None
        self.current_session_id = None

        self.setWindowTitle(APP_NAME)
        self.resize(1040, 760)

        self.build_ui()
        self.build_menu()
        self.reload_sessions(select_last=False)

        self.voice_listener = self._make_voice_listener()
        if AUTO_START_VOICE:
            self.voice_listener.start()

    def build_ui(self):
        self.chat_selector = QComboBox()
        self.chat_selector.currentIndexChanged.connect(self.on_session_changed)

        self.new_chat_button = QPushButton("Новый сценарий")
        self.new_chat_button.clicked.connect(self.create_new_chat)

        self.voice_button = QPushButton("Голос: вкл" if AUTO_START_VOICE else "Голос: выкл")
        self.voice_button.clicked.connect(self.toggle_voice)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Чат:"))
        top_row.addWidget(self.chat_selector)
        top_row.addWidget(self.new_chat_button)
        top_row.addWidget(self.voice_button)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setPlaceholderText("Здесь будет история диалога...")

        self.input = ChatInput()
        self.input.setPlaceholderText(
            "Напиши запрос. Например: открой сайт openai, "
            "запусти блокнот, открой файл report.pdf, "
            "открой папку Downloads, какая сегодня погода в Хельсинки"
        )
        self.input.setFixedHeight(110)
        self.input.send_requested.connect(self.send_message)

        self.status_label = QLabel("Готов")
        self.status_label.setAlignment(Qt.AlignLeft)

        send_button = QPushButton("Отправить")
        send_button.clicked.connect(self.send_message)

        clear_button = QPushButton("Очистить текущий чат")
        clear_button.clicked.connect(self.clear_chat)

        buttons_row = QHBoxLayout()
        buttons_row.addWidget(send_button)
        buttons_row.addWidget(clear_button)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.chat)
        layout.addWidget(QLabel("Ввод:"))
        layout.addWidget(self.input)
        layout.addLayout(buttons_row)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def build_menu(self):
        menu = self.menuBar()
        settings_menu = menu.addMenu("Настройки")

        settings_action = QAction("Параметры", self)
        settings_action.triggered.connect(self.open_settings)
        settings_menu.addAction(settings_action)

    def _make_voice_listener(self):
        listener = VoiceListener(input_device=self.input_device)
        listener.heard_command.connect(self.on_voice_command)
        listener.heard_text.connect(self.on_voice_text)
        listener.status.connect(self.status_label.setText)
        listener.failed.connect(self.on_voice_error)
        return listener

    def reload_sessions(self, select_last=True):
        sessions = list_sessions()

        self.chat_selector.blockSignals(True)
        self.chat_selector.clear()
        for session_id, title in sessions:
            self.chat_selector.addItem(title, session_id)
        self.chat_selector.blockSignals(False)

        if sessions:
            if self.current_session_id:
                index = self.chat_selector.findData(self.current_session_id)
                if index >= 0:
                    self.chat_selector.setCurrentIndex(index)
                elif select_last:
                    self.chat_selector.setCurrentIndex(len(sessions) - 1)
                else:
                    self.chat_selector.setCurrentIndex(0)
            else:
                self.chat_selector.setCurrentIndex(len(sessions) - 1 if select_last else 0)

            self.current_session_id = self.chat_selector.currentData()
            self.load_history()
        else:
            self.current_session_id = None
            self.chat.clear()

    def create_new_chat(self):
        self.current_session_id = create_session()
        self.reload_sessions(select_last=True)
        self.input.setFocus()
        self.status_label.setText("Создан новый сценарий")

    def on_session_changed(self):
        session_id = self.chat_selector.currentData()
        if session_id:
            self.current_session_id = session_id
            self.load_history()

    def append_chat(self, role, text):
        if not self.current_session_id:
            self.current_session_id = create_session()

        time_str = datetime.now().strftime("%H:%M:%S")
        prefix = "Вы" if role == "user" else "Ассистент"

        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_text = safe_text.replace("\n", "<br>")

        self.chat.append(f"<b>[{time_str}] {prefix}:</b><br>{safe_text}<br>")

        scrollbar = self.chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        save_message(self.current_session_id, role, text)

    def load_history(self):
        if not self.current_session_id:
            self.chat.clear()
            return

        rows = load_session_history(self.current_session_id, limit=1000)
        self.chat.clear()

        for role, content, created_at in rows:
            prefix = "Вы" if role == "user" else "Ассистент"
            time_str = created_at.split("T")[-1] if "T" in created_at else created_at

            safe_text = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe_text = safe_text.replace("\n", "<br>")

            self.chat.append(f"<b>[{time_str}] {prefix}:</b><br>{safe_text}<br>")

        scrollbar = self.chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_chat(self):
        if not self.current_session_id:
            return

        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM messages WHERE session_id = ?", (self.current_session_id,))
        conn.commit()
        conn.close()

        self.chat.clear()
        self.status_label.setText("Текущий чат очищен")

    def open_settings(self):
        dlg = SettingsDialog(
            self,
            self.model_name,
            self.ollama_url,
            self.speak_enabled,
            self.input_device,
        )

        if dlg.exec():
            self.model_name, self.ollama_url, self.speak_enabled, self.input_device = dlg.values()

            set_setting("model_name", self.model_name)
            set_setting("ollama_url", self.ollama_url)
            set_setting("speak_enabled", "1" if self.speak_enabled else "0")
            set_setting("input_device", "" if self.input_device is None else str(self.input_device))

            if self.voice_listener and self.voice_listener.isRunning():
                self.voice_listener.stop()
                self.voice_listener.wait(2000)

            self.voice_listener = self._make_voice_listener()
            self.voice_listener.start()
            self.voice_button.setText("Голос: вкл")
            self.status_label.setText("Настройки сохранены и голос перезапущен")

    def send_message(self):
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Подожди, предыдущая команда ещё выполняется")
            return

        if not self.current_session_id:
            self.create_new_chat()

        text = self.input.toPlainText().strip()
        if not text:
            QMessageBox.information(self, APP_NAME, "Введите запрос.")
            return

        self.input.clear()
        self.append_chat("user", text)

        rename_session_if_needed(self.current_session_id, text)
        self.reload_sessions(select_last=False)

        index = self.chat_selector.findData(self.current_session_id)
        if index >= 0:
            self.chat_selector.setCurrentIndex(index)

        self.status_label.setText("Выполняю...")

        self.worker = AssistantWorker(
            self.current_session_id,
            text,
            self.model_name,
            self.ollama_url,
            self.speak_enabled,
        )
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self.on_answer)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_answer(self, text):
        self.append_chat("assistant", text)
        self.status_label.setText("Готов")
        self.worker = None

    def on_error(self, error_text):
        self.append_chat("assistant", f"Ошибка: {error_text}")
        self.status_label.setText("Ошибка")
        self.worker = None

    def toggle_voice(self):
        if self.voice_listener and self.voice_listener.isRunning():
            self.voice_button.setEnabled(False)
            self.voice_button.setText("Голос: выключается...")
            self.voice_listener.stop()
            self.voice_listener.wait(2000)
            self.voice_listener = self._make_voice_listener()
            self.voice_button.setEnabled(True)
            self.voice_button.setText("Голос: выкл")
            self.status_label.setText("Голосовой режим выключен")
            return

        self.voice_listener = self._make_voice_listener()
        self.voice_listener.start()
        self.voice_button.setText("Голос: вкл")
        self.status_label.setText("Голосовой режим включён")

    def on_voice_command(self, text):
        if self.worker and self.worker.isRunning():
            self.status_label.setText("Команда услышана, но ассистент ещё занят")
            return

        self.input.setPlainText(text)
        self.send_message()

    def on_voice_text(self, text):
        self.status_label.setText(f"Слышу: {text}")

    def on_voice_error(self, error_text):
        self.voice_button.setText("Голос: выкл")
        self.status_label.setText(f"Ошибка голоса: {error_text}")
        QMessageBox.warning(self, APP_NAME, f"Ошибка голосового режима:\n{error_text}")

    def closeEvent(self, event):
        if self.voice_listener and self.voice_listener.isRunning():
            self.voice_listener.stop()
            self.voice_listener.wait(1500)
        event.accept()