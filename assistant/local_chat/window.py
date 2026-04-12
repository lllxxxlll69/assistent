from __future__ import annotations

import socket
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from assistant.local_chat.core import (
    DEFAULT_DISCOVERY_PORT,
    DEFAULT_TCP_PORT,
    LanChatMessenger,
    format_mac,
    short_hash,
)


COLORS = {
    "bg_main": "#222222",
    "bg_panel": "#1b1b1b",
    "bg_block": "#1b1b1b",
    "bg_accent": "#222222",
    "border": "#2a2a2a",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
}


class ManualPeerDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Добавить пира")
        self.setModal(True)
        self.resize(420, 220)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.mac_edit = QLineEdit()
        self.ip_edit = QLineEdit()
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(DEFAULT_TCP_PORT)
        self.name_edit = QLineEdit()
        form.addRow("MAC", self.mac_edit)
        form.addRow("IP", self.ip_edit)
        form.addRow("TCP порт", self.port_spin)
        form.addRow("Имя", self.name_edit)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel_button = QPushButton("Отмена")
        save_button = QPushButton("Сохранить")
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)
        layout.addLayout(buttons)

        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self.accept)

    def values(self) -> dict[str, object]:
        return {
            "mac": self.mac_edit.text().strip(),
            "ip": self.ip_edit.text().strip(),
            "port": self.port_spin.value(),
            "name": self.name_edit.text().strip(),
        }


class LocalChatWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Локальный чат между пользователями")
        self.resize(1080, 760)
        self.setMinimumSize(920, 640)

        self.messenger: LanChatMessenger | None = None
        self._selected_peer_mac: str | None = None

        self._build_ui()
        self._apply_styles()
        self._refresh_me()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_messenger)
        self.poll_timer.start(500)

        QTimer.singleShot(0, self._start_chat)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Локальный P2P чат")
        title.setObjectName("title")
        subtitle = QLabel("Отдельный пользовательский чат в локальной сети. Эти сообщения не попадают в ИИ.")
        subtitle.setObjectName("subtitle")

        config_card = QFrame()
        config_card.setObjectName("card")
        config_layout = QGridLayout(config_card)
        config_layout.setContentsMargins(12, 12, 12, 12)
        config_layout.setHorizontalSpacing(10)
        config_layout.setVerticalSpacing(8)

        self.name_edit = QLineEdit(socket.gethostname())
        self.mac_override_edit = QLineEdit()
        self.mac_override_edit.setPlaceholderText("AA:BB:CC:DD:EE:FF")
        self.tcp_port_spin = QSpinBox()
        self.tcp_port_spin.setRange(1, 65535)
        self.tcp_port_spin.setValue(DEFAULT_TCP_PORT)
        self.discovery_port_spin = QSpinBox()
        self.discovery_port_spin.setRange(1, 65535)
        self.discovery_port_spin.setValue(DEFAULT_DISCOVERY_PORT)
        self.start_button = QPushButton("Запустить")
        self.stop_button = QPushButton("Остановить")
        self.discover_button = QPushButton("Обнаружить")
        self.link_button = QPushButton("Добавить пира")

        config_layout.addWidget(QLabel("Имя узла"), 0, 0)
        config_layout.addWidget(self.name_edit, 0, 1)
        config_layout.addWidget(QLabel("MAC override"), 0, 2)
        config_layout.addWidget(self.mac_override_edit, 0, 3)
        config_layout.addWidget(QLabel("TCP порт"), 1, 0)
        config_layout.addWidget(self.tcp_port_spin, 1, 1)
        config_layout.addWidget(QLabel("Discovery порт"), 1, 2)
        config_layout.addWidget(self.discovery_port_spin, 1, 3)
        config_layout.addWidget(self.start_button, 2, 0)
        config_layout.addWidget(self.stop_button, 2, 1)
        config_layout.addWidget(self.discover_button, 2, 2)
        config_layout.addWidget(self.link_button, 2, 3)

        self.self_info_label = QLabel("")
        self.self_info_label.setObjectName("info")
        self.status_label = QLabel("Чат ещё не запущен.")
        self.status_label.setObjectName("status")

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_panel = QFrame()
        left_panel.setObjectName("card")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)
        left_layout.addWidget(QLabel("Найденные пиры"))
        self.peers_list = QListWidget()
        left_layout.addWidget(self.peers_list, 1)

        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        self.chat_title = QLabel("История сообщений")
        self.chat_title.setObjectName("section")
        self.history_box = QPlainTextEdit()
        self.history_box.setReadOnly(True)
        self.history_box.setPlaceholderText("Здесь будет отдельная история локального чата между пользователями.")
        self.history_box.setMinimumHeight(320)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(10, 10, 10, 10)
        composer_layout.setSpacing(8)
        self.message_edit = QPlainTextEdit()
        self.message_edit.setPlaceholderText("Введите сообщение выбранному пиру")
        self.message_edit.setMaximumHeight(90)
        self.send_button = QPushButton("Отправить")
        self.broadcast_button = QPushButton("Всем")
        composer_layout.addWidget(self.message_edit, 1)
        composer_layout.addWidget(self.send_button)
        composer_layout.addWidget(self.broadcast_button)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("События локального чата")
        self.log_box.setMaximumHeight(160)

        right_layout.addWidget(self.chat_title)
        right_layout.addWidget(self.history_box, 1)
        right_layout.addWidget(composer)
        right_layout.addWidget(QLabel("События"))
        right_layout.addWidget(self.log_box)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([320, 720])

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(config_card)
        layout.addWidget(self.self_info_label)
        layout.addWidget(self.status_label)
        layout.addWidget(splitter, 1)

        self.start_button.clicked.connect(self._start_chat)
        self.stop_button.clicked.connect(self._stop_chat)
        self.discover_button.clicked.connect(self._discover)
        self.link_button.clicked.connect(self._open_manual_link_dialog)
        self.send_button.clicked.connect(self._send_message)
        self.broadcast_button.clicked.connect(self._broadcast_message)
        self.peers_list.itemSelectionChanged.connect(self._on_peer_selection_changed)

        self.stop_button.setEnabled(False)
        self.discover_button.setEnabled(False)
        self.link_button.setEnabled(False)
        self.send_button.setEnabled(False)
        self.broadcast_button.setEnabled(False)

    def _apply_styles(self) -> None:
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(
            f"""
            QWidget {{
                color: {COLORS["text"]};
                background: {COLORS["bg_main"]};
            }}
            QFrame#card {{
                background: {COLORS["bg_panel"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 16px;
            }}
            QFrame#composer {{
                background: {COLORS["bg_block"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
            }}
            QLabel#title {{
                font-size: 22px;
                font-weight: 700;
            }}
            QLabel#subtitle, QLabel#info, QLabel#status {{
                color: {COLORS["muted"]};
                font-size: 13px;
            }}
            QLabel#section {{
                font-size: 16px;
                font-weight: 600;
            }}
            QPushButton, QLineEdit, QPlainTextEdit, QListWidget, QSpinBox {{
                background: {COLORS["bg_block"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 12px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background: {COLORS["bg_accent"]};
            }}
            QListWidget::item {{
                padding: 8px;
                margin: 0;
                border-radius: 10px;
            }}
            QListWidget::item:selected {{
                background: {COLORS["bg_accent"]};
                border: 1px solid {COLORS["text"]};
            }}
            """
        )

    def _refresh_me(self) -> None:
        if self.messenger is None:
            self.self_info_label.setText("Узел ещё не запущен.")
            return
        me = self.messenger.get_self_info()
        self.self_info_label.setText(
            f"Имя: {me['name']} | MAC: {me['mac']} | IP: {me['ip']}:{me['tcp_port']} | session: {me['session_id']}"
        )

    def _start_chat(self) -> None:
        if self.messenger is not None and self.messenger.running:
            return
        self._stop_chat(silent=True)
        try:
            messenger = LanChatMessenger(
                name=self.name_edit.text().strip() or socket.gethostname(),
                tcp_port=self.tcp_port_spin.value(),
                discovery_port=self.discovery_port_spin.value(),
                mac_override=self.mac_override_edit.text().strip() or None,
            )
            messenger.start()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Локальный чат", str(exc))
            self._append_log(f"Ошибка запуска: {exc}")
            self.status_label.setText("Не удалось запустить локальный чат.")
            return

        self.messenger = messenger
        self.status_label.setText("Локальный чат запущен.")
        self._append_log("Локальный чат запущен.")
        self._refresh_me()
        self._update_controls_for_running(True)
        self._refresh_peers()
        self._refresh_history()

    def _stop_chat(self, silent: bool = False) -> None:
        if self.messenger is not None:
            self.messenger.stop()
            self.messenger = None
        self._update_controls_for_running(False)
        self._selected_peer_mac = None
        self.peers_list.clear()
        self.history_box.clear()
        self._refresh_me()
        if not silent:
            self.status_label.setText("Локальный чат остановлен.")
            self._append_log("Локальный чат остановлен.")

    def _update_controls_for_running(self, running: bool) -> None:
        self.name_edit.setEnabled(not running)
        self.mac_override_edit.setEnabled(not running)
        self.tcp_port_spin.setEnabled(not running)
        self.discovery_port_spin.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.discover_button.setEnabled(running)
        self.link_button.setEnabled(running)
        self.send_button.setEnabled(running)
        self.broadcast_button.setEnabled(running)

    def _poll_messenger(self) -> None:
        if self.messenger is None:
            return
        for event in self.messenger.drain_events():
            event_type = event.get("type", "")
            if event_type in {"status", "error"}:
                self.status_label.setText(str(event.get("text", "")))
                self._append_log(str(event.get("text", "")))
            if event_type in {"peers_updated", "message_received", "message_sent", "ledger_updated"}:
                self._refresh_peers()
                self._refresh_history()
        if self.messenger is not None and not self.messenger.running:
            self._update_controls_for_running(False)

    def _append_log(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        current = self.log_box.toPlainText().strip()
        updated = current + ("\n" if current else "") + f"- {cleaned}"
        self.log_box.setPlainText(updated)
        bar = self.log_box.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _refresh_peers(self) -> None:
        if self.messenger is None:
            self.peers_list.clear()
            return
        peers = self.messenger.get_peers()
        previous = self._selected_peer_mac
        self.peers_list.blockSignals(True)
        self.peers_list.clear()
        for peer in peers:
            suffix = "online" if peer.alive else "stale"
            item = QListWidgetItem(f"{peer.name}\n{peer.mac} | {peer.ip}:{peer.port} | {suffix}")
            item.setData(Qt.UserRole, peer.mac)
            self.peers_list.addItem(item)
            if peer.mac == previous:
                item.setSelected(True)
        self.peers_list.blockSignals(False)

    def _refresh_history(self) -> None:
        if self.messenger is None:
            self.history_box.clear()
            return
        records = self.messenger.get_history(peer_mac=self._selected_peer_mac, limit=120)
        lines: list[str] = []
        for record in records:
            event = record["event"]
            direction = "Вы" if event.get("direction") == "out" else event.get("peer_name", "peer")
            timestamp = event.get("timestamp", "-")
            text = event.get("text", "")
            block = short_hash(record["block_hash"], 12)
            lines.append(f"[{timestamp}] {direction}\n{text}\nledger={block}")
        self.history_box.setPlainText("\n\n".join(lines))
        bar = self.history_box.verticalScrollBar()
        bar.setValue(bar.maximum())
        if self._selected_peer_mac:
            self.chat_title.setText(f"История: {self._selected_peer_mac}")
        else:
            self.chat_title.setText("История сообщений")

    def _on_peer_selection_changed(self) -> None:
        current = self.peers_list.currentItem()
        self._selected_peer_mac = current.data(Qt.UserRole) if current is not None else None
        self._refresh_history()

    def _discover(self) -> None:
        if self.messenger is None:
            return
        self.messenger.broadcast_discovery()

    def _open_manual_link_dialog(self) -> None:
        if self.messenger is None:
            return
        dialog = ManualPeerDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        try:
            self.messenger.manual_link(
                str(values["mac"]),
                str(values["ip"]),
                int(values["port"]),
                str(values["name"]),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Пир", str(exc))
            return
        self._refresh_peers()

    def _send_message(self) -> None:
        if self.messenger is None:
            return
        if not self._selected_peer_mac:
            QMessageBox.information(self, "Отправка", "Сначала выберите пира в списке слева.")
            return
        text = self.message_edit.toPlainText().strip()
        if not text:
            return
        try:
            self.messenger.send_message(self._selected_peer_mac, text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Отправка", str(exc))
            self._append_log(f"Ошибка отправки: {exc}")
            return
        self.message_edit.clear()
        self._refresh_history()

    def _broadcast_message(self) -> None:
        if self.messenger is None:
            return
        text = self.message_edit.toPlainText().strip()
        if not text:
            return
        try:
            self.messenger.broadcast_message(text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Broadcast", str(exc))
            self._append_log(f"Ошибка broadcast: {exc}")
            return
        self.message_edit.clear()
        self._refresh_history()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_chat(silent=True)
        super().closeEvent(event)
