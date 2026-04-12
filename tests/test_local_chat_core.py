from __future__ import annotations

import unittest

from assistant.local_chat.core import LanChatMessenger, format_mac


class LocalChatCoreTests(unittest.TestCase):
    def test_format_mac_normalizes_input(self) -> None:
        self.assertEqual(format_mac("aabbccddeeff"), "AA:BB:CC:DD:EE:FF")

    def test_manual_link_registers_peer_and_emits_update(self) -> None:
        messenger = LanChatMessenger(
            name="Alice",
            tcp_port=47010,
            discovery_port=47001,
            mac_override="AA:BB:CC:DD:EE:01",
        )

        peer = messenger.manual_link("AA:BB:CC:DD:EE:02", "192.168.0.7", 47011, "Bob")

        self.assertEqual(peer.name, "Bob")
        peers = messenger.get_peers()
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0].mac, "AA:BB:CC:DD:EE:02")
        events = messenger.drain_events()
        self.assertTrue(any(item["type"] == "peers_updated" for item in events))

    def test_send_message_records_ledger_with_stubbed_transport(self) -> None:
        messenger = LanChatMessenger(
            name="Alice",
            tcp_port=47010,
            discovery_port=47001,
            mac_override="AA:BB:CC:DD:EE:01",
        )
        messenger.manual_link("AA:BB:CC:DD:EE:02", "127.0.0.1", 47011, "Bob")
        messenger._send_tcp_payload = lambda *_args, **_kwargs: {"status": "ok", "block_hash": "remote"}  # type: ignore[method-assign]

        record = messenger.send_message("AA:BB:CC:DD:EE:02", "hello")

        self.assertEqual(record["event"]["direction"], "out")
        self.assertEqual(record["event"]["peer_name"], "Bob")
        history = messenger.get_history(peer_mac="AA:BB:CC:DD:EE:02")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["event"]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
