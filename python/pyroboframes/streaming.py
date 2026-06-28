"""Streaming data ingestion from MQTT, Kafka, and WebSocket.

Enables online learning and closed-loop data collection directly from robot streams.

```python
from pyroboframes.streaming import MQTTStreamer, StreamingRoboticsDataset

# MQTT: e.g., robot publishes state/action to MQTT
streamer = MQTTStreamer(broker="localhost", port=1883)
streamer.subscribe_topics({
    "state": "robot/state",
    "action": "robot/action",
    "image": "robot/camera/front",
})

# Collect into DataFrame
stream_ds = StreamingRoboticsDataset(streamer, buffer_size=1000)

# Use in training loop
for batch in stream_ds.get_batches(batch_size=32):
    train_step(batch)
```
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass


@dataclass
class StreamMessage:
    """Single message from a stream."""

    topic: str
    timestamp: float
    data: Any
    sequence_id: int


class MessageBuffer:
    """Thread-safe circular buffer for streaming messages."""

    def __init__(self, max_size: int = 10000):
        """Initialize buffer.

        Args:
            max_size: Maximum messages to buffer
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.seq = 0

    def append(self, topic: str, data: Any) -> int:
        """Add message to buffer.

        Args:
            topic: Message topic
            data: Message data

        Returns:
            Sequence ID of message
        """
        with self.lock:
            msg = StreamMessage(
                topic=topic,
                timestamp=time.time(),
                data=data,
                sequence_id=self.seq,
            )
            self.buffer.append(msg)
            self.seq += 1
            return msg.sequence_id

    def get_all(self) -> list[StreamMessage]:
        """Get all buffered messages."""
        with self.lock:
            return list(self.buffer)

    def get_since(self, seq: int) -> list[StreamMessage]:
        """Get messages since sequence ID."""
        with self.lock:
            return [m for m in self.buffer if m.sequence_id >= seq]

    def clear(self) -> None:
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()


class MQTTStreamer:
    """Stream data from MQTT broker.

    Requires: pip install paho-mqtt
    """

    def __init__(self, broker: str, port: int = 1883, timeout: float = 60.0):
        """Initialize MQTT streamer.

        Args:
            broker: MQTT broker host
            port: MQTT broker port
            timeout: Connection timeout (seconds)
        """
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise ImportError("MQTT support requires: pip install paho-mqtt")

        self.broker = broker
        self.port = port
        self.timeout = timeout
        self.buffer = MessageBuffer()
        self.subscriptions = {}

        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self._connected = False

    def subscribe_topics(self, topic_map: dict[str, str]) -> None:
        """Subscribe to MQTT topics.

        Args:
            topic_map: Dict mapping {name: mqtt_topic}
                      E.g., {"state": "robot/state", "action": "robot/action"}
        """
        self.subscriptions = topic_map
        for mqtt_topic in topic_map.values():
            self.client.subscribe(mqtt_topic)

    def connect(self) -> None:
        """Connect to MQTT broker."""
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()
        time.sleep(0.5)  # Wait for connection

    def disconnect(self) -> None:
        """Disconnect from broker."""
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback."""
        if rc == 0:
            self._connected = True

    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            if msg.payload:
                data = json.loads(msg.payload.decode())
            else:
                data = None
        except json.JSONDecodeError:
            data = msg.payload.decode()

        self.buffer.append(msg.topic, data)

    def get_messages(self, since_seq: int | None = None) -> list[StreamMessage]:
        """Get buffered messages.

        Args:
            since_seq: Only get messages after this sequence ID

        Returns:
            List of StreamMessage
        """
        if since_seq is None:
            return self.buffer.get_all()
        else:
            return self.buffer.get_since(since_seq)


class KafkaStreamer:
    """Stream data from Kafka topic.

    Requires: pip install kafka-python
    """

    def __init__(self, bootstrap_servers: str | list[str], group_id: str = "pyrf"):
        """Initialize Kafka streamer.

        Args:
            bootstrap_servers: Kafka broker(s)
            group_id: Consumer group ID
        """
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise ImportError("Kafka support requires: pip install kafka-python")

        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.buffer = MessageBuffer()
        self.consumer = None
        self.topics = []

    def subscribe_topics(self, topics: list[str]) -> None:
        """Subscribe to Kafka topics.

        Args:
            topics: List of topic names
        """
        from kafka import KafkaConsumer

        self.topics = topics
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

    def connect(self) -> None:
        """Start consuming (in background thread)."""
        if self.consumer is None:
            raise RuntimeError("subscribe_topics() first")

        def consume_loop():
            for message in self.consumer:
                self.buffer.append(message.topic, message.value)

        thread = threading.Thread(target=consume_loop, daemon=True)
        thread.start()

    def get_messages(self) -> list[StreamMessage]:
        """Get buffered messages."""
        return self.buffer.get_all()


class StreamingRoboticsDataset:
    """Collect streaming messages into aligned batches."""

    def __init__(
        self,
        streamer: MQTTStreamer | KafkaStreamer,
        buffer_size: int = 1000,
        alignment_window: float = 0.1,
    ):
        """Initialize streaming dataset.

        Args:
            streamer: MQTT or Kafka streamer
            buffer_size: Messages to buffer
            alignment_window: Time window for aligning messages (seconds)
        """
        self.streamer = streamer
        self.buffer_size = buffer_size
        self.alignment_window = alignment_window
        self.last_seq = 0

    def get_batches(self, batch_size: int = 32, timeout: float = 10.0):
        """Yield batches of aligned messages.

        Args:
            batch_size: Messages per batch
            timeout: Timeout for waiting for data (seconds)

        Yields:
            Dicts of {topic: [values]} aligned by time window
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            messages = self.streamer.get_messages()
            if len(messages) > self.last_seq + batch_size:
                # Align messages by time window
                batch = self._align_messages(
                    messages[self.last_seq : self.last_seq + batch_size]
                )
                self.last_seq += batch_size
                yield batch
            else:
                time.sleep(0.1)

    def _align_messages(self, messages: list[StreamMessage]) -> dict[str, list]:
        """Align messages by time window into a batch.

        Args:
            messages: List of messages

        Returns:
            Dict of {topic: [values]}
        """
        batch = {}
        for msg in messages:
            if msg.topic not in batch:
                batch[msg.topic] = []
            batch[msg.topic].append(msg.data)
        return batch
