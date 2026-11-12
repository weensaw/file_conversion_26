from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, variance


@dataclass(frozen=True)
class SimulationConfig:
    source_count: int = 3
    worker_count: int = 2
    buffer_capacity: int = 5

    arrival_min: float = 1.0
    arrival_max: float = 3.0
    service_mean: float = 1.5

    request_limit: int = 1000
    seed: int = 42

    def validate(self) -> None:
        integer_fields = {
            "source_count": self.source_count,
            "worker_count": self.worker_count,
            "buffer_capacity": self.buffer_capacity,
            "request_limit": self.request_limit,
        }

        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}: требуется целое число.")
            if value < 1:
                raise ValueError(f"{name}: значение должно быть не меньше 1.")

        for name, value in {
            "arrival_min": self.arrival_min,
            "arrival_max": self.arrival_max,
            "service_mean": self.service_mean,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name}: требуется конечное число.")

        if not 0 < self.arrival_min < self.arrival_max:
            raise ValueError(
                "Нужно выполнить условие: 0 < arrival_min < arrival_max."
            )

        if self.service_mean <= 0:
            raise ValueError("service_mean должно быть больше 0.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed должен быть целым числом.")


class ConversionRequest:
    def __init__(
        self,
        source_id: int,
        sequence_number: int,
        created_at: float,
    ) -> None:
        self._source_id = source_id
        self._sequence_number = sequence_number
        self._file_name = f"file_{source_id + 1}_{sequence_number}.png"
        self._input_format = "png"
        self._output_format = "jpg"

        self._created_at = created_at
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._status = "CREATED"

    @property
    def source_id(self) -> int:
        return self._source_id

    @property
    def label(self) -> str:
        return f"{self._source_id + 1}.{self._sequence_number}"

    @property
    def created_at(self) -> float:
        return self._created_at

    def mark_waiting(self) -> None:
        if self._status != "CREATED":
            raise RuntimeError("В буфер можно поставить только новую заявку.")
        self._status = "WAITING"

    def start(self, now: float) -> None:
        if self._status != "WAITING":
            raise RuntimeError("Обслуживание возможно только после ожидания.")
        if now < self._created_at:
            raise RuntimeError("Время начала меньше времени поступления.")

        self._started_at = now
        self._status = "PROCESSING"

    def complete(self, now: float) -> None:
        if self._status != "PROCESSING" or self._started_at is None:
            raise RuntimeError("Заявка не находится на обслуживании.")
        if now < self._started_at:
            raise RuntimeError("Время завершения меньше времени начала.")

        self._finished_at = now
        self._status = "COMPLETED"

    def reject(self, now: float) -> None:
        if self._status != "WAITING":
            raise RuntimeError("Вытеснить можно только ожидающую заявку.")

        self._finished_at = now
        self._status = "REJECTED"

    def snapshot(self) -> dict:
        return {
            "label": self.label,
            "source_id": self._source_id,
            "sequence_number": self._sequence_number,
            "file_name": self._file_name,
            "input_format": self._input_format,
            "output_format": self._output_format,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "status": self._status,
        }


class Source:
    def __init__(
        self,
        source_id: int,
        arrival_min: float,
        arrival_max: float,
    ) -> None:
        self._id = source_id
        self._generated_count = 0
        self._arrival_min = arrival_min
        self._arrival_max = arrival_max

    def next_arrival(self, now: float, rng: random.Random) -> float:
        return now + rng.uniform(self._arrival_min, self._arrival_max)

    def generate(self, now: float) -> ConversionRequest:
        self._generated_count += 1
        return ConversionRequest(
            source_id=self._id,
            sequence_number=self._generated_count,
            created_at=now,
        )

    def snapshot(self) -> dict:
        return {
            "source_id": self._id,
            "generated": self._generated_count,
        }


class EventType(Enum):
    ARRIVAL = "ARRIVAL"
    COMPLETION = "COMPLETION"


@dataclass(order=True, frozen=True)
class Event:
    time: float
    order: int
    kind: EventType = field(compare=False)
    target_id: int = field(compare=False)


class EventCalendar:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._next_order = 0

    def schedule(
        self,
        time: float,
        kind: EventType,
        target_id: int,
    ) -> None:
        event = Event(
            time=time,
            order=self._next_order,
            kind=kind,
            target_id=target_id,
        )
        self._next_order += 1
        heapq.heappush(self._events, event)

    def pop_next(self) -> Event:
        if not self._events:
            raise RuntimeError("Календарь событий пуст.")
        return heapq.heappop(self._events)

    def cancel_arrivals(self) -> None:
        self._events = [
            event
            for event in self._events
            if event.kind is not EventType.ARRIVAL
        ]
        heapq.heapify(self._events)

    def is_empty(self) -> bool:
        return not self._events


class Buffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("В буфере должна быть хотя бы одна ячейка.")

        self._slots: list[ConversionRequest | None] = [None] * capacity

        # Номер помещения в буфер разрешает совпадения времён:
        # последней считается фактически последняя поставленная заявка.
        self._placement_order: list[int] = [-1] * capacity
        self._next_placement_order = 0

    def first_free_slot(self) -> int | None:
        for index, request in enumerate(self._slots):
            if request is None:
                return index
        return None

    def newest_request_slot(self) -> int | None:
        occupied = [
            index
            for index, request in enumerate(self._slots)
            if request is not None
        ]

        if not occupied:
            return None

        return max(occupied, key=lambda index: self._placement_order[index])

    def first_occupied_from(self, pointer: int) -> int | None:
        for offset in range(self.capacity()):
            index = (pointer + offset) % self.capacity()
            if self._slots[index] is not None:
                return index
        return None

    def put(self, slot: int, request: ConversionRequest) -> None:
        if self._slots[slot] is not None:
            raise RuntimeError("Нельзя записать заявку в занятую ячейку.")

        request.mark_waiting()
        self._slots[slot] = request
        self._placement_order[slot] = self._next_placement_order
        self._next_placement_order += 1

    def remove(self, slot: int) -> ConversionRequest:
        request = self._slots[slot]

        if request is None:
            raise RuntimeError("Нельзя извлечь заявку из пустой ячейки.")

        self._slots[slot] = None
        self._placement_order[slot] = -1
        return request

    def is_empty(self) -> bool:
        return all(request is None for request in self._slots)

    def capacity(self) -> int:
        return len(self._slots)

    def snapshot(self) -> list:
        return [
            request.snapshot() if request is not None else None
            for request in self._slots
        ]


class Worker:
    def __init__(self, worker_id: int) -> None:
        self._id = worker_id
        self._current_request: ConversionRequest | None = None
        self._finish_at: float | None = None
        self._busy_time = 0.0
        self._completed_count = 0

    def is_free(self) -> bool:
        return self._current_request is None

    def start(
        self,
        request: ConversionRequest,
        now: float,
        duration: float,
    ) -> None:
        if not self.is_free():
            raise RuntimeError("Прибор уже занят.")
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Длительность должна быть конечной и неотрицательной.")

        request.start(now)
        self._current_request = request
        self._finish_at = now + duration

    def finish(self, now: float) -> ConversionRequest:
        if self._current_request is None or self._finish_at is None:
            raise RuntimeError("Нельзя завершить работу свободного прибора.")

        if not math.isclose(now, self._finish_at, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError("Событие завершения пришло в неверный момент.")

        request = self._current_request
        started_at = request.snapshot()["started_at"]

        request.complete(now)
        self._busy_time += now - started_at
        self._completed_count += 1

        self._current_request = None
        self._finish_at = None
        return request

    def snapshot(self) -> dict:
        return {
            "worker_id": self._id,
            "request": (
                self._current_request.snapshot()
                if self._current_request is not None
                else None
            ),
            "finish_at": self._finish_at,
            "busy_time": self._busy_time,
            "completed": self._completed_count,
        }


class Statistics:
    def __init__(self, source_count: int) -> None:
        self._source_totals = {
            source_id: {
                "generated": 0,
                "rejected": 0,
                "completed": 0,
            }
            for source_id in range(source_count)
        }

        self._waiting_times = {
            source_id: [] for source_id in range(source_count)
        }
        self._service_times = {
            source_id: [] for source_id in range(source_count)
        }
        self._system_times = {
            source_id: [] for source_id in range(source_count)
        }

    def record_generated(self, request: ConversionRequest) -> None:
        self._source_totals[request.source_id]["generated"] += 1

    def record_rejected(self, request: ConversionRequest) -> None:
        if request.snapshot()["status"] != "REJECTED":
            raise RuntimeError("Учитываемая заявка должна иметь статус REJECTED.")

        self._source_totals[request.source_id]["rejected"] += 1

    def record_completed(self, request: ConversionRequest) -> None:
        data = request.snapshot()

        if data["status"] != "COMPLETED":
            raise RuntimeError("Учитываемая заявка должна быть завершена.")

        source_id = request.source_id

        waiting = data["started_at"] - data["created_at"]
        service = data["finished_at"] - data["started_at"]
        system = data["finished_at"] - data["created_at"]

        self._source_totals[source_id]["completed"] += 1
        self._waiting_times[source_id].append(waiting)
        self._service_times[source_id].append(service)
        self._system_times[source_id].append(system)

    @staticmethod
    def _average(values: list[float]) -> float | None:
        return mean(values) if values else None

    @staticmethod
    def _variance(values: list[float]) -> float | None:
        # Несмещённая выборочная дисперсия: знаменатель n - 1.
        # При 0 или 1 наблюдении оценка не определена.
        return variance(values) if len(values) >= 2 else None

    def source_table(self) -> list:
        rows = []

        for source_id, counts in self._source_totals.items():
            generated = counts["generated"]
            rejected = counts["rejected"]

            rows.append(
                {
                    "source_id": source_id,
                    **counts,
                    "rejection_probability": (
                        rejected / generated if generated else None
                    ),
                    "mean_system": self._average(
                        self._system_times[source_id]
                    ),
                    "mean_waiting": self._average(
                        self._waiting_times[source_id]
                    ),
                    "mean_service": self._average(
                        self._service_times[source_id]
                    ),
                    "variance_waiting": self._variance(
                        self._waiting_times[source_id]
                    ),
                    "variance_service": self._variance(
                        self._service_times[source_id]
                    ),
                }
            )

        return rows

    def worker_table(self, workers: list[Worker], end_time: float) -> list:
        rows = []

        for worker in workers:
            data = worker.snapshot()
            rows.append(
                {
                    "worker_id": data["worker_id"],
                    "completed": data["completed"],
                    "busy_time": data["busy_time"],
                    "utilization": (
                        data["busy_time"] / end_time
                        if end_time > 0
                        else 0.0
                    ),
                }
            )

        return rows

    def rejection_probability(self) -> float:
        generated = sum(
            row["generated"] for row in self._source_totals.values()
        )
        rejected = sum(
            row["rejected"] for row in self._source_totals.values()
        )
        return rejected / generated if generated else 0.0


class AdmissionDispatcher:
    def __init__(self, buffer: Buffer, statistics: Statistics) -> None:
        self._buffer = buffer
        self._statistics = statistics

    def accept(self, request: ConversionRequest, now: float) -> list:
        actions = []
        slot = self._buffer.first_free_slot()

        if slot is None:
            slot = self._buffer.newest_request_slot()

            if slot is None:
                raise RuntimeError("Не найдена заявка для вытеснения.")

            victim = self._buffer.remove(slot)
            victim.reject(now)
            self._statistics.record_rejected(victim)

            actions.append(
                f"Отказ: заявка {victim.label} вытеснена "
                f"из ячейки {slot + 1} по Д1ОО4."
            )

        self._buffer.put(slot, request)
        actions.append(
            f"Заявка {request.label} помещена в ячейку {slot + 1}."
        )
        return actions


class SelectionDispatcher:
    def __init__(
        self,
        buffer: Buffer,
        workers: list[Worker],
        calendar: EventCalendar,
        service_mean: float,
    ) -> None:
        self._buffer = buffer
        self._workers = workers
        self._calendar = calendar
        self._service_mean = service_mean

        self._buffer_pointer = 0
        self._worker_pointer = 0

    def _find_free_worker(self) -> int | None:
        for offset in range(len(self._workers)):
            index = (self._worker_pointer + offset) % len(self._workers)
            if self._workers[index].is_free():
                return index
        return None

    def assign_available(self, now: float, rng: random.Random) -> list:
        actions = []

        while not self._buffer.is_empty():
            worker_id = self._find_free_worker()

            if worker_id is None:
                actions.append(
                    "Все приборы заняты: заявки остаются в буфере."
                )
                break

            slot = self._buffer.first_occupied_from(self._buffer_pointer)

            if slot is None:
                raise RuntimeError("В непустом буфере не найдена заявка.")

            request = self._buffer.remove(slot)

            # expovariate принимает интенсивность λ = 1 / среднего времени.
            duration = rng.expovariate(1.0 / self._service_mean)

            self._workers[worker_id].start(request, now, duration)
            self._calendar.schedule(
                time=now + duration,
                kind=EventType.COMPLETION,
                target_id=worker_id,
            )

            self._buffer_pointer = (slot + 1) % self._buffer.capacity()
            self._worker_pointer = (worker_id + 1) % len(self._workers)

            actions.append(
                f"Заявка {request.label}: ячейка {slot + 1} → "
                f"прибор {worker_id + 1}; "
                f"завершение в t={now + duration:.4f}."
            )

        return actions

    def pointers(self) -> dict:
        return {
            "buffer": self._buffer_pointer,
            "worker": self._worker_pointer,
        }


class Simulation:
    def __init__(self, config: SimulationConfig) -> None:
        config.validate()
        self._config = config
        self.initialize()

    def initialize(self) -> None:
        self._rng = random.Random(self._config.seed)
        self._now = 0.0
        self._generated_total = 0
        self._generation_closed = False
        self._step_number = 0

        self._sources = [
            Source(
                source_id=index,
                arrival_min=self._config.arrival_min,
                arrival_max=self._config.arrival_max,
            )
            for index in range(self._config.source_count)
        ]

        self._workers = [
            Worker(index) for index in range(self._config.worker_count)
        ]

        self._calendar = EventCalendar()
        self._buffer = Buffer(self._config.buffer_capacity)
        self._statistics = Statistics(self._config.source_count)

        self._admission = AdmissionDispatcher(
            self._buffer,
            self._statistics,
        )

        self._selection = SelectionDispatcher(
            buffer=self._buffer,
            workers=self._workers,
            calendar=self._calendar,
            service_mean=self._config.service_mean,
        )

        self._last_actions = ["Модель инициализирована."]

        for source_id, source in enumerate(self._sources):
            self._calendar.schedule(
                time=source.next_arrival(0.0, self._rng),
                kind=EventType.ARRIVAL,
                target_id=source_id,
            )

    def _close_generation(self) -> None:
        self._generation_closed = True
        self._calendar.cancel_arrivals()
        self._last_actions.append(
            "Лимит генерации достигнут. "
            "Новые заявки больше не поступают; "
            "оставшиеся будут дообслужены."
        )

    def _handle_arrival(self, event: Event) -> None:
        source = self._sources[event.target_id]
        request = source.generate(self._now)

        self._generated_total += 1
        self._statistics.record_generated(request)
        self._last_actions.append(
            f"Источник {event.target_id + 1} создал заявку {request.label}."
        )

        if self._generated_total >= self._config.request_limit:
            self._close_generation()
        else:
            self._calendar.schedule(
                time=source.next_arrival(self._now, self._rng),
                kind=EventType.ARRIVAL,
                target_id=event.target_id,
            )

        self._last_actions.extend(
            self._admission.accept(request, self._now)
        )

    def _handle_completion(self, event: Event) -> None:
        request = self._workers[event.target_id].finish(self._now)
        self._statistics.record_completed(request)

        self._last_actions.append(
            f"Прибор {event.target_id + 1} завершил заявку {request.label}."
        )

    def step(self) -> bool:
        if self.is_finished():
            return False

        if self._calendar.is_empty():
            raise RuntimeError(
                "Модель не завершена, но календарь пуст: "
                "нарушена логика планирования."
            )

        event = self._calendar.pop_next()

        if event.time < self._now:
            raise RuntimeError("Модельное время не может идти назад.")

        self._now = event.time
        self._step_number += 1
        self._last_actions = []

        if event.kind is EventType.ARRIVAL:
            self._handle_arrival(event)
        else:
            self._handle_completion(event)

        self._last_actions.extend(
            self._selection.assign_available(self._now, self._rng)
        )

        return True

    def run(self) -> None:
        while self.step():
            pass

    def is_finished(self) -> bool:
        return (
            self._generation_closed
            and self._buffer.is_empty()
            and all(worker.is_free() for worker in self._workers)
        )

    def snapshot(self) -> dict:
        source_rows = self._statistics.source_table()

        return {
            "time": self._now,
            "step": self._step_number,
            "generated_total": self._generated_total,
            "request_limit": self._config.request_limit,
            "generation_closed": self._generation_closed,
            "finished": self.is_finished(),
            "sources": [source.snapshot() for source in self._sources],
            "source_statistics": source_rows,
            "buffer": self._buffer.snapshot(),
            "workers": [worker.snapshot() for worker in self._workers],
            "pointers": self._selection.pointers(),
            "actions": list(self._last_actions),
            "rejection_probability": (
                self._statistics.rejection_probability()
            ),
        }

    def results(self) -> dict:
        if not self.is_finished():
            raise RuntimeError(
                "Итоговые результаты доступны после дообслуживания "
                "всех оставшихся заявок."
            )

        return {
            "end_time": self._now,
            "request_count": self._generated_total,
            "step_count": self._step_number,
            "sources": self._statistics.source_table(),
            "workers": self._statistics.worker_table(
                self._workers,
                self._now,
            ),
            "rejection_probability": (
                self._statistics.rejection_probability()
            ),
        }


if __name__ == "__main__":
    from pprint import pprint

    simulation = Simulation(SimulationConfig())
    simulation.run()

    print("Модель завершена.")
    pprint(simulation.results(), sort_dicts=False)