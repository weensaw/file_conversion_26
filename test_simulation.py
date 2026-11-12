import unittest

from simulation import (
    AdmissionDispatcher,
    Buffer,
    ConversionRequest,
    EventCalendar,
    EventType,
    SelectionDispatcher,
    Simulation,
    SimulationConfig,
    Statistics,
    Worker,
)


class FixedRandom:
    """Для проверки алгоритма обслуживание всегда длится 1 единицу."""

    def expovariate(self, rate):
        return 1.0


class SimulationTests(unittest.TestCase):
    def request(self, number, time=0.0):
        return ConversionRequest(
            source_id=0,
            sequence_number=number,
            created_at=time,
        )

    def test_first_free_slot_without_shifting(self):
        buffer = Buffer(3)

        first = self.request(1)
        second = self.request(2)
        third = self.request(3)

        buffer.put(0, first)
        buffer.put(1, second)
        buffer.put(2, third)

        buffer.remove(1)

        self.assertEqual(buffer.first_free_slot(), 1)
        self.assertEqual(buffer.snapshot()[0]["label"], "1.1")
        self.assertIsNone(buffer.snapshot()[1])
        self.assertEqual(buffer.snapshot()[2]["label"], "1.3")

    def test_overflow_rejects_latest_placement(self):
        buffer = Buffer(2)
        statistics = Statistics(1)
        admission = AdmissionDispatcher(buffer, statistics)

        first = self.request(1)
        second = self.request(2)
        third = self.request(3)
        fourth = self.request(4)

        for request in (first, second, third, fourth):
            statistics.record_generated(request)

        admission.accept(first, 0.0)
        admission.accept(second, 0.0)

        # Освобождаем ячейку 0, затем заполняем её снова.
        # Последняя заявка теперь находится НЕ в последней ячейке.
        buffer.remove(0)
        admission.accept(third, 0.0)
        admission.accept(fourth, 0.0)

        self.assertEqual(third.snapshot()["status"], "REJECTED")
        self.assertEqual(buffer.snapshot()[0]["label"], "1.4")
        self.assertEqual(buffer.snapshot()[1]["label"], "1.2")
        self.assertEqual(statistics.source_table()[0]["rejected"], 1)

    def test_ring_pointers_are_independent_and_wrap(self):
        buffer = Buffer(3)
        workers = [Worker(0), Worker(1)]
        calendar = EventCalendar()

        selection = SelectionDispatcher(
            buffer=buffer,
            workers=workers,
            calendar=calendar,
            service_mean=1.0,
        )

        for index in range(3):
            buffer.put(index, self.request(index + 1))

        selection.assign_available(0.0, FixedRandom())

        self.assertEqual(selection.pointers(), {"buffer": 2, "worker": 0})
        self.assertEqual(workers[0].snapshot()["request"]["label"], "1.1")
        self.assertEqual(workers[1].snapshot()["request"]["label"], "1.2")

        event = calendar.pop_next()
        workers[event.target_id].finish(event.time)
        selection.assign_available(event.time, FixedRandom())

        self.assertEqual(selection.pointers(), {"buffer": 0, "worker": 1})
        self.assertEqual(workers[0].snapshot()["request"]["label"], "1.3")

    def test_equal_time_events_keep_scheduling_order(self):
        calendar = EventCalendar()

        calendar.schedule(2.0, EventType.ARRIVAL, 1)
        calendar.schedule(2.0, EventType.COMPLETION, 0)

        self.assertEqual(calendar.pop_next().kind, EventType.ARRIVAL)
        self.assertEqual(calendar.pop_next().kind, EventType.COMPLETION)

    def test_generation_stops_and_remaining_requests_are_completed(self):
        simulation = Simulation(
            SimulationConfig(
                source_count=3,
                worker_count=2,
                buffer_capacity=3,
                arrival_min=0.1,
                arrival_max=0.2,
                service_mean=2.0,
                request_limit=100,
                seed=7,
            )
        )

        generation_stop_time = None

        while simulation.step():
            state = simulation.snapshot()

            if state["generation_closed"] and generation_stop_time is None:
                generation_stop_time = state["time"]

        results = simulation.results()
        state = simulation.snapshot()

        generated = sum(row["generated"] for row in results["sources"])
        completed = sum(row["completed"] for row in results["sources"])
        rejected = sum(row["rejected"] for row in results["sources"])

        self.assertEqual(generated, 100)
        self.assertEqual(generated, completed + rejected)
        self.assertGreater(rejected, 0)

        for row in results["sources"]:
            self.assertEqual(
                row["generated"],
                row["completed"] + row["rejected"],
            )

        self.assertTrue(state["finished"])
        self.assertTrue(all(slot is None for slot in state["buffer"]))
        self.assertTrue(
            all(worker["request"] is None for worker in state["workers"])
        )

        self.assertGreater(results["end_time"], generation_stop_time)

        for row in results["workers"]:
            self.assertGreaterEqual(row["utilization"], 0.0)
            self.assertLessEqual(row["utilization"], 1.0 + 1e-12)

        self.assertEqual(
            completed,
            sum(row["completed"] for row in results["workers"]),
        )

    def test_statistics_for_known_request(self):
        statistics = Statistics(1)
        request = self.request(1, time=2.0)

        statistics.record_generated(request)
        request.mark_waiting()
        request.start(5.0)
        request.complete(9.0)
        statistics.record_completed(request)

        row = statistics.source_table()[0]

        self.assertEqual(row["mean_waiting"], 3.0)
        self.assertEqual(row["mean_service"], 4.0)
        self.assertEqual(row["mean_system"], 7.0)
        self.assertEqual(row["rejection_probability"], 0.0)

        # Одного наблюдения недостаточно для выборочной дисперсии.
        self.assertIsNone(row["variance_waiting"])

    def test_same_seed_reproduces_results(self):
        config = SimulationConfig(request_limit=100, seed=42)

        first = Simulation(config)
        second = Simulation(config)

        first.run()
        second.run()

        self.assertEqual(first.results(), second.results())


if __name__ == "__main__":
    unittest.main()