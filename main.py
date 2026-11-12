from __future__ import annotations

import argparse
import math
from dataclasses import replace
from statistics import NormalDist

from simulation import Simulation, SimulationConfig


def format_value(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(headers, rows):
    """Вывод таблицы без дополнительных библиотек."""
    rows = [[str(value) for value in row] for row in rows]

    widths = [
        max(
            len(str(header)),
            max((len(row[index]) for row in rows), default=0),
        )
        for index, header in enumerate(headers)
    ]

    def line(row):
        return " | ".join(
            str(value).ljust(width)
            for value, width in zip(row, widths)
        )

    print(line(headers))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(line(row))


class StepView:
    def render(self, state):
        print("\n" + "=" * 78)
        print(
            f"ШАГ {state['step']}   "
            f"Модельное время: {state['time']:.4f}   "
            f"Создано: {state['generated_total']}/{state['request_limit']}"
        )

        if state["finished"]:
            phase = "Моделирование завершено"
        elif state["generation_closed"]:
            phase = "Генерация закрыта. Дообслуживание оставшихся заявок"
        else:
            phase = "Генерация и обслуживание"

        print(phase)

        print("\nИСТОЧНИКИ")
        source_rows = []

        for row in state["source_statistics"]:
            probability = row["rejection_probability"]
            source_rows.append(
                [
                    f"И{row['source_id'] + 1}",
                    row["generated"],
                    row["completed"],
                    row["rejected"],
                    (
                        f"{probability * 100:.2f}%"
                        if probability is not None
                        else "—"
                    ),
                ]
            )

        print_table(
            ["Источник", "Создано", "Обслужено", "Отказы", "Доля отказов"],
            source_rows,
        )

        print("\nБУФЕР")
        buffer_pointer = state["pointers"]["buffer"]

        print_table(
            ["Ячейка", "Заявка", "Поступление", "Указатель выбора"],
            [
                [
                    index + 1,
                    request["label"] if request else "свободна",
                    (
                        format_value(request["created_at"])
                        if request
                        else "—"
                    ),
                    "<-- УБ" if index == buffer_pointer else "",
                ]
                for index, request in enumerate(state["buffer"])
            ],
        )

        print("\nПРИБОРЫ")
        worker_pointer = state["pointers"]["worker"]

        print_table(
            ["Прибор", "Состояние", "Заявка", "Окончание", "Указатель выбора"],
            [
                [
                    f"П{worker['worker_id'] + 1}",
                    "занят" if worker["request"] else "свободен",
                    (
                        worker["request"]["label"]
                        if worker["request"]
                        else "—"
                    ),
                    format_value(worker["finish_at"]),
                    (
                        "<-- УП"
                        if worker["worker_id"] == worker_pointer
                        else ""
                    ),
                ]
                for worker in state["workers"]
            ],
        )

        print("\nДЕЙСТВИЯ ШАГА")
        for action in state["actions"]:
            print(f"• {action}")

        print(
            "\nОбщая доля отказов: "
            f"{state['rejection_probability'] * 100:.2f}%"
        )


class SummaryView:
    def render(self, results):
        print("\nИТОГИ МОДЕЛИРОВАНИЯ")
        print(
            f"Создано заявок: {results['request_count']}. "
            f"Событий: {results['step_count']}. "
            f"Время реализации: {results['end_time']:.4f}."
        )

        if "accuracy_note" in results:
            print(results["accuracy_note"])

        print("\nТАБЛИЦА 1. ПОКАЗАТЕЛИ ИСТОЧНИКОВ")

        print_table(
            [
                "Ист.",
                "Создано",
                "Обслуж.",
                "Отказ",
                "Pотк",
                "Tсист",
                "Tож",
                "Tобсл",
                "Dож",
                "Dобсл",
            ],
            [
                [
                    f"И{row['source_id'] + 1}",
                    row["generated"],
                    row["completed"],
                    row["rejected"],
                    format_value(row["rejection_probability"]),
                    format_value(row["mean_system"]),
                    format_value(row["mean_waiting"]),
                    format_value(row["mean_service"]),
                    format_value(row["variance_waiting"]),
                    format_value(row["variance_service"]),
                ]
                for row in results["sources"]
            ],
        )

        print("\nТАБЛИЦА 2. ПОКАЗАТЕЛИ ПРИБОРОВ")

        print_table(
            ["Прибор", "Обслужено", "Время занятости", "Коэф. использования"],
            [
                [
                    f"П{row['worker_id'] + 1}",
                    row["completed"],
                    format_value(row["busy_time"]),
                    format_value(row["utilization"]),
                ]
                for row in results["workers"]
            ],
        )

        print(
            "\nВремена — в условных единицах; дисперсии — "
            "в квадрате этих единиц."
        )
        print(
            "Временные показатели рассчитаны по обслуженным заявкам. "
            "«—» означает недостаточно наблюдений."
        )


class AutomaticRunner:
    """
    Подбор объёма моделирования по приближённой формуле пособия.

    Проверка выполняется отдельно для каждого источника.
    Это учебный критерий для вероятности отказа:
    он не доказывает заданную точность всех средних и дисперсий.
    """

    def __init__(
        self,
        base_config,
        relative_error=0.1,
        confidence=0.9,
        max_requests=200_000,
        max_iterations=10,
    ):
        if not 0 < relative_error < 1:
            raise ValueError("Относительная точность должна быть между 0 и 1.")
        if not 0 < confidence < 1:
            raise ValueError("Доверительная вероятность должна быть между 0 и 1.")
        if max_requests < base_config.request_limit:
            raise ValueError(
                "Максимальное число заявок должно быть не меньше "
                "начального числа."
            )
        if max_iterations < 2:
            raise ValueError("Нужно разрешить минимум две итерации.")

        self._base_config = base_config
        self._relative_error = relative_error
        self._confidence = confidence
        self._max_requests = max_requests
        self._max_iterations = max_iterations

        # Для двусторонней доверительной вероятности 0.9:
        # z ≈ 1.64485, в пособии используется округлённое значение.
        self._z = NormalDist().inv_cdf((1 + confidence) / 2)

    def _estimate_required_count(self, probability):
        if probability is None or not 0 < probability < 1:
            return None

        return max(
            2,
            math.ceil(
                self._z**2
                * (1 - probability)
                / (probability * self._relative_error**2)
            ),
        )

    def run(self):
        request_count = self._base_config.request_limit
        previous_probabilities = None
        result = None

        for iteration in range(1, self._max_iterations + 1):
            # Новый воспроизводимый запуск с отдельным seed.
            config = replace(
                self._base_config,
                request_limit=request_count,
                seed=self._base_config.seed + iteration - 1,
            )

            simulation = Simulation(config)
            simulation.run()
            result = simulation.results()

            enough_observations = True
            stable_estimates = previous_probabilities is not None

            # Если текущего объёма недостаточно, следующий запуск
            # увеличивается как минимум на 50%.
            next_count = math.ceil(request_count * 1.5)

            current_probabilities = {}

            for row in result["sources"]:
                source_id = row["source_id"]
                generated = row["generated"]
                probability = row["rejection_probability"]
                required = self._estimate_required_count(probability)

                current_probabilities[source_id] = probability

                if required is None or generated < required:
                    enough_observations = False

                if required is None:
                    # При отсутствии отказов нельзя подставить p=0
                    # в формулу и объявить точность достигнутой.
                    # Аналогично осторожно обрабатываем p=1.
                    next_count = max(next_count, request_count * 2)

                elif generated > 0:
                    # Формула даёт число заявок одного источника.
                    # Переводим его в требуемое общее число,
                    # используя наблюдавшуюся долю этого источника.
                    source_share = generated / request_count
                    needed_total = math.ceil(required / source_share)
                    next_count = max(next_count, needed_total)

                previous = (
                    previous_probabilities.get(source_id)
                    if previous_probabilities is not None
                    else None
                )

                if (
                    previous is None
                    or previous <= 0
                    or probability is None
                    or abs(probability - previous)
                    > self._relative_error * previous
                ):
                    stable_estimates = False

            if enough_observations and stable_estimates:
                result["accuracy_note"] = (
                    f"Учебный критерий подбора выборки выполнен "
                    f"для вероятностей отказа всех источников: "
                    f"δ={self._relative_error:.0%}, "
                    f"доверительная вероятность={self._confidence:.0%}. "
                    f"Итераций: {iteration}. "
                    f"Seed итогового запуска: {config.seed}."
                )
                result["precision_reached"] = True
                return result

            if request_count >= self._max_requests:
                reason = "достигнут предел числа заявок"
                break

            if iteration == self._max_iterations:
                reason = "достигнут предел числа итераций"
                break

            previous_probabilities = current_probabilities

            # Если необходимый объём уже набран, но оценки нестабильны,
            # сохраняется увеличение минимум на 50%.
            request_count = min(next_count, self._max_requests)

        result["accuracy_note"] = (
            f"Точность по учебному критерию не подтверждена: {reason}. "
            f"Показаны результаты последнего завершённого запуска. "
            f"Итераций: {iteration}. "
            f"Seed итогового запуска: {config.seed}."
        )
        result["precision_reached"] = False
        return result


class Application:
    def __init__(self, config):
        self._config = config
        self._step_view = StepView()
        self._summary_view = SummaryView()

    def run_step_mode(self):
        simulation = Simulation(self._config)
        self._step_view.render(simulation.snapshot())

        while not simulation.is_finished():
            command = input(
                "\nEnter — следующий шаг; "
                "a — завершить без остановок; "
                "q — выйти: "
            ).strip().lower()

            if command == "q":
                print("Просмотр прерван. Моделирование не завершено.")
                return

            if command == "a":
                simulation.run()
                self._step_view.render(simulation.snapshot())
                break

            if command != "":
                print("Используй Enter, a или q.")
                continue

            simulation.step()
            self._step_view.render(simulation.snapshot())

        self._summary_view.render(simulation.results())

    def run_auto_mode(self, max_requests, max_iterations):
        runner = AutomaticRunner(
            base_config=self._config,
            max_requests=max_requests,
            max_iterations=max_iterations,
        )

        # Промежуточные состояния и таблицы не печатаются.
        results = runner.run()
        self._summary_view.render(results)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Модель сервиса конвертации файлов. Вариант 26."
    )

    parser.add_argument(
        "--mode",
        choices=["step", "auto"],
        default="step",
        help="step — пошаговый режим, auto — автоматический",
    )
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--buffer", type=int, default=5)

    parser.add_argument("--arrival-min", type=float, default=1.0)
    parser.add_argument("--arrival-max", type=float, default=3.0)
    parser.add_argument("--service-mean", type=float, default=1.5)

    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help=(
            "Лимит заявок в пошаговом режиме "
            "или начальная выборка в автоматическом"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-requests", type=int, default=200_000)
    parser.add_argument("--max-iterations", type=int, default=10)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    request_limit = args.requests

    if request_limit is None:
        request_limit = 20 if args.mode == "step" else 1000

    config = SimulationConfig(
        source_count=args.sources,
        worker_count=args.workers,
        buffer_capacity=args.buffer,
        arrival_min=args.arrival_min,
        arrival_max=args.arrival_max,
        service_mean=args.service_mean,
        request_limit=request_limit,
        seed=args.seed,
    )

    try:
        config.validate()
        application = Application(config)

        if args.mode == "step":
            application.run_step_mode()
        else:
            application.run_auto_mode(
                max_requests=args.max_requests,
                max_iterations=args.max_iterations,
            )

    except ValueError as error:
        parser.error(str(error))
    except (KeyboardInterrupt, EOFError):
        print("\nРабота остановлена пользователем.")


if __name__ == "__main__":
    main()