import time


class PerformanceLogger:

    def __init__(self):

        self.times = {}

    def start(self):

        self.times["start"] = time.perf_counter()

    def checkpoint(self, name):

        self.times[name] = time.perf_counter()

    def result(self):

        start = self.times.get("start", 0)

        vision = self.times.get("vision", start)

        database = self.times.get("database", vision)

        llm = self.times.get("llm", database)

        return {

            "vision": round(
                vision - start,
                3
            ),

            "database": round(
                database - vision,
                3
            ),

            "llm": round(
                llm - database,
                3
            ),

            "total": round(
                llm - start,
                3
            )

        }