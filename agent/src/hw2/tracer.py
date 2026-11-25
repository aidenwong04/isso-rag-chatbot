import json
import time
from pathlib import Path

class Tracer():
    def __init__(self, output_path):
        self.run_id = Path(output_path).parent.name
        self.step_count = 1
        self.usage = {"input": 0, "output": 0}
        self.file = open(output_path, mode="w", buffering=1)

    def emit(self, kind, alert_id=None,**fields):
        if kind == "model_call":
            # Every counter the call reported, cache_read included, so that
            # usage_total matches the model_call events counter by counter.
            for counter, n in fields["usage"].items():
                self.usage[counter] = self.usage.get(counter, 0) + n
    
        event = {"step":self.step_count,
                "ts": time.time(),
                "run_id": self.run_id,
                "alert_id": alert_id,
                "kind": kind
                }
        for k,v in fields.items():
            event[k] = v
        if kind == "terminal":
            event["usage_total"] = self.usage.copy()

        json.dump(event, self.file)
        self.file.write("\n")
        self.step_count += 1

    def close(self):
        self.file.close()
