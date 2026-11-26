"""In-process operational metrics; no MRI images or patient identifiers logged."""
from threading import Lock

class Monitor:
    def __init__(self):
        self.lock=Lock()
        self.total=0
        self.errors=0
        self.latency=0.0
        self.buckets=[0]*6
        self.bounds=(.1,.25,.5,1.,2.,5.)

    def observe(self, seconds, ok):
        with self.lock:
            self.total+=1
            self.errors+=int(not ok)
            self.latency+=seconds
            for i,b in enumerate(self.bounds):
                if seconds<=b: self.buckets[i]+=1

    def snapshot(self):
        with self.lock:
            return {"requests":self.total,"errors":self.errors,
                    "error_rate":self.errors/self.total if self.total else 0,
                    "mean_latency_ms":1000*self.latency/self.total if self.total else 0,
                    "latency_sum_seconds":self.latency,"buckets":list(self.buckets)}

    def prometheus(self):
        s=self.snapshot()
        lines=["# TYPE neuroseg_predictions_total counter",
               f"neuroseg_predictions_total {s['requests']}",
               "# TYPE neuroseg_prediction_errors_total counter",
               f"neuroseg_prediction_errors_total {s['errors']}",
               "# TYPE neuroseg_prediction_duration_seconds histogram"]
        for bound,count in zip(self.bounds,s["buckets"]):
            lines.append(f'neuroseg_prediction_duration_seconds_bucket{{le="{bound}"}} {count}')
        lines.extend([f'neuroseg_prediction_duration_seconds_bucket{{le="+Inf"}} {s["requests"]}',
                      f'neuroseg_prediction_duration_seconds_count {s["requests"]}',
                      f'neuroseg_prediction_duration_seconds_sum {s["latency_sum_seconds"]}'])
        return "\n".join(lines)+"\n"
