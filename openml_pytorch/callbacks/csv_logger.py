import csv
import os
from .callback import Callback
from .recording import AvgStatsCallback

class CSVLoggerCallback(Callback):
    """
    Log training and validation metrics to a CSV file.
    """

    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_path = os.path.join(self.log_dir, f"{self.experiment_name}_logs.csv")
        self.file = None
        self.writer = None
        
        os.makedirs(self.log_dir, exist_ok=True)

    def begin_fit(self):
        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        headers = ["Epoch", "Train Loss", "Valid Loss"]
        for cb in self.run.cbs:
            if isinstance(cb, AvgStatsCallback):
                for metric in cb.train_stats.metrics:
                    metric_name = metric.__name__ if hasattr(metric, "__name__") else str(metric)
                    headers.append(f"Train {metric_name}")
                    headers.append(f"Valid {metric_name}")
                break
        headers.append("Learning Rate")
        self.writer.writerow(headers)

    def after_epoch(self):
        row = [self.epoch]
        
        train_loss = ""
        valid_loss = ""
        train_metrics = []
        valid_metrics = []
        
        for cb in self.run.cbs:
            if isinstance(cb, AvgStatsCallback):
                if len(cb.train_stats.avg_stats) > 0:
                    train_loss = float(cb.train_stats.avg_stats[0])
                if len(cb.valid_stats.avg_stats) > 0:
                    valid_loss = float(cb.valid_stats.avg_stats[0])
                    
                for i in range(1, len(cb.train_stats.avg_stats)):
                    train_metrics.append(float(cb.train_stats.avg_stats[i]))
                        
                for i in range(1, len(cb.valid_stats.avg_stats)):
                    valid_metrics.append(float(cb.valid_stats.avg_stats[i]))
                break
                
        row.append(train_loss)
        row.append(valid_loss)
        
        for i in range(max(len(train_metrics), len(valid_metrics))):
            row.append(train_metrics[i] if i < len(train_metrics) else "")
            row.append(valid_metrics[i] if i < len(valid_metrics) else "")
            
        lr = ""
        if hasattr(self.run, 'opt') and self.run.opt and len(self.run.opt.param_groups) > 0:
            lr = self.run.opt.param_groups[0]["lr"]
        row.append(lr)
        
        self.writer.writerow(row)
        self.file.flush()

    def after_fit(self):
        if self.file is not None:
            self.file.close()
