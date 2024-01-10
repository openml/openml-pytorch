import openml
from openml.tasks import OpenMLClassificationTask

task = openml.tasks.get_task(361175)
openml.config.apikey = 'd7f058387fb3c8ba41e1ae61ebd999a0'
# Define task parameters
task_type = openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
evaluation_measure = 'predictive_accuracy'
estimation_procedure = {
    'type': 'crossvalidation',
    'parameters': {
        'number_repeats': '1',
        'number_folds': '10',
        'percentage': '',
        'stratified_sampling': 'true'
    },
    'data_splits_url': 'https://api.openml.org/api_splits/get/361175/Task_361175_splits.arff'
}
target_name = 'CATEGORY'
class_labels = ['Adrenal_gland', 'Bile-duct', 'Bladder', 'Breast', 'Cervix', 'Colon', 'Esophagus', 'HeadNeck', 'Kidney', 'Liver', 'Lung', 'Ovarian', 'Pancreatic', 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus']
cost_matrix = None

# 'split': <openml.tasks.split.OpenMLSplit object at 0x7efca59476a0>

# Create the task
new_task = openml.tasks.create_task(
    task_type=task_type,
    dataset_id=task.dataset_id, 
    estimation_procedure_id = task.estimation_procedure_id,
    # estimation_procedure=estimation_procedure,
    target_name=target_name,
    class_labels=class_labels,
    cost_matrix=cost_matrix
)

print(new_task)

new_task.publish()