import json
import matplotlib.pyplot as plt


def load_json_arr(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
        return json_data

experiment_metrics = load_json_arr('C:/Users/EaglesonLabs/detectron object detection/output/metrics.json')


for experiment in experiment_metrics:
    print(experiment)
plt.plot(
    [x['iteration'] for x in experiment_metrics], 
    [x['bbox/AP'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics], 
    [x['bbox/AP-radiolucent'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics], 
    [x['bbox/AP-radiopaqu'] for x in experiment_metrics])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['bbox/AP', 'bbox/AP-radiolucent', 'bbox/AP-radiopaqu','validation_loss'], loc='upper left')
# Set the title and axis labels
plt.title("faster_rcnn_X_101_32x8d_FPN_3x")
plt.xlabel("Iteration")
plt.ylabel("Loss / Accuracy")
plt.show()