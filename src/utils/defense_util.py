from enum import Enum
import matplotlib.pyplot as plt


class Metric(Enum):
    ASR = 0
    CDA = 1
    STEP = 2


def metric_dict():
    return {
        Metric.ASR: [],
        Metric.CDA: [],
        Metric.STEP: [],
    }


def plot_defense(metric_dict):
    asr = metric_dict[Metric.ASR]
    cda = metric_dict[Metric.CDA]
    step = metric_dict[Metric.STEP]

    plt.plot(step, asr, label='ASR')
    plt.plot(step, cda, label='CDA')

    # Adding title and labels
    plt.title('ASR/CDA Plot')
    plt.xlabel('step')

    # Showing legend
    plt.legend()

    # Display the plot
    plt.show()
