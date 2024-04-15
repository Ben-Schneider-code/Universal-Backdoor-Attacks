
import torch


class DistillationLoss(torch.nn.Module):
    def __init__(self):
        """
        Initialize the DistillationLoss module.
        """
        super(DistillationLoss, self).__init__()

    def forward(self, y_student, y_teacher, temperature=1):
        y_student_soft = torch.nn.functional.softmax(y_student / temperature, dim=1)
        y_teacher_soft = torch.nn.functional.softmax(y_teacher / temperature, dim=1)

        # Compute the cross-entropy loss
        loss = -torch.mean(torch.sum(y_teacher_soft * torch.log(y_student_soft + 1e-7), dim=1))

        return loss
