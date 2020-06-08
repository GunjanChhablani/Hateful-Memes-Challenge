from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torchnlp.metrics import get_accuracy
from core.utils.configuration import configmapper

@configmapper.map('metrics','binary_auroc')
def binary_auroc(outputs,labels):
    """Function to compute Area Under ROC Curve Score

    Parameters
    ----------
    outputs: torch.Tensor
        Tensor containing the logit outputs from the model
    labels: torch.Tensor
        Tensor containing the labels (Not one-hot encoded)

    Returns
    -------
    roc_auc_score : int,
        The roc_auc_score computed between the outputs and the labels

    ## Update tips:
    ## More functionality can be added through the parameters.
    ## Custom AUROC function/class can also be defined.
    """
    outputs_index = F.softmax(outputs.detach(),dim=1)[:,1]
    try:
        return roc_auc_score(labels.detach().numpy(),outputs_index.numpy())
    except:
        return 0
@configmapper.map('metrics','accuracy')
def accuracy(outputs,labels):
    """Function to compute Accuracy Score

    Parameters
    ----------
    outputs: torch.Tensor
        Tensor containing the logit outputs from the model
    labels: torch.Tensor
        Tensor containing the labels (Not one-hot encoded)

    Returns
    -------
    accuracy_score : int
        The accuracy_score computed between the outputs and the labels

    ## Update tips:
    ## More functionality can be added through the parameters.
    ## Custom function/class can also be defined.
    """
    outputs_argmax = torch.argmax(F.softmax(outputs.detach(),dim=1),dim=1)
    return get_accuracy(labels,outputs_argmax)[0]
