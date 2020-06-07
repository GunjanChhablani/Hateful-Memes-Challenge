import matplotlib.pylot as plt
from torchvision.utils import make_grid

def get_batch_grid(batch,nrow):
    return make_grid(batch).permute(1,2,0)
