from torch.utils.data import Dataset
from core.utils.reader import JsonlReader
import os
from PIL import Image
from core.modules.builder import get_image_processor
class MemesDataset(Dataset):
  """ Dataset class for Hateful Memes """
  def __init__(self,config,typ='train'):
    """Init Function for the Dataset Class

    Parameters:
    ------
    config : The Config object containing configuration for data
    typ : String value specifying whether it is train,dev or test

    """

    self._config = config
    self.type = typ
    self.reader = JsonlReader(self._config.annotations.as_dict()[typ])
    self.annotations = self.reader.read()
    self.transform = get_image_processor(self._config.image_processor)

  def __len__(self):
    """Function to return the size of the data"""
    return self.reader.size
  def __getitem__(self,idx):
    record_dic=self.annotations[idx]
    img = Image.open(os.path.join(self._config.data_dir,record_dic['img'])).convert('RGB')
    text = record_dic['text']
    if self.transform:
        img = self.transform(img)
    label =record_dic['label']
    if(self.type in ['train','dev']):
      if(self._config.get_image):
        if(self._config.get_text):
          return img,text,label
        else:
          return img,label
      elif(self._config.get_text):
        return text,label
      else:
        raise Exception('Need atleast some features to return.')
        return
    else:
      if(self._config.get_image):
        if(self._config.get_text):
          return img,text
        else:
          return img
      elif(self._config.get_text):
        return text
      else:
        raise Exception('Need atleast some features')
        return
