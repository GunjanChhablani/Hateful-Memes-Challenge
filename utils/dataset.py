from torch.utils.data import Dataset, DataLoader
import jsonlines
from PIL import Image

class MemesDataset(Dataset):
  """ Dataset class for Hateful Memes """
  def __init__(self,json_name,data_path,transform,get_image = True, get_text=False,training=True):
    """Init Function for the Dataset Class

    Params:
    ------
    json_name : Name of json_file from where the data is to be loaded.
    data_path : Path to the main_folder where the data is stored.
    transform : Any specific transform to be applied to the image data.
    get_image : Boolean value specifying whether to output the image along with the label or not.
    get_text : Boolean value specifying whether to output the text along with the label or not.
    training : Boolean value specifying where the data to be generated is for training.

    """

    assert get_image in ['True','False'], 'get_image should be a boolean'
    assert get_text in ['True','False'],'get_text should be a boolean'
    assert training in ['True','False'], 'training should be a boolean'

    self.reader = jsonlines.open(os.path.join(data_path,json_name),'r')
    self.default_transform = default_transform
    self.data_path = data_path
    self.training = training
    self.out_typ = int(get_image)+2*int(get_text) ## Just a toy encoding, can use numbers or types too for the definition.

  def __getitem__(self,idx):
    record_dic=reader.read()
    img = Image.open(os.path.join(record_dic['img']))
    text = record_dic['text']
    label = torch.LongTensor(record_dic['label'])
    img = self.transform(img)

    if(self.training):
      if(self.out_typ==0):
        raise('Need atleast one set of features')
        return
      elif(self.out_typ==1):
        return img,label

      elif(self.out_type==2):
        return text,label
      else:
        return img,text,label
    else:
      if(self.out_typ==0):
        raise('Need atleast one set of features')
        return
      elif(self.out_typ==1):
        return img

      elif(self.out_type==2):
        return text
      else:
        return img,text
