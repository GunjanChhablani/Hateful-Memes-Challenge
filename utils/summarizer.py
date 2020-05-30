import os
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchviz import make_dot


class Summarizer():
    def __init__(self,model_name,data_name,dir_name,version,version_specifics,images,type ='classify'):
        """ Initializer for Summarizer Class

        #Arguments:
        model = Name of the model being used.
        data = Name of the data being used.
        version = Version of the model being used.
        version_specifics = Comments on the version of the model, i.e. Optimization method, Learning rate, Hyperparameters, etc.
        """
        self.writer = SummaryWriter(log_dir=model_name+'_'+data_name,comment = version+'_'+version_specifics)
        self.model = model_name
        self.data = data_name
        self.comment = version+'_'+version_specifics
        self.dir = dir_name
        self.model_path = os.path.join(self.dir,model_name)
        try:
            os.makedir(self.dir)
            os.makedir(self.model_path)
        except:
            print("Failed to Create Directory.")

    def save_params(self,param_list,param_name_list,combine,combine_name,epoch,batch_size,batch=None):
        if(combine==False):
            for i in range(len(param_list)):
                if(isinstance(param_list[i],Variable)):
                    param_list[i] = param_list[i].data.cpu().numpy()
                self.writer.add_scalar(param_name_list[i],param_list[i],Summarizer._global_step(epoch,batch_size,batch))

        else:
            scalar_dict = dict(zip(param_name_list,param_list))
            self.writer.add_scalars(combine_name,scalar_dict,Summarizer._global_step(epoch,batch_size,batch))


    def save_batch_images(self,image_name,image_batch,epoch,batch_size,batch=None,dataformats = 'CHW'):
        self.writer.add_images(image_name,image_batch,Summarizer._global_step(epoch,batch_size,batch),dataformats=dataformats)

    def save_prcurve(self,labels,preds,epoch,batch_size,batch=None):
        self.writer.add_pr_curve('pr_curve',labels,preds,Summarizer._global_step(epoch,batch_size,batch))

    def save_hyperparams(self,hparam_list,hparam_name_list,metric_list,metric_name_list):
        self.writer.add_hparams(zip(metric_name_list,metric_list),zip(hparam_name_list,hparam_list))

    def save_models(self,model_list,model_names_list,epoch):
        for model_name,model in zip(model_names_list,model_list):
            torch.save(model.state_dict(),os.path.join(self.model_path,model_name))

    def save_fig(self,fig,fig_name,epoch,batch_size,batch=None):
        self.writer.add_figure(fig_name,fig,SummaryWriter._global_step(epoch,batch_size,batch))

    def display_params(params_list,params_name_list,epoch,num_epochs,batch_size,batch):
        for i in range(len(params_list)):
            if isinstance(params_list[i],Variable):
                params_list[i] = params_list[i].data.cpu().numpy()
        print('Epoch: {}/{}, Batch: {}/{}'.format(epoch,num_epochs,batch,batch_size))
        for i in range(len(params_list)):
            print('{}:{}'.format(params_name_list[i],params_list[i]))

    def draw_model_architecture(model,output,input,input_name,save_name):
        make_dot(output,params = dict(list(model.named_parameters()))+[(input_name,input)])

    def close(self):
        self.writer.close()

    @staticmethod
    def _global_step(epoch,batch_size,batch):
        if(batch):
            return epoch*batch_size + batch
        else:
            return epoch
