#working on deployment of our model

import torch
import json
import os
from transformers import DistilBertTokenizer,DistilBertModel

MAX_LEN = 512

#during training pytorch saves the weight , but when we load the model from saved , it is required to say what types of model it is. One it is done pytorch can auto detect about weight.

class DistilBERTClass(torch.nn.Module):#this class hold our customized model

    def __init__(self):
        
        super(DistilBERTClass,self).__init__()

        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.pre_classifier = torch.nn.Linear(768,768)

        self.dropout = torch.nn.Dropout(0.3) 

        self.classifier = torch.nn.Linear(768,4)

    #defining forward pass
    def forward(self,input_ids,attention_mask):
        
        output_1 = self.l1(input_ids = input_ids, attention_mask = attention_mask)

        hidden_state = output_1[0]

        pooler = hidden_state[:,0]
        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)#the output is logit it is raw number.

        return output

def model_fn(model_dir): #this function will use to load model from save stage.

    print("Loading model from: ", model_dir)#model is at S3 but it will download to sagemaker.

    model = DistilBERTClass()

    model_state_dict = torch.load(os.path.join(model_dir, 'pytorch_distilbert_news.bin'),map_location = torch.device('cpu'))#loading the weights, as not using it for training in inference no need gpu

    model.load_state_dict(model_state_dict)

    return model


def input_fn(request_body, request_content_type): #it handle the pre processing steps to to formate that is accepted by the model.
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        sentence = input_data['inputs']
        return sentence
    else:
        raise ValueError(f"Unsupported content types: {request_content_type}")

def predict_fn(input_data,model):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')#when we want to model do inference have to tokenize it first.

    inputs = tokenizer(input_data, return_tensors = "pt").to(device)#here "pt" refers to pytorch

    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(ids,mask)#pytorch is inteliggent so when calling model its actually calling model.forward

    probabilities = torch.softmax(outputs, dim = 1).cpu().numpy()#converting the raw output of model to probabilities

    class_names = ["Business", "Science", "Entertainment", "Health"]#it needs to be correct order

    predicted_class = probabilities.argmax(axis=1)[0]#argmax will tell which index its belongs to #3

    predicted_label = class_names[predicted_class]#Health

    return {'predicted_label': predicted_label, 'probabilities':probabilities.tolist()}


def output_fn(prediction,accept):#after obtaining prediction from the model this function make it into a response can be understood by users and in proper format

    if accept == "application/json":
        return json.dumps(prediction),accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
        

    
    
    
    
                          
        
    
    
