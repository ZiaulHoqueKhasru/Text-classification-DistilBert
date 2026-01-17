import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel  # here distil refers to small model
#!pip install tiktoken
from tqdm import tqdm
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import tiktoken

s3_path = 's3://sagemaker-text-classifier-uci-news-aggregation/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL',' PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[['TITLE','CATEGORY']]
my_dict = {

    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'
}


def update_catagory(x):
    return my_dict[x]

    
df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_catagory(x) if x in my_dict else x)

print(df)

#sometimes after big model training due to issue of s3 bucket it can fails which leads to time loss and resource loss, its mendatory to check with small instance first
# to do so
df = df.sample(frac=0.05, random_state=1)
df = df.reset_index(drop=True)
#Tips end here
print(df)

#encode Catagory

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)

    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))

#encode title


tokenizer = DistilBertTokenizer.from_pretrained('Distilbert-base-uncased')

#to do input data and make it tokenized simultenously declaring a class

class NewsDataset(Dataset):
    def __init__(self,dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self,index):
        title = str(self.data.TITLE[index])
        title = " ".join(title.split()) # is a cleaning trick. It removes extra spaces, tabs, or newlines to ensure the text is clean.

        inputs = self.tokenizer.encode_plus(
            title,
            None,#None is used to explicitly tell the tokenizer: "There is no second sentence."
            add_special_tokens = True,#Automatically adds the [CLS] at the start and [SEP] at the appropriate places.
            max_length = 40, #as the data set is have only title it is ok for nlp task it need to be high,Sets the limit. The final output must be exactly 40 tokens long.
            padding = 'max_length',#If the combined length is less than 40 (which is likely here), it adds "Padding Tokens" (zeros) to the end until the length hits 40.
            truncation = True, #zIf the combined length (Sentence A + Sentence B + 3 special tokens) is more than 40, it will cut off words from the end to prevent an error.
            return_token_type_ids= True,#Asks the tokenizer to create a "segment ID" mask.Sentence A gets ID 0.Sentence B gets ID 1.DistilBERT typically doesn't use these (it ignores them or treats them as zeros), but BERT uses them heavily to differentiate sentences.
            return_attention_mask = True
        )

        ids = inputs['input_ids'] # This is the list of integers representing the words.
        mask = inputs['attention_mask']# attention_mask->It stops the model from trying to find meaning in the empty padding zeros.

        return {
            'ids': torch.tensor(ids,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'targets':torch.tensor(self.data.ENCODE_CAT[index],dtype=torch.long)
        }

    def __len__(self):
        return self.len #have to return the data frame length in pytorch


train_size = 0.8

train_dataset = df.sample(frac=train_size,random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

#train_dataset.reset_index(True)#it is giving error so try this
train_dataset = train_dataset.reset_index(drop=True) 
#or-> train_dataset.reset_index(drop=True, inplace=True)



print("Full dataset: {}".format(df.shape))
print("Train dataset: {}".format(train_dataset.shape))
print(f"Test dataset: {test_dataset.shape}")

max_len = 512
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 2

training_set = NewsDataset(train_dataset, tokenizer, max_len)
testing_set = NewsDataset(test_dataset,tokenizer,max_len)

train_parameters = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
    }
test_parameters = {
    'batch_size': TEST_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
    }

training_loader = DataLoader(training_set, **train_parameters) #here, ** means taking dictionary input i mean whole dictionary input not like variable
testing_loader = DataLoader(testing_set, **test_parameters)


#Training phase: we will make a class inherite nn module, to finetune or modify existing Distilbert model, inheritacne is necessary for any custom model in pytorch

class DistilBERTClass(torch.nn.Module):#this class hold our customized model

    def __init__(self): #this is gonna be our model
        #always define the layers and other components with trainable parameter in init method for effective learning and parameter mannagement
        
        super(DistilBERTClass,self).__init__()#calling the init method of parent class which is torch.nn.Module, what is done here is that initializing parentclass and goes back to the DistilBERTClass super function doing it. In newest it can be write super().__init__().

        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')#this will load a pretrained distil bert model with uncased vocavulary , this model will serve as the backbone of custom neural network provide a powerful feature extraction. This is base model set as layer one we will built on top of it.

        self.pre_classifier = torch.nn.Linear(768,768)#[DistilBERT Output (768)] -> [Linear(768, 768)] -> [ReLU Activation] -> [Final Classifier (768 -> 4)], we adding additional weight here. $$y = xW^T + b$$, so total parameter = 768(x)*768(w)+768(b)

        self.dropout = torch.nn.Dropout(0.3) # we are dropping 30% nouron randomly for model not to be biased based on specific feature, like color, texure etc. by this model will not continuesly depend on specific high weight neuron continuously, so it will prevent overfitting.

        self.classifier = torch.nn.Linear(768,4)#it is called output logit, which provide all output class value based on training it is a raw output which is not probability distributed, later apply softmax on it.


    #defining forward pass
    def forward(self,input_ids,attention_mask):
        
        output_1 = self.l1(input_ids = input_ids, attention_mask = attention_mask)#it is calling the Distilbert class declared in innit method which will perform forward pass for input data using distilbert model

        hidden_state = output_1[0]#it is the output of final transformer layer of distil bert which is perform immidiate previously by self.l1 for input data.

        pooler = hidden_state[:,0]# this is doing some slicing, it extract the first token state for classification, in here it is [cls] token, which have knowledge of what to do for upcoming sentence, and can efficiently classify the sentence category,what we are trying accessing the output vector from transformer the cls token.

        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)#the output is logit it is raw number.

        return output



def calculate_accu(big_idx,targets): #here big index(big_idx) is the model prediting index and target is the target
    
    n_correct = (big_idx==targets).sum().item()#checking how much correct gues model doing and by .item() converting tensor to numeric number . As batch size is 4 we will see 4 row of tensor 
    '''
    [0.88,0.1,0.33,0.71 # I love The Office 1 0 0 0 target 1 0 0 0
    [0.99,0.04,0.5,0.77] # Friends is a great show 1 0 0 0 target 1 0 0 0
    [0.38,0.12,0.1,0.88] # Elon Musk lands on Mars 0 0 0 1 target 0 0 0 1
    [0.2,00.1,.7,0.55] # Breakthrough in cancer vaccine 0 0 1 0 target 0 0 1 0 #print(big_idx == targets) # tensor ([True, True, True, Truel)
    #print(big_idx = targets).sum() # tensor (4)
    print(big_idx = targets). sum(). item () # 4
    '''
    return n_correct


def train(epoch, model, device, training_loader, optimizer, loss_function):

    true_loss = 0
    numb_correct = 0
    numb_true_steps = 0
    numb_true_examples = 0
    model.train() #model will save back propagation weight update value,Dropout or BatchNorm behave differently during training vs. testing

    for _,data in enumerate(training_loader,0): #here '_' representing index and it is the loop for each epoch here 4
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device,dtype = torch.long)

        outputs = model(ids, mask)
        
        loss = loss_function(outputs,targets)
        true_loss +=loss.item() #for a particular epoch loss will be sum but for every epoc true_loss set to 0 so the cumalitive loss is for each epoch.
        big_val, big_idx = torch.max(outputs.data, dim = 1)#it will find predicted class labels based on the highest output score logits, it will found accrouss colunm also known as first dimension
        numb_correct += calculate_accu(big_idx, targets)#using index not value as targets 0,1,2,3, also torch.max will return first dimension position of max outputs will match it for total correct
        numb_true_steps += 1 # as we looked at one batch
        numb_true_examples += targets.size(0)#will get the zeroth elements for each batch we have 4 targets here so it will add so every time we run we add 4 with numb_true_examples-> simply say due to batch is 4 adding 4 row counts at a time, see accu_count function

        if _ % 5000 == 0: #100k total data /4 batch = 25000 steps %5000 = 5 output accuracy
            loss_step = true_loss / numb_true_steps
            accu_step = (numb_correct * 100 ) / numb_true_examples
            print(f"Training loss per 5000 steps: {loss_step}")
            print(f"Training accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()#during back propagation need to previously calculated gradient set to 0 , as pytorch cumalitively do it.

        loss.backward()

        optimizer.step()#it adjust the weight based on calculated gradient like adam / stocastic gradient descent.

    print(f"The total accuracy for epoch {epoch}: {(numb_correct * 100) / numb_true_examples}")
    epoch_loss = true_loss / numb_true_steps
    print(f"Training loss epoch : {epoch_loss}")

    return #not returning anything just iterating


def valid(epoch, model, testing_loader, device, loss_function):
    
    model.eval()

    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    with torch.no_grad():

        for _,data  in enumerate(testing_loader,0):# here 0 is starting position of loop
    
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device,dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
    
            outputs = model(ids,mask).squeeze()#if output have different dimension squeeze just ensure proper dimension.
    
            loss = loss_function(outputs,targets)
            tr_loss += loss.item()#true_loss += loss.item() # Summing up errors: 2.3 + 1.1 + 0.8...
            big_val, big_idx = torch.max(outputs.data, dim = 1 )
            n_correct += calculate_accu(big_idx, targets)
    
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)#as testing batch size is 2 so each time 2 row counts for output will sum
    
            if _ % 1000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100)/nb_tr_examples
                print(f"Validation loss per 1000 steps: {loss_step}")
                print(f"Validation accuracy per 1000 steps: {accu_step}" )
    
    
            epoch_loss = tr_loss / nb_tr_steps
            epoch_accu = (n_correct * 100)/nb_tr_examples
            print(f"Validation loss per Epoch: {epoch_loss} at epoch {epoch}")
            print(f"Validation accuracy epoch: {epoch_accu} at epoch {epoch}")
            
    return epoch_accu

        
def main():
    print("Start")
    #need to access the hyperparameter from training notebook to do so we have to pass on it in sage maker it can be done like this. Let it in comment as we will define here everything:
    #import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    # To parse the arguments, we usually add this line:
    args = parser.parse_args()
    
    # Example usage:
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")#Once we save this script (e.g., as train.py), you can run it from your terminal and override the defaults like this: #python train.py --epochs 50 --train_batch_size 8 --learning_rate 0.001


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBERTClass()
    model.to(device)
    LEARNING_RATE = 1e-05 # we can use arg parse value here
    optimizer = torch.optim.Adam(params = model.parameters(), lr= LEARNING_RATE)

    loss_function = torch.nn.CrossEntropyLoss()

    #Train Loop

    EPOCHS = 4

    for epoch in range (EPOCHS):
        print(f"Starting epoch: {epoch}")
        train(epoch,model,device,training_loader,optimizer,loss_function)
        valid(epoch,model,testing_loader,device,loss_function)#we are not passing optimizer here as its not required


    output_dir = os.environ['SM_MODEL_DIR']#to save output in sage make it is mandatory cant change any name, as it is sagemaker designated output directory

    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')#at the end .bin is mandatoryly required

    output_vocab_file = os.path.join(output_dir,'vocab_distilbert_news.bin')

    torch.save(model.state_dict(),output_model_file)

    tokenizer.save_vocabulary(output_vocab_file)#will save model used vocabolary
    
        
if __name__ == '__main__':
    main()
