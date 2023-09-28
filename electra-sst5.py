import pandas as pd
import torch
from transformers import ElectraTokenizer,ElectraForSequenceClassification,get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import wandb
import random
import numpy as np
import pytreebank
import os
from sklearn.model_selection import train_test_split

sweep_config = {
    'method': 'grid', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-5, 1e-3, 1e-2]
        },
        'batch_size': {
            'values': [16,32]
        },
        'epochs':{
            'values':[30,40,50]
        }
    }
}

sweep_defaults = {
    'learning_rate': 1e-5,
    'batch_size': 16,
    'epochs':30
}

sweep_id = wandb.sweep(sweep_config, project = "ELECTRA-SST5", entity="madhudorai24")

data = pytreebank.load_sst("/kaggle/input/stanford-sentiment-treebank-v2-sst2/SST2-Data/SST2-Data/trainDevTestTrees_PTB/trees/")
out_dir = "/kaggle/working/"
os.makedirs(out_dir, exist_ok=True)  # Create the directory if it doesn't exist
out_path = os.path.join(out_dir, "sst_{}.txt")
for cat in ['train','test','dev']:
    with open(out_path.format(cat),"w") as file:
        for item in data[cat]:
            file.write("__label__{}\t{}\n".format(
                item.to_labeled_lines()[0][0] +1,
                item.to_labeled_lines()[0][1]
            ))
    
    print("done with {}".format(file))

train = pd.read_csv("/kaggle/working/sst_train.txt",sep="\t",header=None,names=['label','text'])
train['label'] = train['label'].str.replace("__label__","")
train['label'] = train['label'].astype(int).astype("category")
traindf, validdf = train_test_split(train, test_size=0.2, random_state=42)
testdf = pd.read_csv("/kaggle/working/sst_test.txt",sep="\t",header=None,names=['label','text'])
testdf['label'] = testdf['label'].str.replace("__label__","")
testdf['label'] = testdf['label'].astype(int).astype("category")

tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")

def dataset(df):
    sentences = df.text.values
    labels = df.label.values
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            truncation = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

train_ds = dataset(traindf)
val_ds = dataset(validdf)
test_ds = dataset(testdf)

def ret_dataloader():
    batch_size = wandb.config.batch_size
    train_dataloader = DataLoader(
                train_ds,  # The training samples.
                sampler = RandomSampler(train_ds), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_ds, # The validation samples.
                sampler = SequentialSampler(val_ds), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    test_dataloader = DataLoader(
                test_ds, # The test samples.
                sampler = SequentialSampler(test_ds), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    return train_dataloader,validation_dataloader, test_dataloader

#TRANSFORMER MODEL
def ret_model():
    model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator", num_labels=5, return_dict = True)
    return model

def ret_optim(model):
    optimizer = torch.optim.SGD(model.parameters(),
                      lr = wandb.config.learning_rate, 
                      momentum=0)
    return optimizer

def ret_scheduler(train_dataloader,optimizer):
    epochs = wandb.config.epochs
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps = 0, num_training_steps = total_steps)
    return scheduler

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_valid_test():
    wandb.init(config=sweep_defaults)
    sweep_NAME = wandb.run.name
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ret_model()
    model.to(device)
    params_to_include = ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    for name, param in model.named_parameters():
        if name in params_to_include:
         param.requires_grad = True
        else:
         param.requires_grad = False

    train_dataloader,validation_dataloader, test_dataloader = ret_dataloader()
    optimizer = ret_optim(model)
    scheduler = ret_scheduler(train_dataloader,optimizer)

    print(wandb.config)

    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    epochs = wandb.config.epochs
    best_val_accuracy = 0.0
    # For each epoch...
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        total_correct =0
        total_samples =0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            outputs = model(b_input_ids,
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct = (preds == b_labels).sum().item()
            total_correct += correct
            total_samples += len(b_labels)

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = total_correct / total_samples         
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        wandb.log({'avg_train_loss':avg_train_loss, 'train_accuracy': train_accuracy})

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training accuracy: {0:.2f}".format(train_accuracy))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        val_accuracy =0 
        total_eval_loss = 0
        val_correct = 0
        val_total_samples = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
    
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                val_outputs = model(b_input_ids, 
                                      token_type_ids=None, 
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                val_loss = val_outputs.loss
                val_logits = val_outputs.logits
                total_eval_loss += val_loss.item()

                val_preds = val_logits.argmax(dim=1)
                val_correct += (val_preds == b_labels).sum().item()
                val_total_samples += len(b_labels)
            
        # Report the final accuracy for this validation run.
        avg_eval_loss = total_eval_loss / len(validation_dataloader) 
        val_accuracy = val_correct/val_total_samples 
        print(" Average Validation loss: {0:.2f}".format(avg_eval_loss))
        print(" Validation Accuracy: {0:.2f}".format(val_accuracy))
        wandb.log({'eval_loss':avg_eval_loss, 'val_acc':val_accuracy})
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training accuracy': train_accuracy,
                'Valid. Loss.': avg_eval_loss,
                'Valid.accuracy': val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        
        #save the model if its the best epoch
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_file = f"best_model{sweep_NAME}_epoch{epoch_i}.pt"
            torch.save(model.state_dict(), checkpoint_file)

    print("")
    print("Training complete for 1 model!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

     # ========================================
        #               Testing
        # ========================================
    best_model = ret_model()
    best_model.load_state_dict(torch.load(checkpoint_file))
    best_model.to(device)
    # Set the model to evaluation mode
    best_model.eval()

    # Tracking variables for evaluation
    total_test_loss = 0
    test_correct = 0
    test_total_samples = 0

    # Evaluate data on the test dataset
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            test_outputs = best_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            test_loss = test_outputs.loss
            test_logits = test_outputs.logits
            total_test_loss += test_loss.item()

            test_preds = test_logits.argmax(dim=1)
            test_correct += (test_preds == b_labels).sum().item()
            test_total_samples += len(b_labels)

    # Calculate and log test accuracy
    test_accuracy = test_correct / test_total_samples
    wandb.log({'test_accuracy': test_accuracy})
    print("Test Accuracy: {:.2f}".format(test_accuracy))

wandb.agent(sweep_id,function=train_valid_test)
wandb.finish()
