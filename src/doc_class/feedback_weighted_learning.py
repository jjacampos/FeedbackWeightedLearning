from argparse import ArgumentParser
import torchtext
from model import MultilayerPerceptron
from sklearn.metrics import f1_score, confusion_matrix
from torch.nn.functional import softmax, log_softmax
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.distributions import Categorical


def save_model(model, optimizer, path):
    torch.save({'model': model, 
                'optimizer': optimizer}, path)

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint['model'], checkpoint['optimizer']

def load_data(train_file, val_file, test_file, embedding_dim, min_freq, data_path, prev_vocabulary=None, prev_label=None):

    print('Loading the data')

    VOCABULARY, LABELS= prev_vocabulary, prev_label
    
    if VOCABULARY == None:
        VOCABULARY = torchtext.data.Field(batch_first=True, sequential=True, use_vocab=True, tokenize = 'basic_english', tokenizer_language='en')
    if LABELS == None:
        LABELS = torchtext.data.Field(batch_first=True, sequential=False, use_vocab=True, unk_token=None)

    fields = [('document', VOCABULARY), ('labels', LABELS)]

    train, val, test = torchtext.data.TabularDataset.splits(
    path=data_path, train=train_file,
    validation=val_file, test=test_file, format='csv',
        fields= fields, skip_header=True)

    if prev_vocabulary == None:
        VOCABULARY.build_vocab(train, vectors=torchtext.vocab.GloVe(dim=embedding_dim), min_freq=min_freq)
    if prev_label == None:
        LABELS.build_vocab(train)
    
    print('Data loaded')
    print(len(train))
    print(len(val))
    print(len(test))
    return train, val, test, fields

def train_s0_system(model, optimizer, train_batches, valid_batches, vocabulary, num_labels, embedding_dim , hidden_dim, epochs, model_output_path, device):
    """S0 supervised system training with the train split

    Parameters:
    train_batches: the train batches
    valid_batches: the validation batches
    vocabulary: the vocabulary torchtext class
    num_labels: the number of labels 
    embedding_dim: the dimensionality of the embeddings
    hidden_dim: the hidden size
    epochs: the number of epochs for training
    device: the device (GPU/CPU)
    model_output_path : the output path for the model

    Returns:
    model: the best model after training for the specified epochs 
   """    

    print('S0 supervised learning')

    criterion = torch.nn.CrossEntropyLoss().to(device)

    model.to(device)

    prev_f1 = 0.0
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, train_batch in enumerate(train_batches):
            optimizer.zero_grad()
            x, y = train_batch.document, train_batch.labels
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print('The training loss of {} epoch is :{}'.format(epoch +1, train_loss/len(train_batches)))

        #Evaluate after each epoch
        f1 = eval_system(model, valid_batches, 'VALID')
        if f1 > prev_f1:
            save_model(model, optimizer, model_output_path)
            prev_f1 = f1
    
    return load_model(model_output_path, device)


def deployment_supervised_learning(model, optimizer, deployment_batches, valid_batches, vocabulary, num_labels, epochs, model_output_path, device):
    """Supervised learning with deployment batches 

    Parameters:
    deployment_batches: the deployment batches
    valid_batches: the validation batches
    vocabulary: the vocabulary torchtext class
    num_labels: the number of labels 
    epochs: the number of epochs for training
    device: the device (GPU/CPU)
    model_output_path : the output path for the model

    Returns:
    model: the best model after training for the specified epochs 

   """    
    print('DEPLOYMENT SUPERVISED LEARNING')
    writer = SummaryWriter()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.to(device)

    prev_f1 = 0.0
    for epoch in range(epochs):
        train_loss, step_loss = 0.0, 0.0
        model.train()
        for i, deployment_batch in enumerate(deployment_batches):
            optimizer.zero_grad()
            x, y = deployment_batch.document, deployment_batch.labels
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step_loss += loss.item()


        #Evaluate after each epoch
        f1 = eval_system(model, valid_batches, 'VALID')
        if f1 > prev_f1:
            torch.save(model, model_output_path)
            prev_f1 = f1

        writer.add_scalar('loss', train_loss/len(deployment_batches), epoch+1)
        writer.add_scalar('f1', f1, epoch +1)
        print('The training loss of {} epoch is :{}'.format(epoch +1, train_loss/len(deployment_batches)))

    
    return torch.load(model_output_path)


def eval_system(model, eval_batches, eval_type):
    """Evaluate system 

    Parameters:
    model: the model to evaluate
    eval_batches: the evaluation data batched
    eval_type: string that specifies if is validation or test

    Returns:
    f1: the f1 score in the evaluation batches

   """    
    model.eval()
    true_labels = []
    pred_labels = []
    all_weights = []
    all_ps = []
    all_softmax = []
    with torch.no_grad():
        for eval_batch in eval_batches:
            x, y = eval_batch.document, eval_batch.labels
            preds = model(x)
            preds_argmax = preds.cpu().argmax(1).numpy()
            for pred, true in zip(preds_argmax, y.cpu().numpy()):
                true_labels.append(true)
                pred_labels.append(pred)

    fscore = f1_score(true_labels, pred_labels, average='micro')
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    print('------------------------{}-------------------------'.format(eval_type))
    print('F1: {}'.format(fscore))
    
    if eval_type == 'TEST':
        print('Confusion matrix:\n {}'.format(conf_matrix))
        
    return fscore

def sample(num_samples, lambda_value, logits, num_labels):
    """Sample num_samples from a multinomial distribution

    Parameters:
    num_samples: the amount of samples to get
    lambda_value: lambda value used fo the smoothing
    logits: the logits of the NN
    num_labels: amount of posible labels

    Returns:
    samples: (batch_size x num_samples) tensor with the indexes of the samples
    q: (batch_size x num_samples) tensor with the softmax probabilities of the samples
    q_star: (batch_size x num_samples) tensor with the smoothed softmax probabilities of the samples
    log_q:(batch_size x num_samples) tensor with the logsoftmax probabilities of the samples

   """    
    #Softmax -- q dim (batch_size x labels)
    q = softmax(logits, dim=1)
    #For numerical stability avoid softmax+log -> directly logsoftmax 
    log_q = log_softmax(logits, dim=1)
    #Uniform distribtuion for smoothing
    q_star = lambda_value * q + (1-lambda_value) * (1/num_labels)
    #Create categorical distribution parametrized by smoothed probs
    m = torch.distributions.categorical.Categorical(probs=q_star)

    #Sample from distribution -- samples dim (num_samples x batch_size)
    samples = m.sample_n(num_samples)
    #Transpose (batch_size x num_samples)
    samples.transpose_(1,0)

    #Select the values of the sampled indexes -- q, q_star, log_q dim (batch_size x num_samples)
    q = torch.gather(q, 1, samples)
    q_star = torch.gather(q_star, 1, samples)
    log_q = torch.gather(log_q, 1, samples)
    
    return samples, q, q_star, log_q
    
def get_p_star(samples, y, beta):
    """Get the values for the p_star distribution

    Parameters:
    samples: tensor of dim (batch_size x num_samples) with the samples indexes
    y: tensor of dim (batch_size) with the true label for each example
    beta: the strength of the feedback

    Returns:
    p_star: tensor of dim(batch_size x num_samples ) with +beta for correct samples and -beta for incorrect ones. 
   """    
    #Convert y to (batch_size x 1) dimensionality for comparison with samples
    y = y.unsqueeze(-1)
    positives = (samples==y).float()*beta
    negatives = torch.logical_not(samples==y).float()*-beta
    p_star = positives + negatives
    return p_star

def get_loss(num_samples, y, lambda_value, logits, num_labels, beta, device):
    """Get the loss for the feedback weighted sampling

    Parameters:
    num_samples: the amount of samples to get
    y: tensor of dim (batch_size) with the true label for each example
    lambda_value: lambda value used fo the smoothing
    logits: the logits of the NN
    num_labels: amount of posible labels
    beta: the strength of the feedback

    Returns:
    loss: the loss for the feedback weighted learning
    """
    samples, q,  q_star, log_q = sample(num_samples, lambda_value, logits, num_labels)
    p_star = get_p_star(samples, y, beta)

    w_log = p_star - torch.log(q_star)
    self_norm_w_log = torch.log(torch.sum(torch.exp(p_star)/q_star, 1)).unsqueeze(1)
    w_log -=self_norm_w_log

    #Get the mean over samples and over batch
    loss = -torch.mean(torch.mean(torch.exp(w_log) * log_q, 1))
    
    return loss

def feedback_weighted_learning(model, optimizer, deployment_batches, valid_batches, vocab, num_labels, num_samples, lambda_value, feedback_model_path, beta, device, epochs):

    """Get the loss for the feedback weighted sampling

    Parameters:
    num_samples: the amount of samples to get
    y: tensor of dim (batch_size) with the true label for each example
    lambda_value: lambda value used fo the smoothing
    logits: the logits of the NN
    num_labels: amount of posible labels
    beta: the strength of the feedback

    Returns:
    loss: the loss for the feedback weighted learning
    """    
    print('FEEDBACK WEIGHTED LEARNING')
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    best_f1 = 0.0

    #Do training
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, deployment_batch in enumerate(deployment_batches):
            
            optimizer.zero_grad()
            x, y = deployment_batch.document, deployment_batch.labels
            loss = get_loss(num_samples, y, lambda_value, model(x), num_labels, beta, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        f1 = eval_system(model, valid_batches, 'VALID')
        
        writer.add_scalar('loss', train_loss/len(deployment_batches), epoch+1)
        writer.add_scalar('f1', f1, epoch +1)
                
        print('The deployment loss of {} epoch is :{}'.format(epoch +1, train_loss/len(deployment_batches)))
        if f1 >= best_f1:
            torch.save(model, feedback_model_path)
            best_f1 = f1

    writer.close()
    #Changed for hyperparameter optimization
    return best_f1, torch.load(feedback_model_path)

        
def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--num_samples',
        default=3,
        type=int,
        help='Number of samples for the system to take into account. Preferably num_samples << num_classes')

    parser.add_argument(
        '--train_file',
        default='train.csv',
        type=str,
        help='Name of the train file')

    parser.add_argument(
        '--deployment_file',
        default='deployment.csv',
        type=str,
        help='Name of the deployment file')
    
    parser.add_argument(
        '--test_file',
        default='test.csv',
        type=str,
        help='Name of the test file')

    parser.add_argument(
        '--val_file',
        default='val.csv',
        type=str,
        help='Name of the validation file')

    parser.add_argument(
        '--s0_epochs',
        default=50,
        type=int,
        help='Number of ecpochs for system 0 training')

    parser.add_argument(
        '--batch_size',
        default=256,
        type=int,
        help='Batch size')

    parser.add_argument(
        '--output_dir',
        default=None,
        type=str,
        help='Output directory for the model')

    parser.add_argument(
        '--s0_system_path',
        default=None,
        type=str,
        required=False,
        help='Path to the s0 system. Here the system expects a model.bin file and Vocab.Field torchtext file with the vocabulary.')

    parser.add_argument(
        '--embedding_dim',
        default=300,
        type=int,
        required=False,
        help='Embedding dimensionality')

    parser.add_argument(
        '--hidden_dim',
        default=200,
        type=int,
        required=False,
        help='Hidden dimentionality size.')

    parser.add_argument(
        '--min_freq',
        default=3,
        type=int,
        required=False,
        help='Min frequency for vocabulary building')

    parser.add_argument(
        '--lambda_value',
        default=0.97,
        type=float,
        required=False,
        help='Lambda value for sampling')

    parser.add_argument(
        '--beta',
        default=76,
        type=float,
        required=False,
        help='Beta feedback value')

    parser.add_argument(
        '--data_path',
        default='../../data/doc_class_splits', 
        type=str,
        help='Path to the training data')

    parser.add_argument(
        '--deployment_epochs',
        type=int,
        required=False,
        default=50,
        help='The epochs on the deployment data')

    parser.add_argument(
        '--do_deployment_supervised',
        action="store_true", 
        help='Flag for specifiying if supervised training is wanted on the deployment data')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') 
    #In case the S0 system is not specified train a new one
    if args.s0_system_path == None:
        train_dataset, valid_dataset, test_dataset, fields = load_data(args.train_file, args.val_file, args.test_file, args.embedding_dim,  args.min_freq, args.data_path)
        vocab = fields[0][1]
        label = fields[1][1]

        print(len(label.vocab))
        #Save the vocabulary and the labels binaries for future executions
        torch.save(vocab, args.output_dir + 'vocab.pt')
        torch.save(label, args.output_dir + 'labels.pt')

        #Get the data iterators 
        train_batches, valid_batches, test_batches = torchtext.data.BucketIterator.splits(
            datasets=(train_dataset, valid_dataset, test_dataset),
            batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
            sort_key=lambda x: len(x.document),
            device=device,
            sort_within_batch=True)

        model = MultilayerPerceptron(len(vocab.vocab), vocab.vocab.vectors, args.embedding_dim, args.hidden_dim, len(label.vocab))

        optimizer = torch.optim.Adam(model.parameters())

        #Train the s0 system
        s0_system, s0_optimizer = train_s0_system(model, optimizer, train_batches, valid_batches, vocab, len(label.vocab) , args.embedding_dim, args.hidden_dim, args.s0_epochs, args.output_dir + 's0_system.pt', device)
        fscore = eval_system(s0_system, test_batches, 'TEST')
        args.s0_system_path = args.output_dir
    #Load the previously trained system
    else:
        vocab = torch.load(args.s0_system_path + 'vocab.pt')
        label= torch.load(args.s0_system_path + 'labels.pt')
        model, optimizer = load_model(args.s0_system_path + 's0_system.pt', device)

    #Load deployment dataset
    deployment_dataset, valid_dataset, test_dataset, _ = load_data(args.deployment_file, args.val_file, args.test_file, args.embedding_dim,  args.min_freq, args.data_path, vocab, label)

    #Batch the data
    deployment_batches, valid_batches, test_batches = torchtext.data.BucketIterator.splits(
        datasets=(deployment_dataset, valid_dataset, test_dataset),
        batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.document),
        device=device,
        sort_within_batch=True)

    best_f1, deployment_feedback_model = feedback_weighted_learning(model, optimizer, deployment_batches, valid_batches, vocab, len(label.vocab), args.num_samples, args.lambda_value, args.output_dir + 'feedback_system.pt', args.beta, device, args.deployment_epochs)

    if args.do_deployment_supervised:
        #Load the s0 system again
        model, optimizer = load_model(args.s0_system_path + 's0_system.pt', device)
        
        deployment_supervised_model = deployment_supervised_learning(model, optimizer, deployment_batches, valid_batches,vocab, len(label.vocab) , args.deployment_epochs, args.output_dir + 'after_deployment_normal_train.pt', device)
        
        print('Deployment with supervised evaluation')
        _ = eval_system(deployment_supervised_model, test_batches, 'TEST')

    print('S0 System evaluation')
    s0_system, _ = load_model(args.s0_system_path + 's0_system.pt', device)
    _ = eval_system(s0_system, test_batches, 'TEST')
    
    print('Deployment with feedback evaluation')
    _ = eval_system(deployment_feedback_model, test_batches, 'TEST')



if __name__== "__main__":
    main()
  
 
