require('.')

-- load vocab
local vocab = treelstm.Vocab('data/sst/vocab-cased.txt')

-- load dataset
local dataset = treelstm.read_serapis_dataset('data/serapis/', vocab)
printf('num data = %d\n', dataset.size)

-- load model and predict dataset
local model = treelstm.TreeLSTMSentiment.load("trained_models/sent-constituency.2class.1l.150d.2.th")
local predictions = model:predict_dataset(dataset)
print(predictions)
