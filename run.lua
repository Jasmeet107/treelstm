require('.')

-- load vocab
local vocab = treelstm.Vocab('data/sst/vocab-cased.txt')

-- load dataset
local dataset = treelstm.read_serapis_dataset('../sentiment/', vocab)
printf('num data = %d\n', dataset.size)

-- load model and predict dataset
local model = treelstm.TreeLSTMSentiment.load("trained_models/sent-constituency.2class.1l.150d.2.th")
local predictions = model:predict_dataset(dataset)

-- save predictions
file = io.open('../sentiment/predictions.txt', 'w')
local prediction
for i = 1, predictions:size(1) do
    for j = 1, 3 do
        file:write(predictions[i][j])
        file:write(",")
    end
    file:write("\n")
end
file:close()
