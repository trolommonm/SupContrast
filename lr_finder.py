from main_supcon import set_loader, set_model, set_optimizer, parse_option
from torch_lr_finder import LRFinder

opt = parse_option()

# build data loader
train_loader = set_loader(opt)

# build model and criterion
model, criterion = set_model(opt)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Model Summary")
print(model)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# build optimizer
optimizer = set_optimizer(opt, model)

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=5, num_iter=401)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state