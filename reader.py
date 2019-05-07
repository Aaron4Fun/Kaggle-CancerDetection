from . import dataset

def read_data(root_path, batch_size, num_workers):
    
    labels = pd.read_csv(f'{root_path}/data/train_labels.csv')
    img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}
    
    train_ids, cv_ids, train_labels, cv_labels = generate_split(root_path)

    dataset_train = CancerDataset(datafolder=f'{root_path}/data/train/', names=train_ids, datatype='train', transform=data_transforms, labels_dict=img_class_dict)
    dataset_val = CancerDatasetVal(datafolder=f'{root_path}/data/train/', names=cv_ids, datatype='train', transform=data_transforms, labels_dict=img_class_dict)

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=num_workers, collate_fn=train_collate)
    valid_loader = torch.utils.data.DataLoader(dataset_val, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, valid_loader

def read_test(root_path, batch_size, num_workers, data_transforms):
    test_idx = [i for i in range(len(os.listdir(f"{root_path}/data/test")))]
    test_set = CancerDatasetVal(datafolder=f'{root_path}/data/test/', idx=test_idx, datatype='test', transform=data_transforms)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    return test_loader