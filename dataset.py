import pandas as pd
import numpy as np
from PIL import Image


class CancerDatasetVal(Dataset):
    def __init__(self, datafolder, names=None, datatype='train', idx=[], transform = transforms.Compose([transforms.CenterCrop(48),transforms.ToTensor()]), labels_dict={}):
        self.datafolder = datafolder
        self.datatype = datatype
        if datatype == 'test':
            self.image_files_list = [s for s in os.listdir(datafolder)]

        self.names = (np.squeeze(names)).tolist()
        self.transform = transform
        self.labels_dict = labels_dict
        if self.datatype == 'train':
            self.labels = [labels_dict[i] for i in self.names]
 
        else:
            self.image_files_list = [self.image_files_list[i] for i in idx]
            self.labels = [0 for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name_short =  self.names[idx] 
            img_name = os.path.join(self.datafolder, '{}.tif'.format(img_name_short)) 
            img = Image.open(img_name)
            image = self.transform(img)  
        
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = self.transform(image=img)
            image = image['image']
            img_name_short = self.image_files_list[idx].split('.')[0]
        
        if self.datatype == 'train':
            label = self.labels_dict[img_name_short]
        else:
            label = 0
        return image, label


class CancerDataset(Dataset):
    def __init__(self, datafolder, names, datatype='train', idx=[], transform = transforms.Compose([transforms.CenterCrop(48),transforms.ToTensor()]), labels_dict={}):
        self.datafolder = datafolder
        self.datatype = datatype
        self.names = (np.squeeze(names)).tolist()
        self.transform = transform
        self.labels_dict = labels_dict
        if self.datatype == 'train':
            self.labels = [labels_dict[i] for i in self.names]
            self.dict_train = self.balance_train()
            self.labels_train = list(self.dict_train.keys())
        else:
            self.labels = [0 for _ in range(len(self.image_files_list))]

    def __len__(self):
        return len(self.labels)

    def balance_train(self):
        dict_train = {}
        for name, label in zip(self.names, self.labels):
            if not label in dict_train.keys():
                dict_train[label] = [name]
            else:
                dict_train[label].append(name)                
        return dict_train
    
    def __getitem__(self, idx):
        #print(idx)
        label = self.labels_dict[self.names[idx]]
        names_label = self.dict_train[label]
        nums = len(names_label)
        if nums == 1:
            anchor_name = names_label[0]
            positive_name = names_label[0]
        else:
            positive_name = random.sample(names_label, 1)
        anchor_name = self.names[idx]        
        while(anchor_name == positive_name):
            positive_name = random.sample(names_label, 1)

        negative_label = random.choice(list(set(self.labels_train) ^ set([label])))
        negative_name = random.choice(self.dict_train[negative_label])        

        anchor_image, anchor_label = self.get_image(anchor_name)
        positive_image, positive_label = self.get_image(positive_name[0])
        negative_image,  negative_label = self.get_image(negative_name)

        assert anchor_name != negative_name
        assert anchor_label != negative_label
        assert anchor_label == positive_label
        return [anchor_image, positive_image, negative_image], \
               [anchor_label, positive_label, negative_label]

        
    def get_image(self, idx):
        img_name = os.path.join(self.datafolder, '{}.tif'.format(idx)) 
        img = Image.open(img_name)
        image = self.transform(img)
        img_name_short =  idx    
        if self.datatype == 'train':
            label = self.labels_dict[img_name_short]
        else:
            label = 0
        return image, label

size = 224    
data_transforms = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5), 
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3), 
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), 
    ]),
    transforms.RandomChoice([
        transforms.RandomRotation((0,0)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomRotation((90,90)),
        transforms.RandomRotation((180,180)),
        transforms.RandomRotation((270,270)),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((90,90)),
        ]),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((270,270)),
        ]) 
    ]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

data_transforms_test = albumentations.Compose([
    albumentations.Resize(224, 224),
    albumentations.Normalize(),
    AT.ToTensor()
    ])

def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels


def return_tumor_or_not(dic, one_id):
    return dic[one_id]

def create_dict(root_path):
    df = pd.read_csv(f"{root_path}/data/train_labels.csv")
    result_dict = {}
    for index in range(df.shape[0]):
        one_id = df.iloc[index,0]
        tumor_or_not = df.iloc[index,1]
        result_dict[one_id] = int(tumor_or_not)
    return result_dict

def find_missing(train_ids, cv_ids, root_path):
    all_ids = set(pd.read_csv(f"{root_path}/data/train_labels.csv")['id'].values)
    wsi_ids = set(train_ids + cv_ids)

    missing_ids = list(all_ids-wsi_ids)
    return missing_ids


def generate_split(root_path):
    ids = pd.read_csv(f"{root_path}/data/patch_id_wsi.csv")
    wsi_dict = {}
    for i in range(ids.shape[0]):
        wsi = ids.iloc[i,1]
        train_id = ids.iloc[i,0]
        wsi_array = wsi.split('_')
        number = int(wsi_array[3])
        if wsi_dict.get(number) is None:
            wsi_dict[number] = [train_id]
        else:
            wsi_dict[number].append(train_id)

    wsi_keys = list(wsi_dict.keys())
    np.random.seed()
    np.random.shuffle(wsi_keys)
    amount_of_keys = len(wsi_keys)

    keys_for_train = wsi_keys[0:int(amount_of_keys*0.8)]
    keys_for_cv = wsi_keys[int(amount_of_keys*0.8):]
    train_ids = []
    cv_ids = []

    for key in keys_for_train:
        train_ids += wsi_dict[key]

    for key in keys_for_cv:
        cv_ids += wsi_dict[key]

    dic = create_dict(root_path)

    missing_ids = find_missing(train_ids, cv_ids, root_path)
    missing_ids_total = len(missing_ids)
    np.random.seed()
    np.random.shuffle(missing_ids)

    train_missing_ids = missing_ids[0:int(missing_ids_total*0.8)]
    cv_missing_ids = missing_ids[int(missing_ids_total*0.8):]

    train_ids += train_missing_ids
    cv_ids += cv_missing_ids

    train_labels = []
    cv_labels = []

    train_tumor = 0
    for one_id in train_ids:
        temp = return_tumor_or_not(dic, one_id)
        train_tumor += temp
        train_labels.append(temp)

    cv_tumor = 0
    for one_id in cv_ids:
        temp = return_tumor_or_not(dic, one_id)
        cv_tumor += temp
        cv_labels.append(temp)
    total = len(train_ids) + len(cv_ids)

    print("Amount of train labels: {}, {}/{}".format(len(train_ids), train_tumor, len(train_ids)-train_tumor))
    print("Amount of cv labels: {}, {}/{}".format(len(cv_ids), cv_tumor, len(cv_ids) - cv_tumor))
    print("Percentage of cv labels: {}".format(len(cv_ids)/total))

    return train_ids, cv_ids, train_labels, cv_labels
